"""Chunk một file Python theo cấu trúc AST (function / class / module_header)."""
import logging
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tree_sitter import Node

from ingestion.chunking.parsers import get_parser_for_extension
from ingestion.chunking.ast_utils import (
    get_node_name, extract_docstring, extract_signature, collect_syntax_errors,
    get_range_text, get_line_span, expand_start_byte_to_include_comments,
    split_ast_body_ranges, build_line_map, clean_segment_text,
    extract_decorators, get_relevant_imports, is_async_def, extract_calls,
    collect_module_imports,
)
from ingestion.chunking.splitter import (
    MAX_SEQ_LENGTH, count_tokens_approx, generate_chunk_id, split_large_text_ranges,
)

logger = logging.getLogger('ingestion')

_DEF_TYPES = ('function_definition', 'class_definition')
_MEMBER_TYPES = _DEF_TYPES + ('decorated_definition',)
MAX_ERRORS_PER_CHUNK = 20  # số lỗi tối đa lưu trong metadata mỗi chunk (has_syntax_error vẫn tính trên toàn bộ lỗi)
MAX_HEADER_IMPORTS = 10  # số câu lệnh import tối đa in trong header (metadata 'imports' vẫn giữ đầy đủ)
MAX_HEADER_IMPORT_CHARS = 300
HEADER_TOKEN_MARGIN = 100  # phần dự phòng cho header/signature so với max_seq_length của embedder
# Chỉ đi sâu vào các node statement có thể chứa def/class (không duyệt xuống biểu thức).
# 'ERROR': với file lỗi cú pháp, tree-sitter có thể bọc def/class trong node ERROR.
_CONTAINER_TYPES = {
    'module', 'block', 'if_statement', 'elif_clause', 'else_clause', 'for_statement',
    'while_statement', 'try_statement', 'except_clause', 'finally_clause', 'with_statement',
    'match_statement', 'case_clause', 'except_group_clause',
    'ERROR',
}

_HEADER_PREFIXES = ('# File:', '# Class:', '# Node:', '# Imports:', '# Decorators:', '# Signature:', '# Docstring:')


@dataclass
class _FileContext:
    """Trạng thái dùng chung khi chunk một file."""
    source_bytes: bytes
    relative_path: str
    repo_name: str
    directory_path: str
    ext: str
    errors: list[dict]
    max_chunk_tokens: int
    min_chunk_tokens: int
    module_imports: list[tuple[set[str], str]] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)
    id_counts: dict[str, int] = field(default_factory=dict)


def _compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@lru_cache(maxsize=None)
def _warn_budget(max_chunk_tokens: int) -> None:
    if max_chunk_tokens + HEADER_TOKEN_MARGIN > MAX_SEQ_LENGTH:
        logger.warning(
            f"max_chunk_tokens={max_chunk_tokens} cộng phần header vượt max_seq_length={MAX_SEQ_LENGTH} "
            f"của embedder, chunk có thể bị cắt khi embed."
        )


def _build_enriched_header(
    relative_path: str,
    parent_class: str,
    qualified_name: str,
    signature: str = '',
    decorators: list[str] | None = None,
    docstring: str = '',
    relevant_imports: list[str] | None = None,
) -> str:
    """Tạo header giàu ngữ cảnh phục vụ embedding: File, Class, Node, Imports, Decorators, Signature, Docstring."""
    lines = [f"# File: {relative_path}"]
    if parent_class:
        lines.append(f"# Class: {parent_class}")
    if qualified_name and not qualified_name.endswith("anonymous"):
        lines.append(f"# Node: {qualified_name}")
    if relevant_imports:
        imps = ' | '.join(relevant_imports[:MAX_HEADER_IMPORTS])
        if len(imps) > MAX_HEADER_IMPORT_CHARS or len(relevant_imports) > MAX_HEADER_IMPORTS:
            imps = imps[:MAX_HEADER_IMPORT_CHARS].rstrip(' |') + ' ...'
        lines.append(f"# Imports: {imps}")
    if decorators:
        lines.append(f"# Decorators: {', '.join(decorators)}")
    if signature:
        sig = ' '.join(signature.split())
        for d in sorted(decorators or [], key=len, reverse=True):
            sig = sig.replace(' '.join(d.split()), '')  # header chỉ giữ dòng def, decorator đã có dòng riêng
        lines.append(f"# Signature: {' '.join(sig.split())}")
    if docstring:
        short_doc = ' '.join(docstring.split())[:200]
        lines.append(f"# Docstring: {short_doc}")
    return '\n'.join(lines) + '\n'


def _build_chunk_metadata(
    ctx: _FileContext,
    chunk_type: str,
    node_name: str,
    qualified_name: str,
    parent_class: str,
    start_line: int,
    end_line: int,
    part_start: int,
    part_end: int,
    part: int,
    total_parts: int,
    docstring: str,
    part_errors: list[dict],
    content: str,
    signature: str = '',
    decorators: list[str] | None = None,
    is_async: bool = False,
    imports: list[str] | None = None,
    calls: list[str] | None = None,
    start_byte: int = 0,
    end_byte: int = 0,
    parent_class_id: str = '',
) -> dict:
    return {
        'file_path': ctx.relative_path,
        'directory_path': ctx.directory_path,
        'chunk_type': chunk_type,
        'node_name': node_name,
        'qualified_name': qualified_name,
        'parent_class': parent_class,
        'parent_class_id': parent_class_id,
        'start_line': start_line,
        'end_line': end_line,
        'part_start_line': part_start,
        'part_end_line': part_end,
        'part_index': part + 1,
        'total_parts': total_parts,
        'start_byte': start_byte,
        'end_byte': end_byte,
        'language': ctx.ext.lstrip(".").lower(),
        'signature': signature,
        'decorators': decorators or [],
        'is_async': is_async,
        'imports': imports or [],
        'calls': calls or [],
        'content_hash': _compute_content_hash(content),
        'docstring': docstring,
        'has_syntax_error': bool(part_errors),
        'syntax_errors': part_errors[:MAX_ERRORS_PER_CHUNK],
    }


def _unique_chunk_id(ctx: _FileContext, identity: str) -> str:
    """ID ổn định theo (repo, đường dẫn, tên đầy đủ, loại, part); trùng identity trong cùng file thì thêm số thứ tự xuất hiện."""
    n = ctx.id_counts.get(identity, 0)
    ctx.id_counts[identity] = n + 1
    final_identity = identity if n == 0 else f'{identity}#{n}'
    return generate_chunk_id(f'{ctx.repo_name}:{ctx.relative_path}', final_identity)


def _append_chunk_record(
    ctx: _FileContext,
    sub_text: str,
    header: str,
    chunk_type: str,
    node_name: str,
    qualified_name: str,
    parent_class: str,
    start_line: int,
    end_line: int,
    part_start: int,
    part_end: int,
    part: int,
    total_parts: int,
    docstring: str,
    chunk_errors: list[dict],
    signature: str = '',
    decorators: list[str] | None = None,
    is_async: bool = False,
    imports: list[str] | None = None,
    calls: list[str] | None = None,
    start_byte: int = 0,
    end_byte: int = 0,
    parent_class_id: str = '',
) -> str:
    total_file_lines = ctx.source_bytes.count(b'\n') + 1
    clamped_start_line = max(1, min(start_line, total_file_lines))
    clamped_end_line = max(clamped_start_line, min(end_line, total_file_lines))
    clamped_part_start = max(1, min(part_start, total_file_lines))
    clamped_part_end = max(clamped_part_start, min(part_end, total_file_lines))

    part_errors = [
        e for e in chunk_errors
        if e['line'] <= clamped_part_end and e['end_line'] >= clamped_part_start
    ]

    identity = f'{qualified_name}:{chunk_type}:{part}'
    full_content = header + sub_text
    chunk_id = _unique_chunk_id(ctx, identity)

    ctx.chunks.append({
        'id': chunk_id,
        'content': full_content,
        'metadata': _build_chunk_metadata(
            ctx=ctx,
            chunk_type=chunk_type,
            node_name=node_name,
            qualified_name=qualified_name,
            parent_class=parent_class,
            start_line=clamped_start_line,
            end_line=clamped_end_line,
            part_start=clamped_part_start,
            part_end=clamped_part_end,
            part=part,
            total_parts=total_parts,
            docstring=docstring,
            part_errors=part_errors,
            content=full_content,
            signature=signature,
            decorators=decorators,
            is_async=is_async,
            imports=imports,
            calls=calls,
            start_byte=start_byte,
            end_byte=end_byte,
            parent_class_id=parent_class_id,
        ),
    })
    return chunk_id


def _add_ast_node_chunks(
    ctx: _FileContext,
    outer_node: Node,
    target_node: Node,
    start_byte: int,
    end_byte: int,
    chunk_type: str,
    node_name: str,
    parent_class: str,
    docstring: str = '',
    decorators: list[str] | None = None,
    relevant_imports: list[str] | None = None,
    parent_class_id: str = '',
) -> None:
    qualified_name = f"{parent_class}.{node_name}" if parent_class else node_name
    start_line, end_line = get_line_span(ctx.source_bytes, start_byte, end_byte)
    chunk_errors = [e for e in ctx.errors if e['line'] <= end_line and e['end_line'] >= start_line]

    signature = extract_signature(outer_node, target_node, ctx.source_bytes)
    header = _build_enriched_header(
        ctx.relative_path,
        parent_class,
        qualified_name,
        signature=signature,
        decorators=decorators,
        docstring=docstring,
        relevant_imports=relevant_imports,
    )
    sig_prefix_sample = f"{signature}\n    # (part 99/99)\n"

    budget = max(
        ctx.max_chunk_tokens - count_tokens_approx(header) - count_tokens_approx(sig_prefix_sample),
        ctx.max_chunk_tokens // 2,
    )

    ast_parts = split_ast_body_ranges(target_node, start_byte, ctx.source_bytes, budget, count_tokens_approx)

    expanded_parts: list[tuple[str, int, int, bool]] = []
    for idx, (part_text, p_start, p_end) in enumerate(ast_parts):
        if count_tokens_approx(part_text) <= budget:
            expanded_parts.append((part_text, p_start, p_end, idx > 0))
        else:
            for sub_idx, (sub_t, f_off, l_off) in enumerate(split_large_text_ranges(part_text, budget)):
                expanded_parts.append((sub_t, p_start + f_off, p_start + l_off, idx > 0 or sub_idx > 0))

    is_async = is_async_def(target_node, ctx.source_bytes)
    calls = extract_calls(target_node, ctx.source_bytes)

    total_parts = len(expanded_parts)
    sig_last = signature.split('\n')[-1].strip()
    for part_idx, (sub_text, p_start, p_end, needs_sig) in enumerate(expanded_parts):
        if needs_sig:
            sub_text = f"{signature}\n    # (part {part_idx + 1}/{total_parts})\n{sub_text}"
        elif total_parts > 1:
            lines = sub_text.split('\n')
            idx = next((i for i, l in enumerate(lines) if l.strip().startswith(sig_last)), None)
            if idx is not None:
                lines.insert(idx + 1, f"    # (part 1/{total_parts})")
                sub_text = '\n'.join(lines)

        _append_chunk_record(
            ctx,
            sub_text,
            header,
            chunk_type,
            node_name,
            qualified_name,
            parent_class,
            start_line,
            end_line,
            p_start,
            p_end,
            part_idx,
            total_parts,
            docstring,
            chunk_errors,
            signature=signature,
            decorators=decorators,
            is_async=is_async,
            imports=relevant_imports,
            calls=calls,
            start_byte=start_byte,
            end_byte=end_byte,
            parent_class_id=parent_class_id,
        )


def _add_raw_chunk(
    ctx: _FileContext,
    content: str,
    chunk_type: str,
    node_name: str,
    parent_class: str,
    start_line: int,
    end_line: int,
    docstring: str = '',
    spans: list[tuple[int, int]] | None = None,
    line_map: list[int | None] | None = None,
    decorators: list[str] | None = None,
    relevant_imports: list[str] | None = None,
    signature: str = '',
    calls: list[str] | None = None,
    start_byte: int = 0,
    end_byte: int = 0,
    parent_class_id: str = '',
) -> list[str]:
    """Dùng cho class_container, module_header và entire_file. Trả về danh sách id của các part."""
    qualified_name = f"{parent_class}.{node_name}" if parent_class else node_name
    spans = spans or [(start_line, end_line)]
    chunk_errors = [e for e in ctx.errors if any(e['line'] <= b and e['end_line'] >= a for a, b in spans)]

    header = _build_enriched_header(
        ctx.relative_path,
        parent_class,
        qualified_name,
        signature=signature,
        decorators=decorators,
        docstring=docstring,
        relevant_imports=relevant_imports,
    )
    budget = max(ctx.max_chunk_tokens - count_tokens_approx(header), ctx.max_chunk_tokens // 2)

    n_lines = len(content.split('\n'))
    if line_map is None:
        line_of: list[int | None] = [start_line + i for i in range(n_lines)]
    elif len(line_map) == n_lines:
        line_of = line_map
    else:
        line_of = [None] * n_lines

    ranges = split_large_text_ranges(content, budget)
    total_parts = len(ranges)
    ids: list[str] = []
    for part, (sub_text, first, last) in enumerate(ranges):
        covered = [n for n in line_of[first:last + 1] if n is not None]
        if covered:
            part_start, part_end = covered[0], covered[-1]
            # Chỉ tính lỗi nằm trên đúng các dòng thuộc part (không tính các đoạn nằm giữa, vd các method)
            part_errors = [e for e in chunk_errors if any(e['line'] <= n <= e['end_line'] for n in covered)]
        else:
            part_start, part_end = start_line, end_line
            part_errors = chunk_errors

        if total_parts > 1 and node_name and node_name != "anonymous":
            sub_text = f"# {qualified_name} (part {part + 1}/{total_parts})\n" + sub_text

        ids.append(_append_chunk_record(
            ctx=ctx,
            sub_text=sub_text,
            header=header,
            chunk_type=chunk_type,
            node_name=node_name,
            qualified_name=qualified_name,
            parent_class=parent_class,
            start_line=start_line,
            end_line=end_line,
            part_start=part_start,
            part_end=part_end,
            part=part,
            total_parts=total_parts,
            docstring=docstring,
            chunk_errors=part_errors,
            signature=signature,
            decorators=decorators,
            is_async=False,
            imports=relevant_imports,
            calls=calls,
            start_byte=start_byte,
            end_byte=end_byte,
            parent_class_id=parent_class_id,
        ))
    return ids


def _process_node(
    ctx: _FileContext,
    node: Node,
    parent_class: str = "",
    parent_class_id: str = "",
) -> None:
    src = ctx.source_bytes
    target = node
    if node.type == 'decorated_definition':
        target = next((c for c in node.children if c.type in _DEF_TYPES), node)

    if target.type in _DEF_TYPES:
        name = get_node_name(target, src)
        qualified_name = f"{parent_class}.{name}" if parent_class else name
        start_byte = expand_start_byte_to_include_comments(src, node.start_byte)
        end_byte = target.end_byte
        start_line, end_line = get_line_span(src, start_byte, end_byte)
        doc = extract_docstring(target, src)
        decorators = extract_decorators(node, src)
        rel_imports = get_relevant_imports(node, src, ctx.module_imports)
        ctx.covered_ranges.append((start_byte, end_byte))

        if target.type == 'function_definition':
            _add_ast_node_chunks(
                ctx, node, target, start_byte, end_byte,
                'method' if parent_class else 'function',
                name, parent_class, doc,
                decorators=decorators,
                relevant_imports=rel_imports,
                parent_class_id=parent_class_id,
            )
            return

        body = target.child_by_field_name('body')
        members = [c for c in body.children if c.type in _MEMBER_TYPES] if body else []

        if not members:
            _add_ast_node_chunks(
                ctx, node, target, start_byte, end_byte, 'class', name, parent_class, doc,
                decorators=decorators, relevant_imports=rel_imports,
                parent_class_id=parent_class_id,
            )
            return

        ranges, cursor = [], start_byte
        for m in members:
            m_start = expand_start_byte_to_include_comments(src, m.start_byte)
            ranges.append((cursor, m_start))
            cursor = m.end_byte
        ranges.append((cursor, end_byte))
        ranges = [r for r in ranges if src[r[0]:r[1]].strip()]
        pieces = [get_range_text(src, a, b).strip('\n').rstrip() for a, b in ranges]
        range_spans = [get_line_span(src, a, b) for a, b in ranges]

        container_ids = _add_raw_chunk(
            ctx, '\n\n'.join(pieces), 'class_container', name, parent_class, start_line, end_line, doc,
            spans=range_spans, line_map=build_line_map(pieces, range_spans),
            decorators=decorators, relevant_imports=rel_imports,
            start_byte=start_byte, end_byte=end_byte,
            parent_class_id=parent_class_id,
        )
        class_container_id = container_ids[0]

        for m in members:
            _process_node(ctx, m, parent_class=qualified_name, parent_class_id=class_container_id)

    elif node.type in _CONTAINER_TYPES:
        for child in node.children:
            _process_node(ctx, child, parent_class, parent_class_id)


def _find_uncovered_segments(ctx: _FileContext) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    last_byte = 0
    total = len(ctx.source_bytes)
    for start_b, end_b in sorted(ctx.covered_ranges) + [(total, total)]:
        if start_b > last_byte and ctx.source_bytes[last_byte:start_b].strip():
            segments.append((last_byte, start_b))
        last_byte = max(last_byte, end_b)
    return segments


def _add_module_header_chunks(ctx: _FileContext, segments: list[tuple[int, int]], stem: str) -> None:
    for idx, (a, b) in enumerate(segments):
        piece = clean_segment_text(get_range_text(ctx.source_bytes, a, b))
        if not piece.strip():
            continue
        s_line, e_line = get_line_span(ctx.source_bytes, a, b)
        _add_raw_chunk(ctx, piece, 'module_header', f'{stem}_part{idx + 1}', '', s_line, e_line, spans=[(s_line, e_line)])


def _split_header(content: str) -> tuple[str, str]:
    """Tách (header, body): header là các dòng đầu bắt đầu bằng một trong _HEADER_PREFIXES."""
    lines = content.split('\n')
    i = 0
    while i < len(lines) and lines[i].startswith(_HEADER_PREFIXES):
        i += 1
    return '\n'.join(lines[:i]), '\n'.join(lines[i:])


def _merge_small_chunks(ctx: _FileContext) -> list[dict]:
    """Gộp các function/method liền kề (cùng parent_class) hoặc module_header liền kề nếu một bên quá nhỏ (đếm trên phần thân).
    'Liền kề' nghĩa là giữa hai chunk trong file chỉ có dòng trống."""
    if ctx.min_chunk_tokens <= 0 or len(ctx.chunks) <= 1:
        return ctx.chunks

    src_lines = ctx.source_bytes.split(b'\n')
    funcs = ('function', 'method')
    merged: list[dict] = []
    for chunk in ctx.chunks:
        if not merged:
            merged.append(chunk)
            continue

        prev = merged[-1]
        pm, cm = prev['metadata'], chunk['metadata']
        same_scope = pm['total_parts'] == 1 and cm['total_parts'] == 1 and (
            (pm['chunk_type'] == cm['chunk_type'] == 'module_header')
            or (pm['chunk_type'] in funcs and cm['chunk_type'] in funcs and pm['parent_class'] == cm['parent_class'])
        )
        adjacent = same_scope and cm['start_line'] > pm['end_line'] and not any(
            l.strip() for l in src_lines[pm['end_line']:cm['start_line'] - 1]
        )
        if not adjacent:
            merged.append(chunk)
            continue

        _, prev_body = _split_header(prev['content'])
        _, curr_body = _split_header(chunk['content'])
        if min(count_tokens_approx(prev_body), count_tokens_approx(curr_body)) >= ctx.min_chunk_tokens:
            merged.append(chunk)
            continue

        name = f"{pm['node_name']}+{cm['node_name']}"
        qname = f"{pm['parent_class']}.{name}" if pm['parent_class'] else name
        signature = ' | '.join(s for s in (pm['signature'], cm['signature']) if s)
        decorators = pm['decorators'] + [d for d in cm['decorators'] if d not in pm['decorators']]
        imports = pm['imports'] + [i for i in cm['imports'] if i not in pm['imports']]
        docstring = f"{pm['docstring']}\n{cm['docstring']}".strip()

        new_header = _build_enriched_header(
            ctx.relative_path, pm['parent_class'], qname,
            signature=signature, decorators=decorators, docstring=docstring, relevant_imports=imports,
        )
        combined = f"{new_header}{prev_body.strip(chr(10))}\n\n{curr_body.strip(chr(10))}"
        if count_tokens_approx(combined) > ctx.max_chunk_tokens:
            merged.append(chunk)
            continue

        prev['content'] = combined
        prev['id'] = _unique_chunk_id(ctx, f"{qname}:{pm['chunk_type']}:0")
        pm.update(
            node_name=name,
            qualified_name=qname,
            start_line=min(pm['start_line'], cm['start_line']),
            end_line=max(pm['end_line'], cm['end_line']),
            part_start_line=min(pm['part_start_line'], cm['part_start_line']),
            part_end_line=max(pm['part_end_line'], cm['part_end_line']),
            start_byte=min(pm['start_byte'], cm['start_byte']),
            end_byte=max(pm['end_byte'], cm['end_byte']),
            signature=signature,
            decorators=decorators,
            imports=imports,
            calls=pm['calls'] + [c for c in cm['calls'] if c not in pm['calls']],
            is_async=pm['is_async'] or cm['is_async'],
            docstring=docstring,
            has_syntax_error=pm['has_syntax_error'] or cm['has_syntax_error'],
            syntax_errors=(pm['syntax_errors'] + [e for e in cm['syntax_errors'] if e not in pm['syntax_errors']])[:MAX_ERRORS_PER_CHUNK],
            content_hash=_compute_content_hash(combined),
        )
    return merged


def chunk_file(
    file_path: str | Path,
    repo_root: str | Path,
    max_chunk_tokens: int = 400,
    min_chunk_tokens: int = 60,
) -> list[dict]:
    path_obj = Path(file_path).resolve()
    root_obj = Path(repo_root).resolve()
    try:
        relative_path = path_obj.relative_to(root_obj).as_posix()
    except ValueError:
        logger.warning(f"Bỏ qua file nằm ngoài repo_root ({root_obj}): {file_path}")
        return []
    parent = Path(relative_path).parent.as_posix()
    directory_path = '' if parent == '.' else parent

    try:
        # Chuẩn hóa xuống dòng để nội dung chunk không lẫn \r\n và \n (số dòng không đổi; byte offset tính trên bản đã chuẩn hóa)
        source_bytes = path_obj.read_bytes().replace(b'\r\n', b'\n')
    except OSError:
        logger.exception(f"Failed to read file: {file_path}")
        return []

    if source_bytes.startswith(b'\xef\xbb\xbf'):  # BOM UTF-8
        source_bytes = source_bytes[3:]

    if not source_bytes.strip():
        return []

    ext = path_obj.suffix
    parser = get_parser_for_extension(ext)
    if not parser:
        logger.warning(f"Bỏ qua file do chưa hỗ trợ parser cho định dạng {ext}: {relative_path}")
        return []

    _warn_budget(max_chunk_tokens)

    root_node = parser.parse(source_bytes).root_node

    errors = collect_syntax_errors(root_node, source_bytes) if root_node.has_error else []
    if errors:
        logger.warning(f"Syntax error ({len(errors)} vị trí), vẫn chunk và đánh dấu: {relative_path}")

    ctx = _FileContext(
        source_bytes=source_bytes,
        relative_path=relative_path,
        repo_name=root_obj.name,
        directory_path=directory_path,
        ext=ext,
        errors=errors,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
    ctx.module_imports = collect_module_imports(root_node, source_bytes)
    _process_node(ctx, root_node)

    if not ctx.covered_ranges:
        _add_raw_chunk(
            ctx, get_range_text(source_bytes, 0, len(source_bytes)), 'entire_file', path_obj.stem, '',
            1, len(source_bytes.split(b'\n')),
        )
        return ctx.chunks

    _add_module_header_chunks(ctx, _find_uncovered_segments(ctx), path_obj.stem)
    return _merge_small_chunks(ctx)