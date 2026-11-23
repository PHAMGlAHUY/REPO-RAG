"""Tiện ích mức thấp: làm việc với node tree-sitter và với đoạn byte / dòng của mã nguồn.
Không biết gì về 'chunk' hay token."""
import inspect
import textwrap
from bisect import bisect_left
from functools import lru_cache
from typing import Callable

from tree_sitter import Node


def is_async_def(node: Node, source_bytes: bytes) -> bool:
    if node.type != 'function_definition':
        return False

    for child in node.children:
        if child.type == 'async':
            return True
        if child.type == 'def':
            break

    return source_bytes[node.start_byte:node.start_byte + 32].split()[:1] == [b'async']


# ---------------------------------------------------------------- node tree-sitter
def get_range_text(source_bytes: bytes, start: int, end: int) -> str:
    return source_bytes[start:end].decode('utf-8', errors='replace')


def get_node_text(node: Node, source_bytes: bytes) -> str:
    return get_range_text(source_bytes, node.start_byte, node.end_byte)


def get_node_name(node: Node, source_bytes: bytes) -> str:
    name_node = node.child_by_field_name('name')
    return get_node_text(name_node, source_bytes) if name_node else 'anonymous'


def get_relevant_imports(node: Node, source_bytes: bytes, module_imports: list[tuple[set[str], str]]) -> list[str]:
    """Lọc ra các câu lệnh import thực sự được tham chiếu bên trong `node`."""
    if not module_imports:
        return []

    used_identifiers: set[str] = set()
    stack = [node]
    while stack:
        curr = stack.pop()
        if curr.type == 'identifier':
            used_identifiers.add(get_node_text(curr, source_bytes))
        stack.extend(curr.children)

    relevant: list[str] = []
    for imported_names, stmt_text in module_imports:
        if imported_names & used_identifiers:
            relevant.append(stmt_text)
    return relevant


def extract_signature(outer_node: Node, target_node: Node, source_bytes: bytes) -> str:
    colon = next((c for c in target_node.children if c.type == ':'), None)
    if colon is None:
        return get_node_text(outer_node, source_bytes).split('\n')[0]
    return get_range_text(source_bytes, outer_node.start_byte, colon.end_byte).rstrip()


def extract_docstring(node: Node, source_bytes: bytes) -> str:
    body = node.child_by_field_name('body')
    # bỏ qua comment '#' đứng trước docstring
    first = next((c for c in body.children if c.type != 'comment'), None) if body else None
    if first is not None:
        if first.type == 'expression_statement' and first.child_count > 0 and first.children[0].type == 'string':
            text = get_node_text(first.children[0], source_bytes).lstrip('rRbBuUfF')  # bỏ tiền tố r'''...'''
            for quote in ('"""', "'''", '"', "'"):
                if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
                    text = text[len(quote):-len(quote)]
                    break
            return inspect.cleandoc(text)
    return ''


def extract_decorators(outer_node: Node, source_bytes: bytes) -> list[str]:
    if outer_node.type != 'decorated_definition':
        return []

    return [
        get_node_text(c, source_bytes).strip()
        for c in outer_node.children
        if c.type == 'decorator'
    ]


def extract_calls(node: Node, source_bytes: bytes) -> list[str]:
    """Trích xuất tên các hàm / phương thức được gọi bên trong AST node (duyệt bằng stack, không đệ quy)."""
    calls: list[str] = []
    seen: set[str] = set()

    stack = [node]
    while stack:
        curr = stack.pop()
        if curr.type == 'call':
            fn_node = curr.child_by_field_name('function')
            if fn_node is not None:
                name = source_bytes[fn_node.start_byte:fn_node.end_byte].decode('utf-8', errors='replace').strip()
                if name and name not in seen:
                    seen.add(name)
                    calls.append(name)
        stack.extend(reversed(curr.children))
    return calls


def collect_syntax_errors(root: Node, source_bytes: bytes, limit: int | None = None) -> list[dict]:
    """Liệt kê vị trí lỗi cú pháp (node ERROR / MISSING) theo thứ tự trong file. Duyệt vòng lặp, không đệ quy."""
    errors: list[dict] = []
    stack = [root]
    while stack and (limit is None or len(errors) < limit):
        node = stack.pop()
        line, column = node.start_point[0] + 1, node.start_point[1] + 1
        if node.is_missing:  # parser tự chèn token còn thiếu, vd ")" hoặc ":"
            errors.append({'kind': 'missing', 'expected': node.type, 'line': line, 'end_line': line, 'column': column})
        elif node.is_error:
            first_line = get_node_text(node, source_bytes).strip().splitlines()[:1]
            errors.append({
                'kind': 'error',
                'line': line,
                'end_line': min(node.end_point[0] + 1, line + 5),
                'column': column,
                'snippet': first_line[0][:80] if first_line else '',
            })
        if node.has_error and not node.is_error:  # không đi vào trong node ERROR để khỏi báo lỗi lồng nhau
            stack.extend(reversed(node.children))
    return errors


def _bound_names(node: Node, source_bytes: bytes) -> set[str]:
    """Tên được import vào namespace: `import a.b` -> a; `import a as x` -> x; `from m import y as z` -> z."""
    names: set[str] = set()
    for child in node.children_by_field_name('name'):
        if child.type == 'aliased_import':
            alias = child.child_by_field_name('alias')
            if alias is not None:
                names.add(get_node_text(alias, source_bytes))
        else:  # dotted_name
            names.add(get_node_text(child, source_bytes).split('.')[0])
    return names


def collect_module_imports(root_node: Node, source_bytes: bytes) -> list[tuple[set[str], str]]:
    """Import cấp module (kể cả trong try/except, if TYPE_CHECKING): [(tên được import, câu lệnh một dòng)]."""
    imports: list[tuple[set[str], str]] = []
    seen: set[str] = set()

    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in ('import_statement', 'import_from_statement'):
            stmt = ' '.join(get_node_text(node, source_bytes).split())  # một dòng, để không phá cấu trúc header
            names = _bound_names(node, source_bytes)
            if stmt and names and stmt not in seen:
                seen.add(stmt)
                imports.append((names, stmt))
            continue
        if node.type in ('function_definition', 'class_definition'):
            continue  # bỏ import cục bộ trong hàm/class
        stack.extend(reversed(node.children))
    return imports


# ---------------------------------------------------------------- đoạn byte / dòng
@lru_cache(maxsize=4)
def _newline_offsets(source_bytes: bytes) -> list[int]:
    offsets: list[int] = []
    i = source_bytes.find(b'\n')
    while i != -1:
        offsets.append(i)
        i = source_bytes.find(b'\n', i + 1)
    return offsets


def get_line_span(source_bytes: bytes, start: int, end: int) -> tuple[int, int]:
    """Số dòng (1-based) đầu/cuối của đoạn byte, bỏ khoảng trắng hai đầu."""
    if not source_bytes:
        return 1, 1

    offsets = _newline_offsets(source_bytes)
    total_lines = len(offsets) + 1

    start = max(0, min(start, len(source_bytes)))
    end = max(start, min(end, len(source_bytes)))

    raw = source_bytes[start:end]
    left = raw.lstrip()
    if not left:
        line = min(bisect_left(offsets, start) + 1, total_lines)
        return line, line

    first_pos = start + (len(raw) - len(left))
    last_pos = start + len(raw.rstrip()) - 1

    start_line = min(bisect_left(offsets, first_pos) + 1, total_lines)
    end_line = min(bisect_left(offsets, last_pos) + 1, total_lines)
    return start_line, max(start_line, end_line)


def expand_start_byte_to_include_comments(source_bytes: bytes, start_byte: int) -> int:
    """Lùi start_byte để gồm các comment '#' đứng ngay trước node (bỏ các dòng trống ở đầu)."""
    new_start = start_byte
    line_start = source_bytes.rfind(b'\n', 0, new_start) + 1
    if not source_bytes[line_start:new_start].strip():  # phần đệm trước node chỉ là khoảng trắng
        new_start = line_start
        while new_start > 0:
            prev_start = source_bytes.rfind(b'\n', 0, new_start - 1) + 1
            stripped = source_bytes[prev_start:new_start].strip()
            if stripped.startswith(b'#') or not stripped:
                new_start = prev_start
            else:
                break
    # không để chunk mở đầu bằng các dòng trống
    while True:
        nl = source_bytes.find(b'\n', new_start, start_byte)
        if nl == -1 or source_bytes[new_start:nl].strip():
            break
        new_start = nl + 1
    return max(0, new_start)


def _align_to_line_start(source_bytes: bytes, pos: int) -> int:
    """Lùi pos về đầu dòng nếu phần đứng trước nó trên cùng dòng chỉ là khoảng trắng (giữ nguyên thụt lề)."""
    line_start = source_bytes.rfind(b'\n', 0, pos) + 1
    return line_start if not source_bytes[line_start:pos].strip() else pos


def split_ast_body_ranges(
        node: Node,
        start_byte: int,
        source_bytes: bytes,
        budget: int,
        token_counter: Callable[[str], int],
) -> list[tuple[str, int, int]]:
    body = node.child_by_field_name('body')
    if body is None or not body.children:
        s_line, e_line = get_line_span(source_bytes, start_byte, node.end_byte)
        return [(get_range_text(source_bytes, start_byte, node.end_byte), s_line, e_line)]

    stmt_groups: list[tuple[int, int]] = []
    group_start = body.children[0].start_byte
    for idx, child in enumerate(body.children):
        if child.type == 'comment' and idx + 1 < len(body.children):
            continue
        stmt_groups.append((group_start, child.end_byte))
        if idx + 1 < len(body.children):
            group_start = body.children[idx + 1].start_byte

    def make_part(a: int, b: int) -> tuple[str, int, int]:
        _s_line, _e_line = get_line_span(source_bytes, a, b)
        return get_range_text(source_bytes, _align_to_line_start(source_bytes, a), b), _s_line, _e_line

    parts: list[tuple[str, int, int]] = []
    cur_start, cur_end = start_byte, stmt_groups[0][1]
    cur_tokens = token_counter(get_range_text(source_bytes, cur_start, cur_end))

    for g_start, g_end in stmt_groups[1:]:
        g_tokens = token_counter(get_range_text(source_bytes, g_start, g_end))
        if cur_tokens + g_tokens <= budget:
            cur_end, cur_tokens = g_end, cur_tokens + g_tokens
        else:
            parts.append(make_part(cur_start, cur_end))
            cur_start, cur_end, cur_tokens = g_start, g_end, g_tokens

    parts.append(make_part(cur_start, cur_end))
    return parts


def build_line_map(pieces: list[str], piece_spans: list[tuple[int, int]]) -> list[int | None]:
    """Chunk ghép từ nhiều đoạn rời bằng '\\n\\n': trả về số dòng thật trong file của từng dòng của chunk
    (None cho dòng trống do phần ghép tạo ra). Neo theo dòng cuối vì mỗi đoạn đã được rstrip."""
    mapping: list[int | None] = []
    for i, (piece, (_, last)) in enumerate(zip(pieces, piece_spans)):
        if i:
            mapping.append(None)
        mapping.extend(range(last - len(piece.split('\n')) + 1, last + 1))
    return mapping


def clean_segment_text(raw_text: str) -> str:
    return textwrap.dedent(raw_text).strip('\n')