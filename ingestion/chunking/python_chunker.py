import logging
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from core.load_settings import load_settings

logger = logging.getLogger(__name__)

_PARSER = Parser(Language(tspython.language()))
_DEF_TYPES = {"function_definition", "class_definition"}


def _unwrap(node):
    """Hàm/class có decorator nằm trong decorated_definition, lấy phần định nghĩa bên trong."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition")
    return node


def _name(node) -> str:
    n = _unwrap(node).child_by_field_name("name")
    return n.text.decode("utf-8", errors="replace") if n else "?"


def _make(data: bytes, start: int, end: int, rel: str, layer: str, name: str,
          kind: str, start_row: int, end_row: int) -> dict:
    code = data[start:end].decode("utf-8", errors="replace")
    return {
        "text": f"# {rel}\n{code}",   # nhúng đường dẫn vào text giúp retrieval tốt hơn
        "metadata": {
            "file": rel, "name": name, "type": kind, "layer": layer,
            "start_line": start_row + 1, "end_line": end_row + 1,
        },
    }


def chunk_python(path: Path, root: Path) -> list[dict]:
    rel = path.relative_to(root).as_posix()   # as_posix để dùng "/" trên Windows
    parts = rel.split("/")
    layer = parts[1] if parts[0] == "app" and len(parts) > 2 else parts[0] if len(parts) > 1 else "root"
    max_chars = load_settings()["ingestion"].get("max_chunk_tokens", 1000) * 4  # ~4 ký tự/token

    data = path.read_bytes()
    if not data.strip():
        logger.debug("File rỗng, bỏ qua: %s", rel)
        return []

    tree = _PARSER.parse(data)
    if tree.root_node.has_error:
        logger.warning("File có lỗi cú pháp (vẫn chunk phần parse được): %s", rel)

    chunks, header_parts = [], []

    for node in tree.root_node.children:
        d = _unwrap(node)
        if d is None or d.type not in _DEF_TYPES:
            # import, hằng số, câu lệnh cấp module -> gom vào chunk header
            header_parts.append(data[node.start_byte:node.end_byte].decode("utf-8", errors="replace"))
            continue

        name = _name(node)
        kind = d.type   # function_definition | class_definition
        size = node.end_byte - node.start_byte

        # Class quá dài -> tách thành header của class + từng method
        if kind == "class_definition" and size > max_chars:
            body = d.child_by_field_name("body")
            methods = [c for c in body.children
                       if (u := _unwrap(c)) is not None and u.type == "function_definition"]
            if methods:
                first = methods[0]
                chunks.append(_make(data, node.start_byte, first.start_byte, rel, layer,
                                    name, "class_header", node.start_point[0], first.start_point[0]))
                for m in methods:
                    chunks.append(_make(data, m.start_byte, m.end_byte, rel, layer,
                                        f"{name}.{_name(m)}", "method",
                                        m.start_point[0], m.end_point[0]))
                logger.debug("%s::%s tách thành %d method", rel, name, len(methods))
                continue

        if size > max_chars:
            logger.warning("Chunk dài (%d ký tự): %s::%s", size, rel, name)
        chunks.append(_make(data, node.start_byte, node.end_byte, rel, layer,
                            name, kind, node.start_point[0], node.end_point[0]))

    header = "\n".join(header_parts).strip()
    if header:
        chunks.insert(0, {"text": f"# {rel}\n{header}",
                          "metadata": {"file": rel, "name": "<module>", "type": "header", "layer": layer}})

    logger.debug("%s -> %d chunk", rel, len(chunks))
    return chunks