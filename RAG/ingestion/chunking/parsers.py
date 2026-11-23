import threading
from typing import Callable

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())

# Registry các factory tạo Parser theo extension (chữ thường), dễ mở rộng thêm JS/TS...
PARSER_FACTORIES: dict[str, Callable[[], Parser]] = {
    '.py': lambda: Parser(PY_LANGUAGE),
}

_local = threading.local()


def get_parser_for_extension(ext: str) -> Parser | None:
    """Parser riêng cho từng thread (Parser của tree-sitter không an toàn khi dùng chung giữa các thread)."""
    ext = ext.lower()
    cache: dict[str, Parser | None] | None = getattr(_local, 'parsers', None)
    if cache is None:
        cache = _local.parsers = {}
    if ext not in cache:
        factory = PARSER_FACTORIES.get(ext)
        cache[ext] = factory() if factory else None
    return cache[ext]