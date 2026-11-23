import logging
from typing import Callable

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

logger = logging.getLogger('ingestion')

PY_LANGUAGE = Language(tspython.language())

# Registry các factory tạo Parser theo extension (chữ thường), dễ mở rộng thêm JS/TS...
PARSER_FACTORIES: dict[str, Callable[[], Parser]] = {
    '.py': lambda: Parser(PY_LANGUAGE),
}

# Cache parser theo tiến trình/thread hiện tại (mỗi process có dict riêng khi fork/spawn)
_LOCAL_PARSERS: dict[str, Parser] = {}


def create_parser_for_extension(ext: str) -> Parser | None:
    """Tạo instance Parser mới (phù hợp cho từng worker process chạy song song)."""
    factory = PARSER_FACTORIES.get(ext.lower())
    return factory() if factory else None


def get_parser_for_extension(ext: str) -> Parser | None:
    """Lấy Parser thuộc về process/context hiện tại (không dùng chung cross-process/cross-thread)."""
    norm_ext = ext.lower()
    if norm_ext not in _LOCAL_PARSERS:
        parser = create_parser_for_extension(norm_ext)
        if parser is None:
            return None
        _LOCAL_PARSERS[norm_ext] = parser
    return _LOCAL_PARSERS[norm_ext]