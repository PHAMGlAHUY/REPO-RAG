import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def chunk_text_file(path: Path, root: Path) -> list[dict]:
    rel = path.relative_to(root).as_posix() #ép kiểu đường dẫn từ \ qua /
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError: # bắt lỗi khi tệp không đúng định dạng utf-8
        logger.error("Không đọc được (encoding): %s", rel)
        return []
    logger.debug("%s -> 1 chunk (%d ký tự)", rel, len(text))
    return [{"text": f"# {rel}\n{text}",
             "metadata": {"file": rel, "name": path.name, "type": "config", "layer": "config"}}]