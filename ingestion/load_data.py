import logging
from pathlib import Path
from core.load_settings import load_settings

logger = logging.getLogger(__name__)

def iter_files(root: Path | None = None):
    settings = load_settings()
    cfg = settings["ingestion"]
    root = root or Path(settings["data"]["raw_dir"])

    if not root.exists():
        logger.error("raw_dir không tồn tại: %s", root)
        return

    ignore_dirs = set(cfg["ignore_dirs"])
    ignore_files = set(cfg["ignore_files"])
    text_files = set(cfg["text_files"])

    logger.info("Bắt đầu quét: %s", root)
    kept = skipped = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts  # chỉ xét phần tương đối so với root
        if p.name in ignore_files:
            logger.warning("Bỏ qua file nhạy cảm: %s", p)
            skipped += 1
            continue
        if any(part in ignore_dirs for part in rel_parts):
            skipped += 1
            continue
        if p.suffix == ".py" or p.name in text_files:
            logger.debug("Nhận file: %s", p)
            kept += 1
            yield p
        else:
            logger.debug("Bỏ qua (đuôi không hỗ trợ): %s", p)
            skipped += 1

    logger.info("Quét xong: giữ %d file, bỏ qua %d file", kept, skipped)