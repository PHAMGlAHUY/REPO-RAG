import json
import logging
import time
from collections import Counter
from pathlib import Path

from core.load_settings import load_settings
from core.setup_logging import setup_logging
from ingestion.load_data import iter_files
from ingestion.chunking import chunk_file

from embedding.embedder import embed_passages
from vectorstore.qdrant import ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)


def run():
    setup_logging()
    start = time.perf_counter()
    settings = load_settings()
    root = Path(settings["data"]["raw_dir"])
    logger.info("=== Ingestion pipeline bắt đầu ===")

    chunks, failed = [], []
    for p in iter_files(root):
        try:
            chunks += chunk_file(p, root)
        except Exception:
            logger.exception("Lỗi khi chunk file: %s", p)
            failed.append(str(p))

    by_type = Counter(c["metadata"]["type"] for c in chunks)
    logger.info("Tổng %d chunk | theo loại: %s", len(chunks), dict(by_type))
    if failed:
        logger.warning("%d file lỗi: %s", len(failed), failed)

    out = Path(settings["data"]["processed_dir"]) / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Đã lưu chunk: %s", out)

    logger.info("=== Hoàn tất trong %.2fs ===", time.perf_counter() - start)

    ensure_collection()
    vectors = embed_passages([c['text'] for c in chunks])
    upsert_chunks(chunks, vectors)

    return chunks


if __name__ == "__main__":
    run()