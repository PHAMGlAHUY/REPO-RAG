# Chạy: python -m RAG.ingestion.<tên_file> [repo_root]  (từ thư mục gốc dự án)
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from RAG.config.load_config import load_settings, setup_logging
from RAG.embedding.embedder import check_index_compat, warmup
from RAG.ingestion.pipeline import iter_repository_chunks, load_index_state
from RAG.vectorstore.qdrant import close_qdrant_client, ensure_collection, get_qdrant_client
from RAG.vectorstore.search import ensure_payload_indexes
from RAG.vectorstore.upsert import index_batch

logger = logging.getLogger("ingestion")


def _save_state(path: Path, state: dict[str, str]) -> None:
    """Ghi nguyên tử: ghi file tạm rồi thay thế."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_index(
    repo_root: str | Path,
    state_file: str | Path = "data/index_state.json",
    workers: int | None = None,
    batch_size: int = 100,
    max_chunk_tokens: int | None = None,
    min_chunk_tokens: int | None = None,
    do_warmup: bool = True,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size phải > 0")
    if workers is not None and workers <= 0:
        raise ValueError("workers phải > 0")

    load_settings()  # đảm bảo cấu hình hợp lệ ngay từ đầu

    repo_path = Path(repo_root).resolve()
    state_path = Path(state_file)

    check_index_compat()
    print("Check index compact : \033[32m Done \033[0m")

    if do_warmup:
        warmup()

    print("Warmup : \033[32m Done \033[0m")


    client = get_qdrant_client()\

    print("Connect Qdrant : \033[32m Done \033[0m")

    try:
        ensure_collection(client)

        ensure_payload_indexes(client)

        old_state = load_index_state(state_path)
        seen: dict[str, str] = {}  # hash của các file đã index xong
        deleted_files: list[str] = []
        changed_files: list[str] = []
        purged: set[str] = set()  # file đã xóa chunk cũ, tránh xóa lặp
        total_chunks = 0
        total_batches = 0

        def take_new(files: list[str]) -> list[str]:
            new = [p for p in files if p not in purged]
            purged.update(new)
            return new

        for batch_chunks in iter_repository_chunks(
            repo_root=repo_path,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            batch_size=batch_size,
            workers=workers,
            previous_state=old_state,  # pipeline cần nhận state đã load để so sánh hash
            deleted_files_out=deleted_files,
            changed_files_out=changed_files,
        ):
            if not index_batch(
                client=client,
                chunks=batch_chunks,
                changed_files=take_new(changed_files),
                deleted_files=take_new(deleted_files),
            ):
                raise RuntimeError(f"index_batch return False at batch {total_batches + 1}")

            for ch in batch_chunks:
                md = ch.get("metadata") or {}
                if md.get("file_path") and md.get("file_hash"):
                    seen[md["file_path"]] = md["file_hash"]

            total_batches += 1
            total_chunks += len(batch_chunks)
            logger.info("Indexed batch %d (%d chunks, tổng %d chunks)", total_batches, len(batch_chunks), total_chunks)

        # Phần xóa/đổi còn lại chưa được xử lý (file không còn chunk nào, file bị xóa)
        rest_changed, rest_deleted = take_new(changed_files), take_new(deleted_files)
        if rest_changed or rest_deleted:
            if not index_batch(client=client, chunks=[], changed_files=rest_changed, deleted_files=rest_deleted):
                raise RuntimeError("index_batch return False (delete-only)")

        # Chỉ lưu state khi TOÀN BỘ batch đã thành công
        new_state = dict(old_state)
        for p in deleted_files + changed_files:
            new_state.pop(p, None)
        new_state.update(seen)
        _save_state(state_path, new_state)

        logger.info(
            "Indexing repo successful '%s': %d chunks, %d file changed, %d file deleted.",
            repo_path,
            total_chunks,
            len(changed_files),
            len(deleted_files),
        )
    finally:
        close_qdrant_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index repository vào Qdrant")
    parser.add_argument("repo_root", nargs="?", default=".", help="Đường dẫn thư mục repository")
    parser.add_argument("--state-file", default="data/index_state.json", help="File lưu hash tăng dần")
    parser.add_argument("--workers", type=int, default=None, help="Số worker process chunking")
    parser.add_argument("--batch-size", type=int, default=100, help="Số chunk mỗi batch")
    parser.add_argument("--max-chunk-tokens", type=int, default=None, help="Số token tối đa mỗi chunk")
    parser.add_argument("--min-chunk-tokens", type=int, default=None, help="Số token tối thiểu mỗi chunk")
    parser.add_argument("--no-warmup", action="store_true", help="Bỏ qua bước warmup model embedding")
    args = parser.parse_args()

    setup_logging()
    run_index(
        repo_root=args.repo_root,
        state_file=args.state_file,
        workers=args.workers,
        batch_size=args.batch_size,
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        do_warmup=not args.no_warmup,
    )