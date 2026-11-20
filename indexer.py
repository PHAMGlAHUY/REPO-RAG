from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config.load_config import load_settings, setup_logging
from embedding.embedder import check_index_compat, warmup
from ingestion.pipeline import iter_repository_chunks, load_index_state, save_index_state
from vectorstore.qdrant import close_qdrant_client, ensure_collection, get_qdrant_client
from vectorstore.search import ensure_payload_indexes
from vectorstore.upsert import index_batch

logger = logging.getLogger("ingestion")


def run_index(
    repo_root: str | Path,
    state_file: str | Path = "data/index_state.json",
    workers: int | None = None,
    batch_size: int = 100,
    max_chunk_tokens: int | None = None,
    min_chunk_tokens: int | None = None,
    do_warmup: bool = True,
) -> None:
    setup_logging()
    _ = load_settings()  # đảm bảo cấu hình hợp lệ ngay từ đầu

    repo_path = Path(repo_root).resolve()
    state_path = Path(state_file)

    # 1. Kiểm tra tương thích model embedding & khởi tạo Qdrant
    check_index_compat()
    if do_warmup:
        warmup()

    client = get_qdrant_client()
    try:
        ensure_collection(client)
        ensure_payload_indexes(client)

        # Lưu trạng thái hash từ lần index trước; chỉ commit vào đĩa sau khi batch lưu thành công
        committed_state: dict[str, str] = load_index_state(state_path)
        deleted_files: list[str] = []
        changed_files: list[str] = []
        total_chunks = 0
        total_batches = 0

        # 2. Quét repo song song & yield từng batch chunk
        for batch_chunks in iter_repository_chunks(
            repo_root=repo_path,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            batch_size=batch_size,
            workers=workers,
            state_file=None,  # tự quản lý commit state sau khi index_batch thành công
            deleted_files_out=deleted_files,
            changed_files_out=changed_files,
        ):
            # Chỉ xóa chunk của file đổi/xóa ở batch đầu tiên để tránh xóa lặp
            batch_deleted = deleted_files if total_batches == 0 else ()
            batch_changed = changed_files if total_batches == 0 else ()

            # 3. Xóa chunk cũ (nếu có), tạo dense + sparse vectors và upsert vào Qdrant
            ok = index_batch(
                client=client,
                chunks=batch_chunks,
                changed_files=batch_changed,
                deleted_files=batch_deleted,
            )
            if not ok:
                raise RuntimeError(f"index_batch trả về False ở batch {total_batches + 1}")

            # 4. Cập nhật state cho các file có chunk vừa lưu thành công
            for ch in batch_chunks:
                md = ch.get("metadata") or {}
                rel_path = md.get("file_path")
                file_hash = md.get("file_hash")
                if rel_path and file_hash:
                    committed_state[rel_path] = file_hash

            if total_batches == 0:
                for p in batch_deleted:
                    committed_state.pop(p, None)

            save_index_state(state_path, committed_state)
            total_batches += 1
            total_chunks += len(batch_chunks)
            logger.info("Đã index batch %d (%d chunks, tổng %d chunks)", total_batches, len(batch_chunks), total_chunks)

        # Trường hợp chỉ có file bị xóa/đổi tên mà không có chunk mới nào được yield
        if total_batches == 0 and (deleted_files or changed_files):
            index_batch(
                client=client,
                chunks=[],
                changed_files=changed_files,
                deleted_files=deleted_files,
            )
            for p in deleted_files:
                committed_state.pop(p, None)
            save_index_state(state_path, committed_state)

        logger.info(
            "Hoàn tất index repo '%s': %d chunks, %d file đổi, %d file xóa.",
            repo_path,
            total_chunks,
            len(changed_files),
            len(deleted_files),
        )
    finally:
        # 5. Đóng kết nối Qdrant
        close_qdrant_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index repository vào Qdrant")
    parser.add_argument("repo_root", nargs="?", default=".", help="Đường dẫn thư mục repository")
    parser.add_argument("--state-file", default="data/index_state.json", help="File lưu hash tăng dần")
    parser.add_argument("--workers", type=int, default=None, help="Số worker process chunking")
    parser.add_argument("--batch-size", type=int, default=100, help="Số chunk mỗi batch yield từ pipeline")
    parser.add_argument("--no-warmup", action="store_true", help="Bỏ qua bước warmup model embedding")
    args = parser.parse_args()

    run_index(
        repo_root=args.repo_root,
        state_file=args.state_file,
        workers=args.workers,
        batch_size=args.batch_size,
        do_warmup=not args.no_warmup,
    )