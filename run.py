from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAG = ROOT / "RAG"
LAST_REPO = RAG / "data" / "last_repo.txt"
sys.path.insert(0, str(RAG))

timings: list[tuple[str, float]] = []
_tlock = threading.Lock()


def log(msg: str) -> None:
    print(f"[run] {msg}", flush=True)


@contextmanager
def step(name: str):
    log(f"▶ {name} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _tlock:
            timings.append((name, dt))
        log(f"✔ {name}: {dt:.2f}s")


def add_time(name: str, dt: float) -> None:
    """Cộng dồn thời gian cho các bước lặp theo batch."""
    with _tlock:
        for i, (n, t) in enumerate(timings):
            if n == name:
                timings[i] = (n, t + dt)
                return
        timings.append((name, dt))


def print_summary(total: float) -> None:
    print("\n========== THỜI GIAN CÁC BƯỚC ==========", flush=True)
    for name, dt in timings:
        print(f"{name:<45} {dt:8.2f}s  {dt / total * 100:5.1f}%", flush=True)
    print(f"{'TỔNG':<45} {total:8.2f}s", flush=True)


def resolve_repo(arg: str | None) -> Path:
    if arg:
        repo = Path(arg).resolve()
    elif LAST_REPO.is_file():
        repo = Path(LAST_REPO.read_text(encoding="utf-8").strip())
    else:
        sys.exit("Hãy truyền đường dẫn repo bằng --source")
    if not repo.is_dir():
        sys.exit(f"Không tìm thấy thư mục repo: {repo}")
    LAST_REPO.parent.mkdir(parents=True, exist_ok=True)
    LAST_REPO.write_text(str(repo), encoding="utf-8")
    return repo


def load_models() -> None:
    """Nạp dense + sparse song song, đo riêng từng phần."""
    from embedding.embedder import warmup, get_embedding_dim
    from vectorstore.index import warmup_sparse

    def dense() -> None:
        with step("dense: load + warmup"):
            get_embedding_dim()  # nạp model
            warmup()

    def sparse() -> None:
        with step("sparse: load + warmup"):
            warmup_sparse()

    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in [ex.submit(dense), ex.submit(sparse)]:
            f.result()  # ném lại lỗi nếu có


def run_index(repo: Path, state_path: Path, batch_size: int, workers: int | None) -> None:
    from ingestion.check import check_index
    from ingestion.pipeline import iter_repository_chunks, load_index_state, save_index_state
    from qdrant_client.models import HnswConfigDiff
    from vectorstore.index import create_qdrant_points
    from vectorstore.qdrant import (
        HNSW_EF_CONSTRUCT,
        HNSW_M,
        VECTOR_DB_BATCH_SIZE,
        VECTOR_DB_COLLECTION,
        close_qdrant_client,
        ensure_collection,
        get_qdrant_client,
    )
    from vectorstore.upsert import delete_files, delete_stale, retry

    try:
        with step("kết nối Qdrant"):
            client = get_qdrant_client()
            client.get_collections()

        with step("kiểm tra index (quét + so hash)"):
            res = check_index(repo, state_path)
        if not res["needs_index"]:
            log("Index đã cập nhật, không có thay đổi.")
            return
        log("Cần index: " + "; ".join(res["reasons"]))

        with step("nạp model (dense + sparse song song)"):
            load_models()

        with step("ensure_collection + payload index"):
            ensure_collection(client)

        with step("load_index_state"):
            old_state = {} if res["reset_state"] else load_index_state(state_path)

        new_state: dict[str, str] = {}
        deleted: list[str] = []
        changed: list[str] = []
        keep_ids: list = []  # id chunk mới của các file đã sửa, để xóa chunk cũ sau khi upsert xong
        total_chunks = 0
        n_batches = 0

        def process(batch: list[dict], idx: int) -> None:
            t0 = time.perf_counter()
            points_all = [
                create_qdrant_points(batch[i:i + VECTOR_DB_BATCH_SIZE])
                for i in range(0, len(batch), VECTOR_DB_BATCH_SIZE)
            ]
            t_embed = time.perf_counter() - t0
            add_time("embed (dense + sparse)", t_embed)

            t0 = time.perf_counter()
            for points in points_all:
                retry(lambda p=points: client.upsert(collection_name=VECTOR_DB_COLLECTION, points=p, wait=True))
            t_up = time.perf_counter() - t0
            add_time("qdrant upsert", t_up)
            log(f"  batch {idx}: {len(batch)} chunks | embed {t_embed:.2f}s, upsert {t_up:.2f}s")

        bulk = res["reset_state"]  # collection trống: tắt dựng HNSW trong lúc nạp, bật lại sau
        if bulk:
            client.update_collection(VECTOR_DB_COLLECTION, hnsw_config=HnswConfigDiff(m=0))

        try:
            gen = iter_repository_chunks(
                repo_root=repo,
                batch_size=batch_size,
                workers=workers,
                previous_state=old_state,
                deleted_files_out=deleted,
                changed_files_out=changed,
                state_out=new_state,
            )

            log("▶ chunking + embed + upsert theo batch (chồng thời gian) ...")
            pending = None
            with ThreadPoolExecutor(max_workers=1) as ex:
                while True:
                    t0 = time.perf_counter()
                    try:
                        batch = next(gen)
                    except StopIteration:
                        add_time("chunking (quét hash + tách chunk)", time.perf_counter() - t0)
                        break
                    add_time("chunking (quét hash + tách chunk)", time.perf_counter() - t0)

                    changed_set = set(changed)
                    keep_ids.extend(c["id"] for c in batch if c["metadata"]["file_path"] in changed_set)

                    if pending is not None:
                        pending.result()  # tối đa một batch đang embed/upsert
                    n_batches += 1
                    total_chunks += len(batch)
                    pending = ex.submit(process, batch, n_batches)

                if pending is not None:
                    pending.result()

            with step("xóa chunk cũ của file đã sửa/xóa"):
                delete_stale(client, changed, keep_ids)
                delete_files(client, deleted)
        finally:
            if bulk:
                with step("bật lại HNSW"):
                    client.update_collection(
                        VECTOR_DB_COLLECTION,
                        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
                    )

        with step("lưu index_state.json"):
            save_index_state(state_path, new_state)

        log(f"Index xong: {total_chunks} chunks, {len(changed)} file sửa, {len(deleted)} file xóa")
    finally:
        close_qdrant_client()


def reset_index(state_path: Path) -> None:
    from vectorstore.qdrant import (
        INDEX_FINGERPRINT_PATH, VECTOR_DB_COLLECTION, close_qdrant_client, get_qdrant_client,
    )

    client = get_qdrant_client()
    try:
        if client.collection_exists(VECTOR_DB_COLLECTION):
            client.delete_collection(VECTOR_DB_COLLECTION)
            log(f"đã xóa collection '{VECTOR_DB_COLLECTION}'")
    finally:
        close_qdrant_client()

    data = state_path.parent
    for f in (state_path, Path(INDEX_FINGERPRINT_PATH), data / "index_meta.json", data / "summary_cache.sqlite"):
        if f.is_file():
            f.unlink()
            log(f"đã xóa {f.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiểm tra + nạp index (có đo thời gian từng bước)")
    ap.add_argument("-s", "--source", help="đường dẫn repo cần index")
    ap.add_argument("--state-file", default=None, help="mặc định lấy theo config (ingestion.state_file)")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="xóa collection + state + fingerprint rồi index lại từ đầu")
    args = ap.parse_args()

    t_start = time.perf_counter()

    with step("setup_logging"):
        from config.load_config import setup_logging
        import logging
        setup_logging()
        for n in ("httpx", "httpcore"):
            logging.getLogger(n).setLevel(logging.WARNING)

    repo = resolve_repo(args.source)
    log(f"repo: {repo}")

    with step("import modules"):
        from ingestion.pipeline import STATE_FILE
        import embedding.embedder  # noqa: F401
        import ingestion.check  # noqa: F401
        import vectorstore.index  # noqa: F401
        import vectorstore.qdrant  # noqa: F401
        import vectorstore.upsert  # noqa: F401
    state_path = Path(args.state_file) if args.state_file else Path(STATE_FILE)

    try:
        if args.reset:
            with step("reset index"):
                reset_index(state_path)
        run_index(repo, state_path, args.batch_size, args.workers)
    except Exception as e:
        print(f"[run] lỗi: {type(e).__name__}: {e}", file=sys.stderr)
        print_summary(time.perf_counter() - t_start)
        return 1

    print_summary(time.perf_counter() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())