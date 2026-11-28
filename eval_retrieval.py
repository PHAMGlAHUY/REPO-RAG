"""Đánh giá retrieval: python eval_retrieval.py [--test test.json] [--top-k 5] [--mode auto] [--no-rerank]"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "RAG"))


def is_hit(chunk, case: dict) -> bool:
    """Đúng khi file_path kết thúc bằng case['file'] và (nếu có) case['name'] nằm trong tên của chunk."""
    if not (chunk.file_path or "").replace("\\", "/").endswith(case["file"]):
        return False
    name = case.get("name")
    return not name or name in chunk.names or name in chunk.qualified_names


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(ROOT / "test.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mode", default="auto", choices=["auto", "hybrid", "dense", "sparse"])
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    cases = json.loads(Path(args.test).read_text(encoding="utf-8"))

    import logging
    from config.load_config import setup_logging
    setup_logging()
    for n in ("httpx", "httpcore"):
        logging.getLogger(n).setLevel(logging.WARNING)

    from embedding.embedder import warmup
    from retrieval.retriever import default_reranker, retrieve
    from vectorstore.index import warmup_sparse
    from vectorstore.qdrant import close_qdrant_client, get_qdrant_client

    client = get_qdrant_client()
    try:
        t0 = time.perf_counter()
        warmup()
        warmup_sparse()
        reranker = None if args.no_rerank else default_reranker()
        opts = dict(top_k=args.top_k, mode=args.mode, rerank=reranker is not None, reranker=reranker)
        retrieve(client, "warmup", **opts)  # làm nóng cả kết nối rerank
        print(f"warmup: {time.perf_counter() - t0:.2f}s | mode={args.mode} top_k={args.top_k} rerank={'on' if reranker else 'off'}\n")

        ranks: list[int | None] = []
        times: list[float] = []
        for i, case in enumerate(cases, 1):
            t0 = time.perf_counter()
            chunks = retrieve(client, case["question"], **opts)
            dt = time.perf_counter() - t0
            rank = next((r for r, c in enumerate(chunks, 1) if is_hit(c, case)), None)
            ranks.append(rank)
            times.append(dt)
            print(f"[{i:02d}] {f'HIT@{rank}' if rank else 'MISS':<6} {dt:6.2f}s  {case['question'][:70]}")
            if rank is None:
                print(f"       mong đợi: {case['file']} {case.get('name', '')} | nhận: "
                      + ", ".join(f"{c.file_path}:{c.qualified_name}" for c in chunks[:3]))
    finally:
        close_qdrant_client()

    n = len(cases)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    print("\n========== KẾT QUẢ ==========")
    print(f"Số câu hỏi : {n}")
    print(f"Hit@1      : {hit(1):.1%}")
    print(f"Hit@3      : {hit(3):.1%}")
    print(f"Hit@{args.top_k:<7}: {hit(args.top_k):.1%}")
    print(f"MRR        : {sum(1 / r for r in ranks if r) / n:.3f}")
    print(f"Thời gian  : tb {statistics.mean(times):.2f}s | p50 {pct(times, .5):.2f}s | "
          f"p95 {pct(times, .95):.2f}s | max {max(times):.2f}s | tổng {sum(times):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())