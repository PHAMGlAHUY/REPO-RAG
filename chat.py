"""Chạy thử toàn bộ RAG pipeline: chunking -> embed -> index (Qdrant) -> retrieve -> LLM.

Cách dùng (chạy từ thư mục gốc dự án):

    python chat.py index <đường_dẫn_repo> [--recreate] [--no-prune]   # nạp repo vào Qdrant
    python chat.py chat [--rerank] [--mode hybrid] [--debug]   # hỏi đáp

Trong lúc chat:  /debug bật/tắt hiển thị chunk, /clear xoá lịch sử, /quit thoát.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config.load_config import load_settings, setup_logging

logger = logging.getLogger("chat")


# =========================================================================== #
# BƯỚC 1: INGESTION + EMBED + INDEX
# =========================================================================== #
def run_ingestion(repo_path: Path) -> list[dict]:
    """Quét repo, parse + chunk code -> danh sách chunk."""
    from ingestion.pipeline import iter_repository_chunks

    return iter_repository_chunks(repo_root=str(repo_path))


def run_indexing(chunks: list[dict], recreate: bool, prune: bool = True) -> None:
    """Embed (dense + sparse) và upsert chunk vào Qdrant, từng file một.

    upsert_file_chunks cũng xoá các chunk cũ của file đó (id đổi khi start_line đổi).
    prune=True: index xong thì xoá chunk của các file có trong Qdrant nhưng không còn chunk nào trong lần quét này
    (file bị xoá, bị ignore hoặc không còn nội dung). Chỉ dùng khi collection chứa đúng một repo,
    vì file_path là đường dẫn tương đối nên file của repo khác sẽ bị coi là "thừa".
    """
    from vectorstore.index import get_sparse_embedder
    from vectorstore.qdrant import VECTOR_DB_COLLECTION, ensure_collection, get_qdrant_client
    from vectorstore.upsert import delete_stale_files, upsert_file_chunks

    client = get_qdrant_client()
    if recreate and client.collection_exists(VECTOR_DB_COLLECTION):
        logger.info("Xoá collection cũ: %s", VECTOR_DB_COLLECTION)
        client.delete_collection(VECTOR_DB_COLLECTION)
    ensure_collection(client)

    sparse = get_sparse_embedder()  # lần đầu sẽ tải model BM25 của fastembed

    by_file: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk["metadata"]["file_path"], []).append(chunk)

    total = len(by_file)
    for i, (file_path, file_chunks) in enumerate(by_file.items(), start=1):
        upsert_file_chunks(client, file_chunks, sparse)
        if i % 10 == 0 or i == total:
            logger.info("Đã index %d/%d file", i, total)

    if prune and not recreate:
        stale = delete_stale_files(client, by_file)
        if stale:
            shown = ", ".join(stale[:10]) + (" ..." if len(stale) > 10 else "")
            logger.info("Đã dọn chunk của %d file không còn trong repo: %s", len(stale), shown)

    logger.info("Collection %s hiện có %d điểm", VECTOR_DB_COLLECTION, client.count(VECTOR_DB_COLLECTION).count)


def cmd_index(args: argparse.Namespace, settings: dict) -> None:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        sys.exit(f"Không tìm thấy thư mục repo: {repo}")

    t0 = time.perf_counter()
    logger.info("[1/2] Ingestion + chunking: %s", repo)
    chunks = run_ingestion(repo)
    logger.info("→ %d chunk (%.1fs)", len(chunks), time.perf_counter() - t0)
    if not chunks:
        sys.exit("Không có chunk nào được tạo. Kiểm tra supported_extensions / ignore_dirs.")

    t1 = time.perf_counter()
    logger.info("[2/2] Embedding + index vào Qdrant (recreate=%s)", args.recreate)
    run_indexing(chunks, recreate=args.recreate, prune=not args.no_prune)
    logger.info("→ xong (%.1fs). Bây giờ chạy: python chat.py chat", time.perf_counter() - t1)


# =========================================================================== #
# BƯỚC 2: RETRIEVE (+ RERANK tuỳ chọn) + LLM
# =========================================================================== #
def load_reranker(settings: dict):
    from sentence_transformers import CrossEncoder

    cfg = settings["reranking"]
    device = cfg.get("device", "cpu")
    logger.info("Nạp reranker %s (%s)", cfg["model"], device)
    return CrossEncoder(cfg["model"], device=device)


def rerank(reranker, query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return chunks
    scores = reranker.predict([(query, c["content"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        out.append({**chunk, "score": float(score)})
    return out


def warm_up(llm) -> None:
    """Nạp trước model embedding và LLM để câu hỏi đầu tiên không bị chậm."""
    from embedding.embedder import warmup
    from llm.llm import LLMError

    logger.info("Nạp trước model embedding...")
    warmup()
    try:
        logger.info("Nạp trước LLM...")
        llm.generate([{"role": "user", "content": "ping"}], max_tokens=1)
    except LLMError as exc:
        print(f"[Cảnh báo] Chưa gọi được LLM: {exc}\nKiểm tra Ollama đang chạy và model đã pull chưa.")


def print_sources(chunks: list[dict]) -> None:
    print("\n  Nguồn:")
    for c in chunks:
        m = c["metadata"]
        start = m.get("part_start_line", m.get("start_line"))
        end = m.get("part_end_line", m.get("end_line"))
        print(f"   - {m.get('file_path')}:{start}-{end}  ({m.get('node_name')})  score={c['score']:.4f}")


def cmd_chat(args: argparse.Namespace, settings: dict) -> None:
    from llm.llm import LLMClient, LLMError
    from retrieval.retriever import retrieve
    from vectorstore.index import get_sparse_embedder
    from vectorstore.qdrant import VECTOR_DB_COLLECTION, get_qdrant_client

    client = get_qdrant_client()
    if not client.collection_exists(VECTOR_DB_COLLECTION) or client.count(VECTOR_DB_COLLECTION).count == 0:
        sys.exit(f"Collection {VECTOR_DB_COLLECTION} chưa có dữ liệu. Chạy trước: python chat.py index <đường_dẫn_repo>")
    sparse = get_sparse_embedder()
    llm = LLMClient.from_config(settings)
    reranker = load_reranker(settings) if args.rerank else None
    warm_up(llm)

    top_k = (settings.get("retrieval") or {}).get("top_k", 10)
    rerank_cfg = settings.get("reranking") or {}
    final_k = rerank_cfg.get("top_k", 5) if reranker else top_k
    # Reranker cần nhiều ứng viên hơn số chunk giữ lại thì mới có gì để chọn lại
    fetch_k = max(rerank_cfg.get("candidates", final_k * 4), final_k) if reranker else top_k
    debug = args.debug
    history: list[dict] = []

    print("RAG chat — gõ câu hỏi về repo ( /debug, /clear, /quit ).")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/clear":
            history.clear()
            print("Đã xoá lịch sử.")
            continue
        if question == "/debug":
            debug = not debug
            print(f"debug = {debug}")
            continue

        try:
            t0 = time.perf_counter()
            query = llm.rewrite_query(question, history)
            if debug and query != question:
                print(f"  [query viết lại] {query}")

            chunks = retrieve(client, query, sparse, top_k=fetch_k, mode=args.mode)
            if reranker:
                chunks = rerank(reranker, query, chunks, final_k)
            t_retrieve = time.perf_counter() - t0

            if debug:
                print_sources(chunks)
                print(f"  [retrieve{' + rerank' if reranker else ''}: {t_retrieve:.2f}s, {len(chunks)} chunk]")

            print("\nBot: ", end="", flush=True)
            answer = ""
            for token in llm.answer_stream(question, chunks, history):
                print(token, end="", flush=True)
                answer += token
            print()
            if not debug:
                print_sources(chunks)

            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        except KeyboardInterrupt:  # Ctrl+C khi đang xử lý: dừng câu hỏi này, không thoát chương trình
            print("\n[Đã dừng câu hỏi hiện tại]")
        except LLMError as exc:
            print(f"\n[Lỗi LLM] {exc}\nKiểm tra Ollama đang chạy và model đã pull chưa.")
        except Exception:  # noqa: BLE001 - không để một lỗi làm chết vòng chat
            logger.exception("Lỗi khi xử lý câu hỏi")
            print("\n[Lỗi] Xem logs/application.log để biết chi tiết.")


# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="Test RAG pipeline cho code repo")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Chunk + embed + index một repo vào Qdrant")
    p_index.add_argument("repo", help="Đường dẫn thư mục repo cần nạp")
    p_index.add_argument("--recreate", action="store_true", help="Xoá collection cũ rồi tạo lại")
    p_index.add_argument("--no-prune", action="store_true",
                         help="Không xoá chunk của file đã biến mất khỏi repo (dùng khi collection chứa nhiều repo)")

    p_chat = sub.add_parser("chat", help="Hỏi đáp trên repo đã index")
    p_chat.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p_chat.add_argument("--rerank", action="store_true", help="Rerank bằng cross-encoder")
    p_chat.add_argument("--debug", action="store_true", help="Hiện chunk, thời gian, query viết lại")

    args = parser.parse_args()
    setup_logging()
    settings = load_settings()

    {"index": cmd_index, "chat": cmd_chat}[args.command](args, settings)


if __name__ == "__main__":
    main()