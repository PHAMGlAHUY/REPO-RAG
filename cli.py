"""Giao diện dòng lệnh hỏi đáp trên repo đã index: truy xuất (hybrid + rerank) -> dựng prompt -> Ollama (streaming)."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from config.load_config import load_settings
from ingestion.chunking.splitter import count_tokens_approx
from retrieval.retriever import RetrievedChunk, default_reranker, retrieve
from vectorstore.qdrant import VECTOR_DB_COLLECTION, close_qdrant_client, get_qdrant_client

logger = logging.getLogger('chat')

MODES = ('auto', 'hybrid', 'dense', 'sparse')
MAX_HISTORY = 3          # số lượt hỏi đáp gần nhất đưa lại cho LLM
PROMPT_SAFETY = 200      # token dự phòng cho định dạng prompt
SYSTEM_PROMPT = (
    "Bạn là trợ lý giúp hiểu mã nguồn của một repository. "
    "Chỉ trả lời dựa trên các đoạn mã trong phần Ngữ cảnh; nếu không đủ thông tin thì nói rõ là chưa đủ. "
    "Khi dẫn chứng, ghi số đoạn dạng [1], [2]. Trả lời bằng ngôn ngữ của câu hỏi, ngắn gọn và chính xác."
)
HELP = """Lệnh:
  /help             hiện trợ giúp
  /exit, /quit      thoát
  /reset            xóa lịch sử hội thoại
  /topk N           số đoạn mã lấy làm ngữ cảnh
  /mode M           auto | hybrid | dense | sparse
  /rerank on|off    bật/tắt rerank
  /sources          in lại nguồn của câu trả lời gần nhất"""


@dataclass
class _State:
    top_k: int
    mode: str = 'auto'
    rerank: bool = True
    history: list[tuple[str, str]] = field(default_factory=list)
    last_sources: list[tuple[RetrievedChunk, bool]] = field(default_factory=list)


# ---------------------------------------------------------------- LLM (Ollama)
def _ollama_ready(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def _preload_llm(cfg: dict) -> None:
    """Nạp sẵn model vào bộ nhớ Ollama để câu hỏi đầu tiên không phải chờ."""
    try:
        httpx.post(
            f"{cfg['base_url']}/api/generate",
            json={'model': cfg['model_name'], 'keep_alive': cfg.get('keep_alive', '30m')},
            timeout=float(cfg.get('timeout', 180)),
        )
    except httpx.HTTPError:
        pass


def _stream_chat(cfg: dict, messages: list[dict]):
    payload = {
        'model': cfg['model_name'],
        'messages': messages,
        'stream': True,
        'keep_alive': cfg.get('keep_alive', '30m'),
        'options': {
            'temperature': cfg.get('temperature', 0.2),
            'num_predict': int(cfg.get('max_tokens', 1024)),
            'num_ctx': int(cfg.get('num_ctx', 8192)),
        },
    }
    timeout = httpx.Timeout(float(cfg.get('timeout', 180)))
    with httpx.stream('POST', f"{cfg['base_url']}/api/chat", json=payload, timeout=timeout) as r:
        if r.status_code >= 400:
            r.read()
            raise RuntimeError(f"Ollama trả về {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get('error'):
                raise RuntimeError(data['error'])
            piece = (data.get('message') or {}).get('content', '')
            if piece:
                yield piece
            if data.get('done'):
                break


# ---------------------------------------------------------------- prompt
def _location(c: RetrievedChunk) -> str:
    if c.start_line is None:
        return c.file_path or ''
    return f"{c.file_path}:{c.start_line}-{c.end_line}"


def _format_chunk(n: int, c: RetrievedChunk, related: bool) -> str:
    title = f"[{n}] {_location(c)} — {c.qualified_name or c.chunk_type}"
    if related:
        title += " (hàm được gọi)"
    return f"{title}\n```python\n{c.content}\n```"


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[tuple[str, str]],
    llm_cfg: dict,
) -> tuple[list[dict], list[tuple[RetrievedChunk, bool]]]:
    """Dựng messages sao cho vừa num_ctx - max_tokens. Trả về (messages, [(chunk, là_hàm_được_gọi)])."""
    budget = int(llm_cfg.get('num_ctx', 8192)) - int(llm_cfg.get('max_tokens', 1024)) - PROMPT_SAFETY
    budget -= count_tokens_approx(SYSTEM_PROMPT) + count_tokens_approx(question) + 50

    hist = list(history[-MAX_HISTORY:])
    hist_cap = budget // 4
    while hist and sum(count_tokens_approx(q) + count_tokens_approx(a) for q, a in hist) > hist_cap:
        hist.pop(0)
    budget -= sum(count_tokens_approx(q) + count_tokens_approx(a) for q, a in hist)

    items: list[tuple[RetrievedChunk, bool]] = [(c, False) for c in chunks]
    seen = {c.id for c in chunks}
    for c in chunks:
        for r in c.related_calls:
            if r.id not in seen:
                seen.add(r.id)
                items.append((r, True))

    parts: list[str] = []
    included: list[tuple[RetrievedChunk, bool]] = []
    used = 0
    for c, related in items:
        text = _format_chunk(len(included) + 1, c, related)
        tokens = count_tokens_approx(text)
        if used + tokens > budget:
            if included:
                continue  # đoạn sau (xếp hạng thấp hơn) có thể nhỏ hơn nên vẫn thử tiếp
            text = text[:max(200, (budget - used) * 3)]  # đoạn đầu quá lớn: cắt cho vừa
            tokens = count_tokens_approx(text)
        parts.append(text)
        included.append((c, related))
        used += tokens

    messages: list[dict] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for q, a in hist:
        messages += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    messages.append({'role': 'user', 'content': "Ngữ cảnh:\n\n" + "\n\n".join(parts) + f"\n\nCâu hỏi: {question}"})
    return messages, included


# ---------------------------------------------------------------- hiển thị
def _print_sources(sources: list[tuple[RetrievedChunk, bool]]) -> None:
    if not sources:
        return
    print("\nNguồn:")
    for i, (c, related) in enumerate(sources, 1):
        if c.exact:
            note = "khớp tên"
        elif c.rerank_score is not None:
            note = f"rerank {c.rerank_score:.2f}"
        elif c.score is not None:
            note = f"score {c.score:.3f}"
        else:
            note = ""
        extra = ", ".join(x for x in (note, "hàm được gọi" if related else "") if x)
        print(f"  [{i}] {_location(c)}  {c.qualified_name or c.chunk_type}" + (f"  ({extra})" if extra else ""))


def _quiet_console() -> None:
    """Chế độ CLI: handler console chỉ hiện WARNING trở lên để log không chen vào hội thoại."""
    loggers = [logging.getLogger(), *(l for l in logging.Logger.manager.loggerDict.values() if isinstance(l, logging.Logger))]
    for lg in loggers:
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.WARNING)


def _warm_up() -> None:
    from embedding.embedder import warmup
    from vectorstore.index import warmup_sparse

    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in [ex.submit(warmup), ex.submit(warmup_sparse)]:
            f.result()


# ---------------------------------------------------------------- lệnh
def _command(line: str, st: _State) -> bool:
    """Xử lý lệnh bắt đầu bằng '/'. Trả về True nếu cần thoát."""
    cmd, _, arg = line.partition(' ')
    cmd, arg = cmd.lower(), arg.strip()
    if cmd in ('/exit', '/quit'):
        return True
    if cmd == '/help':
        print(HELP)
    elif cmd == '/reset':
        st.history.clear()
        print("Đã xóa lịch sử hội thoại.")
    elif cmd == '/topk':
        if arg.isdigit() and int(arg) > 0:
            st.top_k = int(arg)
            print(f"top_k = {st.top_k}")
        else:
            print("Dùng: /topk N (N > 0)")
    elif cmd == '/mode':
        if arg in MODES:
            st.mode = arg
            print(f"mode = {st.mode}")
        else:
            print(f"Dùng: /mode {' | '.join(MODES)}")
    elif cmd == '/rerank':
        if arg in ('on', 'off'):
            st.rerank = arg == 'on'
            print(f"rerank = {arg}")
        else:
            print("Dùng: /rerank on|off")
    elif cmd == '/sources':
        _print_sources(st.last_sources) if st.last_sources else print("Chưa có nguồn nào.")
    else:
        print("Lệnh không hợp lệ. Gõ /help để xem danh sách lệnh.")
    return False


# ---------------------------------------------------------------- vòng lặp chính
def _answer(question: str, st: _State, client, rerank_fn, llm_cfg: dict, llm_ok: bool) -> None:
    t0 = time.perf_counter()
    chunks = retrieve(
        client, question,
        top_k=st.top_k,
        mode=st.mode,
        rerank=st.rerank and rerank_fn is not None,
        reranker=rerank_fn,
        merge_parts=True,
        expand_calls=True,
    )
    t_retrieve = time.perf_counter() - t0
    if not chunks:
        print("Không tìm thấy đoạn mã liên quan.")
        return

    messages, sources = build_messages(question, chunks, st.history, llm_cfg)
    st.last_sources = sources

    if not llm_ok:
        print("(Không kết nối được Ollama, chỉ hiện kết quả truy xuất)")
        _print_sources(sources)
        print(f"\n[truy xuất {t_retrieve:.2f}s]")
        return

    t0 = time.perf_counter()
    answer: list[str] = []
    print()
    try:
        for piece in _stream_chat(llm_cfg, messages):
            answer.append(piece)
            print(piece, end='', flush=True)
    except KeyboardInterrupt:
        print("\n[đã dừng]")
    except (httpx.HTTPError, RuntimeError) as e:
        print(f"\n[lỗi LLM] {e}")
    print()
    if answer:
        st.history.append((question, ''.join(answer)))
    _print_sources(sources)
    print(f"\n[truy xuất {t_retrieve:.2f}s | LLM {time.perf_counter() - t0:.2f}s]")


def run_cli(repo: Path | None = None, top_k: int | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    try:
        import readline  # noqa: F401  (lịch sử + phím mũi tên trên Linux/macOS)
    except ImportError:
        pass

    settings = load_settings()
    llm_cfg = settings.get('llm') or {}
    st = _State(top_k=top_k or int((settings.get('retrieval') or {}).get('top_k', 5)))
    _quiet_console()

    client = get_qdrant_client()
    try:
        print("Đang nạp model ...", flush=True)
        _warm_up()

        rerank_fn = None
        try:
            rerank_fn = default_reranker()  # Jina API; None nếu tắt trong config hoặc thiếu API key
        except Exception as e:
            print(f"(Không tạo được reranker: {e})")
        if rerank_fn is None:
            st.rerank = False
            print("(Rerank tắt: kiểm tra reranking.enabled và biến môi trường JINA_API_KEY)")

        llm_ok = _ollama_ready(llm_cfg.get('base_url', 'http://localhost:11434'))
        if llm_ok:
            threading.Thread(target=_preload_llm, args=(llm_cfg,), daemon=True).start()
        else:
            print(f"(Không kết nối được Ollama tại {llm_cfg.get('base_url')}; chỉ hiện kết quả truy xuất)")

        print(f"\nRepo: {repo or '-'} | collection: {VECTOR_DB_COLLECTION} | top_k={st.top_k} | mode={st.mode} "
              f"| rerank={'on' if st.rerank else 'off'}")
        print("Nhập câu hỏi, /help để xem lệnh, /exit để thoát.")

        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith('/'):
                if _command(line, st):
                    break
                continue
            try:
                _answer(line, st, client, rerank_fn, llm_cfg, llm_ok)
            except KeyboardInterrupt:
                print("\n[đã dừng]")
            except Exception as e:
                logger.exception("Lỗi khi xử lý câu hỏi")
                print(f"[lỗi] {type(e).__name__}: {e}")
    finally:
        close_qdrant_client()