import logging
import sqlite3
import threading
import hashlib
import json
import torch
import numpy as np
from pathlib import Path
from typing import Callable
from config.load_config import load_settings, _resolve

import os

_CACHE_ROOT = Path(os.environ.get("MODEL_CACHE_DIR", Path(__file__).resolve().parent.parent / "data" / "model_cache"))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "hf"))
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_CACHE_ROOT / "fastembed"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer

from config.load_config import load_settings

settings = load_settings()
logger = logging.getLogger('embedding')

_cfg = settings['embedding']
SUMMARY_CACHE_PATH = str(_resolve(_cfg.get('summary_cache_path', 'data/summary_cache.sqlite')))
INDEX_META_PATH = str(_resolve(_cfg.get('index_meta_path', 'data/index_meta.json')))

def _resolve_device(device: str) -> str:
    """'auto' -> cuda nếu có GPU, ngược lại cpu."""
    if device != 'auto':
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_DEVICE = _resolve_device(_cfg.get('device', 'auto'))
EMBEDDING_MODEL = _cfg['model']
EMBEDDING_BATCH_SIZE = _cfg.get('batch_size', 32)
QUERY_PREFIX = _cfg.get('query_prefix', '')
DOCUMENT_PREFIX = _cfg.get('document_prefix', '')
TRUST_REMOTE_CODE = _cfg.get('trust_remote_code', False)
MAX_SEQ_LENGTH = _cfg.get('max_seq_length', 512)
SUMMARY_WORKERS = _cfg.get('summary_workers', 4)
SUMMARY_CACHE_PATH = _cfg.get('summary_cache_path', 'data/summary_cache.sqlite')
BACKEND = _cfg.get('backend', 'torch')
INDEX_META_PATH = _cfg.get('index_meta_path', 'data/index_meta.json')

model: SentenceTransformer | None = None
_model_lock = threading.Lock()
_cache_lock = threading.Lock()

ContextSummarizer = Callable[[dict], str]

_db: sqlite3.Connection | None = None

def get_model() -> SentenceTransformer:
    global model
    m = model
    if m is not None:
        return m

    with _model_lock:
        if model is None:
            logger.info(f"Loading embedding model {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
            kwargs: dict = dict(
                device=EMBEDDING_DEVICE,
                trust_remote_code=TRUST_REMOTE_CODE,
                backend=BACKEND,
            )
            if BACKEND == 'torch':
                kwargs['model_kwargs'] = {
                    'dtype': torch.float16 if EMBEDDING_DEVICE.startswith('cuda') else torch.float32,
                }

            try:
                loaded = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True, **kwargs)
            except Exception:
                loaded = SentenceTransformer(EMBEDDING_MODEL, **kwargs)

            if MAX_SEQ_LENGTH:
                current = loaded.max_seq_length
                loaded.max_seq_length = min(MAX_SEQ_LENGTH, current) if current else MAX_SEQ_LENGTH
            model = loaded

        assert model is not None
        return model


def use_model(
    name: str,
    trust_remote_code: bool = False,
    max_seq_length: int | None = None,
    query_prefix: str = '',
    document_prefix: str = '',
    backend: str = 'torch',
) -> None:
    """Đổi model embedding trong cùng tiến trình (dùng cho benchmark). Model mới được tải ở lần encode kế tiếp."""
    global model, EMBEDDING_MODEL, TRUST_REMOTE_CODE, MAX_SEQ_LENGTH, QUERY_PREFIX, DOCUMENT_PREFIX, BACKEND
    with _model_lock:
        EMBEDDING_MODEL = name
        TRUST_REMOTE_CODE = trust_remote_code
        MAX_SEQ_LENGTH = max_seq_length or MAX_SEQ_LENGTH
        QUERY_PREFIX, DOCUMENT_PREFIX = query_prefix, document_prefix
        BACKEND = backend
        model = None


def get_embedding_dim() -> int:
    m = get_model()
    if not hasattr(m, "get_embedding_dimension"):
        raise RuntimeError("Model does not provide get_embedding_dimension")

    dim = m.get_embedding_dimension()
    assert dim is not None
    return int(dim)



def check_index_compat(path: str | Path | None = None) -> None:
    """Gọi khi khởi động. Lần đầu: ghi model + số chiều. Các lần sau: báo lỗi nếu model hoặc số chiều đã đổi."""
    current = {'model': EMBEDDING_MODEL, 'dim': get_embedding_dim()}
    p = Path(path or INDEX_META_PATH)
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(current, indent=2), encoding='utf-8')
        return

    saved = json.loads(p.read_text(encoding='utf-8'))
    if saved != current:
        raise RuntimeError(
            f"Index were created by {saved}, differ from current {current}. "
            f"Delete old collection, file index_state and {p} run again."
        )


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        Path(SUMMARY_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(SUMMARY_CACHE_PATH, check_same_thread=False)
        conn.execute('CREATE TABLE IF NOT EXISTS summaries (key TEXT PRIMARY KEY, summary TEXT NOT NULL)')
        conn.commit()
        _db = conn  # Gán vào biến global sau khi đã setup xong

    assert _db is not None
    return _db


def _cache_get(key: str) -> str | None:
    with _cache_lock:
        row = _get_db().execute('SELECT summary FROM summaries WHERE key = ?', (key,)).fetchone()
    return row[0] if row else None


def _cache_set(key: str, summary: str) -> None:
    with _cache_lock:
        db = _get_db()
        db.execute('INSERT OR REPLACE INTO summaries (key, summary) VALUES (?, ?)', (key, summary))
        db.commit()


def build_contextual_text(
    chunk: dict,
    summarizer: ContextSummarizer | None = None,
) -> str:
    """Thêm câu tóm tắt ngữ cảnh (nếu có summarizer) vào trước nội dung chunk. Tóm tắt được cache trên đĩa theo content_hash."""
    content = chunk.get('content', '')
    if summarizer is None:
        return content

    cache_key = chunk.get('metadata', {}).get('content_hash') or hashlib.sha256(content.encode('utf-8')).hexdigest()
    summary = _cache_get(cache_key)

    if summary is None:
        try:
            summary = (summarizer(chunk) or '').strip()
        except Exception as e:
            logger.warning(f"Lỗi sinh contextual summary cho chunk {cache_key}: {e}")
            return content
        _cache_set(cache_key, summary)

    return f"# Context Summary: {summary}\n{content}" if summary else content


def _warn_truncation(texts: list[str]) -> None:
    m = get_model()
    limit = m.max_seq_length
    if not limit:
        return
    ids = m.tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
    n = sum(len(x) > limit for x in ids)
    if n:
        logger.warning(f"{n}/{len(texts)} văn bản dài hơn max_seq_length={limit}, phần cuối bị cắt")


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Trả về mảng float32 shape (n, dim), đã chuẩn hóa. Chỉ .tolist() khi ghi vào vector DB."""
    if not texts:
        return np.empty((0, get_embedding_dim()), dtype=np.float32)

    prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
    if prefix:
        texts = [prefix + t for t in texts]
    _warn_truncation(texts)
    return get_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)


def embed_chunks(
    chunks: list[dict],
    summarizer: ContextSummarizer | None = None,
) -> np.ndarray:
    """Trả về mảng (n, dim) theo đúng thứ tự chunk đầu vào. Summarizer được gọi song song."""
    texts: list[str] = []
    if summarizer is None or SUMMARY_WORKERS <= 1:
        texts = [build_contextual_text(c, summarizer) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:
            texts = list(pool.map(lambda c: build_contextual_text(c, summarizer), chunks))

    return embed_texts(texts)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], is_query=True)[0]


def warmup() -> None:
    """Chạy 1 câu ngắn, batch_size=1, bỏ qua tokenizer check/prefix để khởi tạo kernel/CUDA nhanh nhất."""
    get_model().encode(
        ['warmup'],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )