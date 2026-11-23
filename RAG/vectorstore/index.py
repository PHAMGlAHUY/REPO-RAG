import logging
import os
from functools import lru_cache
from pathlib import Path

from qdrant_client.models import PointStruct, SparseVector

from embedding.embedder import embed_texts
from vectorstore.qdrant import DENSE_VECTOR_NAME, SPARSE_DISABLE_STEMMER, SPARSE_MODEL, SPARSE_VECTOR_NAME

_CACHE_ROOT = Path(os.environ.get("MODEL_CACHE_DIR", Path(__file__).resolve().parent.parent / "data" / "model_cache"))
FASTEMBED_CACHE = os.environ.get("FASTEMBED_CACHE_PATH", str(_CACHE_ROOT / "fastembed"))

logger = logging.getLogger('vector_database')

SPARSE_BATCH_SIZE = 64

# Các trường dùng để lọc/hiển thị; phần còn lại (imports, syntax_errors, byte offset...) gom vào 'extra'
PAYLOAD_FIELDS = (
    'file_path', 'qualified_name', 'chunk_type', 'start_line', 'end_line',
    'parent_class_id', 'part_index', 'total_parts', 'calls',
    'has_syntax_error', 'content_hash',
)
_NOT_EXTRA = {*PAYLOAD_FIELDS, 'name'}


@lru_cache(maxsize=1)
def get_sparse_embedder():
    from fastembed import SparseTextEmbedding
    kwargs = {'disable_stemmer': True} if SPARSE_DISABLE_STEMMER else {}
    kwargs['cache_dir'] = FASTEMBED_CACHE
    try:
        # Ưu tiên cache cục bộ: không gọi HTTP lên HuggingFace
        return SparseTextEmbedding(model_name=SPARSE_MODEL, local_files_only=True, **kwargs)
    except Exception:
        # Chưa có cache (lần chạy đầu) hoặc fastembed bản cũ không có local_files_only -> tải từ mạng
        return SparseTextEmbedding(model_name=SPARSE_MODEL, **kwargs)


def warmup_sparse() -> None:
    next(iter(get_sparse_embedder().embed(['warmup'], batch_size=1)))


def to_sparse_vector(e) -> SparseVector:
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def _build_payload(chunk: dict) -> dict:
    md = chunk['metadata']
    payload = {k: md[k] for k in PAYLOAD_FIELDS if k in md}
    payload['name'] = (md.get('qualified_name') or '').rpartition('.')[2]
    parent = md.get('parent_class') or ''
    names = [n for n in (md.get('node_name') or payload['name']).split('+') if n]  # chunk gộp có node_name 'a+b'
    payload['names'] = names
    payload['qualified_names'] = [f'{parent}.{n}' if parent else n for n in names]
    payload['extra'] = {k: v for k, v in md.items() if k not in _NOT_EXTRA}
    payload['content'] = chunk['content']
    return payload


def create_qdrant_points(chunks: list[dict]) -> list[PointStruct]:
    if not chunks:
        return []
    texts = [c['content'] for c in chunks]
    dense = embed_texts(texts)
    sparse = get_sparse_embedder().embed(texts, batch_size=SPARSE_BATCH_SIZE)
    return [
        PointStruct(
            id=c['id'],
            vector={DENSE_VECTOR_NAME: d, SPARSE_VECTOR_NAME: to_sparse_vector(s)},
            payload=_build_payload(c),
        )
        for c, d, s in zip(chunks, dense, sparse, strict=True)
    ]