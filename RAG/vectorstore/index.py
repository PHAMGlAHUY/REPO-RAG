import logging
import os
from pathlib import Path

from functools import lru_cache

from qdrant_client.models import PointStruct, SparseVector

from embedding.embedder import embed_texts, build_contextual_text
from vectorstore.qdrant import DENSE_VECTOR_NAME, SPARSE_DISABLE_STEMMER, SPARSE_MODEL, SPARSE_VECTOR_NAME

FASTEMBED_CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "model_cache" / "fastembed"),
)

logger = logging.getLogger('vector_database')

SPARSE_BATCH_SIZE = 64


# Các trường dùng để lọc/hiển thị; phần còn lại (imports, calls, syntax_errors, byte offset...) gom vào 'extra'
PAYLOAD_FIELDS = (
    'file_path', 'chunk_type', 'qualified_name', 'parent_class_id', 'content_hash',
    'has_syntax_error', 'signature', 'start_line', 'end_line', 'part_index', 'total_parts', 'is_async',
)




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


def _embed_sparse(texts: list[str]) -> list[SparseVector]:
    embedder = get_sparse_embedder()
    result = []
    for i in range(0, len(texts), SPARSE_BATCH_SIZE):
        batch = texts[i:i + SPARSE_BATCH_SIZE]
        result.extend(
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in embedder.embed(batch, batch_size=SPARSE_BATCH_SIZE)
        )
    return result


def _build_payload(chunk: dict) -> dict:
    md = chunk['metadata']
    payload = {k: md[k] for k in PAYLOAD_FIELDS if k in md}
    payload['extra'] = {k: v for k, v in md.items() if k not in PAYLOAD_FIELDS}
    payload['content'] = chunk['content']
    return payload


def create_qdrant_points(chunks: list[dict], summarizer=None) -> list[PointStruct]:
    if not chunks:
        logger.warning("No chunks provided to build Qdrant points.")
        return []

    texts = [build_contextual_text(c, summarizer) for c in chunks]
    dense_vectors = embed_texts(texts)
    sparse_vectors = _embed_sparse(texts)

    return [
        PointStruct(
            id=chunk['id'],
            vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
            payload=_build_payload(chunk),
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ]