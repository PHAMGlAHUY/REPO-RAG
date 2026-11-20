import re
from functools import lru_cache
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchAny, MatchValue, PayloadSchemaType, Prefetch, Rrf, RrfQuery, SparseVector,
)

from embedding.embedder import embed_query
from vectorstore.index import get_sparse_embedder
from vectorstore.qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VECTOR_DB_COLLECTION

RRF_K = 60
_WEIGHTS = {'hybrid': (1, 1), 'dense': (1, 0), 'sparse': (0, 1)}  # (dense, sparse)

INDEX_SCHEMA = {
    'chunk_type': PayloadSchemaType.KEYWORD,
    'file_path': PayloadSchemaType.KEYWORD,
    'qualified_name': PayloadSchemaType.KEYWORD,
    'name': PayloadSchemaType.KEYWORD,  # tên ngắn: qualified_name.rpartition('.')[2]; indexer nên ghi sẵn
    'parent_class_id': PayloadSchemaType.KEYWORD,
    'has_syntax_error': PayloadSchemaType.BOOL,
}

_CAMEL_1 = re.compile(r'([a-z0-9])([A-Z])')
_CAMEL_2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_IDENTIFIER_RE = re.compile(r'(?<!\w)[A-Za-z_][A-Za-z0-9_]*(?!\w)')
_IDENT_QUERY = re.compile(r'(?=.*(?:[_.]|[a-z][A-Z]))[A-Za-z_][\w.]*')  # parse_config, parseConfig, Class.method


def is_identifier_query(query: str) -> bool:
    return bool(_IDENT_QUERY.fullmatch(query.strip()))


def split_identifier_tokens(identifier: str) -> list[str]:
    tokens: list[str] = []
    for part in identifier.split('_'):
        for piece in _CAMEL_1.sub(r'\1 \2', _CAMEL_2.sub(r'\1 \2', part)).split():
            if piece.lower() not in tokens:
                tokens.append(piece.lower())
    return tokens


def expand_identifier_query(query: str) -> str:
    seen = {t.lower() for t in query.split()}
    extra = []
    for ident in _IDENTIFIER_RE.findall(query):
        for sub in split_identifier_tokens(ident):
            if sub not in seen:
                seen.add(sub)
                extra.append(sub)
    return f"{query} {' '.join(extra)}" if extra else query


@lru_cache(maxsize=4096)
def _dense(query: str) -> tuple[float, ...]:
    return tuple(embed_query(query).tolist())


@lru_cache(maxsize=4096)
def _sparse(text: str) -> SparseVector:
    e = next(iter(get_sparse_embedder().query_embed(text)))
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def eq(key: str, value) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


def build_filter(
    chunk_type: str | list[str] | None = None,
    file_path: str | None = None,
    exclude_syntax_errors: bool = False,
    **equals,
) -> Filter | None:
    must = [eq(k, v) for k, v in equals.items()]
    if chunk_type:
        types = [chunk_type] if isinstance(chunk_type, str) else list(chunk_type)
        must.append(FieldCondition(key='chunk_type', match=MatchAny(any=types)))
    if file_path:
        must.append(eq('file_path', file_path))
    must_not = [eq('has_syntax_error', True)] if exclude_syntax_errors else None
    return Filter(must=must or None, must_not=must_not) if must or must_not else None


def search(
    client: QdrantClient,
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    mode: str = 'auto',  # 'auto' | 'hybrid' | 'dense' | 'sparse'
    flt: Filter | None = None,
):
    """Điểm dựng truy vấn duy nhất: dense + sparse, RRF có trọng số ngay trong Qdrant (cần qdrant >= 1.16).
    auto: query dạng định danh -> nghiêng sparse; query mô tả tự nhiên -> nghiêng dense."""
    if mode == 'auto':
        wd, ws = (1, 3) if is_identifier_query(query) else (2, 1)
    elif mode in _WEIGHTS:
        wd, ws = _WEIGHTS[mode]
    else:
        raise ValueError(f"mode must be 'auto', 'hybrid', 'dense' or 'sparse', got {mode!r}")

    prefetch = []
    if wd:
        prefetch.append(Prefetch(query=list(_dense(query)), using=DENSE_VECTOR_NAME, limit=prefetch_limit, filter=flt))
    if ws:
        prefetch.append(Prefetch(query=_sparse(expand_identifier_query(query)), using=SPARSE_VECTOR_NAME, limit=prefetch_limit, filter=flt))
    return client.query_points(
        collection_name=VECTOR_DB_COLLECTION,
        prefetch=prefetch,
        query=cast(Any, RrfQuery(rrf=Rrf(weights=[w for w in (wd, ws) if w], k=RRF_K))),
        limit=limit,
        with_payload=True,
    ).points


def exact_lookup(client: QdrantClient, query: str, limit: int, flt: Filter | None = None) -> list:
    """Query dạng định danh -> khớp chính xác qualified_name hoặc name (không có score)."""
    q = query.strip()
    if not is_identifier_query(q):
        return []
    hit = Filter(should=[eq('qualified_name', q), eq('name', q.rpartition('.')[2])])
    must = [hit, flt] if flt else [hit]
    return client.scroll(VECTOR_DB_COLLECTION, scroll_filter=Filter(must=must), limit=limit, with_payload=True)[0]


def ensure_payload_indexes(client: QdrantClient) -> None:
    for key, schema in INDEX_SCHEMA.items():
        client.create_payload_index(VECTOR_DB_COLLECTION, field_name=key, field_schema=schema)


def backfill_name(client: QdrantClient, batch: int = 512) -> None:
    """Chạy một lần cho dữ liệu đã index: ghi payload `name` từ qualified_name."""
    by_name: dict[str, list] = {}
    offset = None
    while True:
        points, offset = client.scroll(VECTOR_DB_COLLECTION, limit=batch, offset=offset, with_payload=['qualified_name'])
        for p in points:
            qn = (p.payload or {}).get('qualified_name')
            if isinstance(qn, str):
                name = qn.rpartition('.')[2]
                by_name.setdefault(name, []).append(p.id)
        if offset is None:
            break
    for name, ids in by_name.items():
        client.set_payload(VECTOR_DB_COLLECTION, payload={'name': name}, points=ids)