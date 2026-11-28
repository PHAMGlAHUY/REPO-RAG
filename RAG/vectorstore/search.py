import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchAny, MatchValue, QuantizationSearchParams,
    SearchParams, SparseVector,
)

from embedding.embedder import embed_query
from vectorstore.index import get_sparse_embedder, to_sparse_vector
from vectorstore.qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VECTOR_DB_COLLECTION

_WEIGHTS = {'hybrid': (1, 1), 'dense': (1, 0), 'sparse': (0, 1)}  # (dense, sparse): số kết quả lấy xen kẽ mỗi lượt
_DENSE_PARAMS = SearchParams(quantization=QuantizationSearchParams(rescore=True, oversampling=2.0))
_POOL = ThreadPoolExecutor(max_workers=2)  # chạy truy vấn dense và sparse song song

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
    return to_sparse_vector(next(iter(get_sparse_embedder().query_embed(text))))


def eq(key: str, value) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


def build_filter(
    chunk_type: str | list[str] | None = None,
    file_path: str | None = None,
    exclude_syntax_errors: bool = False,
) -> Filter | None:
    must = []
    if chunk_type:
        types = [chunk_type] if isinstance(chunk_type, str) else list(chunk_type)
        must.append(FieldCondition(key='chunk_type', match=MatchAny(any=types)))
    if file_path:
        must.append(eq('file_path', file_path))
    must_not = [eq('has_syntax_error', True)] if exclude_syntax_errors else None
    return Filter(must=must or None, must_not=must_not) if must or must_not else None


def _search_dense(client: QdrantClient, query: str, limit: int, flt: Filter | None) -> list:
    return client.query_points(
        collection_name=VECTOR_DB_COLLECTION, query=list(_dense(query)), using=DENSE_VECTOR_NAME,
        query_filter=flt, search_params=_DENSE_PARAMS, limit=limit, with_payload=True,
    ).points


def _search_sparse(client: QdrantClient, query: str, limit: int, flt: Filter | None) -> list:
    return client.query_points(
        collection_name=VECTOR_DB_COLLECTION, query=_sparse(expand_identifier_query(query)),
        using=SPARSE_VECTOR_NAME, query_filter=flt, limit=limit, with_payload=True,
    ).points


def _merge_weighted(dense: list, sparse: list, wd: int, ws: int, limit: int) -> list:
    """Gộp hai danh sách bằng cách lấy xen kẽ (wd điểm dense, rồi ws điểm sparse mỗi lượt), bỏ trùng id.
    Không dùng RRF: thứ tự cuối cùng do reranker quyết định; thứ tự này chỉ để chọn ứng viên."""
    out, seen = [], set()
    di = si = 0
    while len(out) < limit and (di < len(dense) or si < len(sparse)):
        for _ in range(wd):
            if di < len(dense):
                p = dense[di]
                di += 1
                if p.id not in seen:
                    seen.add(p.id)
                    out.append(p)
        for _ in range(ws):
            if si < len(sparse):
                p = sparse[si]
                si += 1
                if p.id not in seen:
                    seen.add(p.id)
                    out.append(p)
    return out[:limit]


def search(
    client: QdrantClient,
    query: str,
    limit: int = 10,
    mode: str = 'auto',  # 'auto' | 'hybrid' | 'dense' | 'sparse'
    flt: Filter | None = None,
):
    """Dense + sparse chạy song song rồi gộp xen kẽ theo trọng số (không RRF).
    auto: query dạng định danh -> nghiêng sparse; query mô tả tự nhiên -> nghiêng dense.
    Lưu ý: score của dense và sparse khác thang đo nên không so sánh trực tiếp được."""
    if mode == 'auto':
        wd, ws = (1, 3) if is_identifier_query(query) else (2, 1)
    elif mode in _WEIGHTS:
        wd, ws = _WEIGHTS[mode]
    else:
        raise ValueError(f"mode must be 'auto', 'hybrid', 'dense' or 'sparse', got {mode!r}")

    if not ws:
        return _search_dense(client, query, limit, flt)
    if not wd:
        return _search_sparse(client, query, limit, flt)

    sparse_future = _POOL.submit(_search_sparse, client, query, limit, flt)
    dense = _search_dense(client, query, limit, flt)
    return _merge_weighted(dense, sparse_future.result(), wd, ws, limit)


def exact_lookup(client: QdrantClient, query: str, limit: int, flt: Filter | None = None) -> list:
    """Query dạng định danh -> khớp chính xác qualified_name hoặc name (không có score)."""
    q = query.strip()
    if not is_identifier_query(q):
        return []
    hit = Filter(should=[eq('qualified_names', q), eq('names', q.rpartition('.')[2])])
    must = [hit, flt] if flt else [hit]
    return client.scroll(VECTOR_DB_COLLECTION, scroll_filter=Filter(must=must), limit=limit, with_payload=True)[0]