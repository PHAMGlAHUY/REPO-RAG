import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config.load_config import load_settings
from vectorstore.qdrant import VECTOR_DB_COLLECTION
from vectorstore.search import build_filter, eq, exact_lookup, search

RETRIEVAL_TOP_K = (load_settings().get('retrieval') or {}).get('top_k', 5)
RERANK_CANDIDATES = 40
CALLABLE_TYPES = ['function', 'method']

# (query, chunks) -> điểm cho từng chunk, cùng thứ tự
RerankerFn = Callable[[str, list['RetrievedChunk']], list[float]]

# giả định tên payload do chunker ghi; phần còn lại nằm trong metadata
_PAYLOAD_FIELDS = (
    'file_path', 'qualified_name', 'name', 'chunk_type', 'start_line', 'end_line',
    'is_async', 'parent_class_id', 'part_index', 'part_total', 'calls',
)


@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float | None = None
    rerank_score: float | None = None
    file_path: str | None = None
    qualified_name: str | None = None
    name: str | None = None
    chunk_type: str = ''
    start_line: int | None = None
    end_line: int | None = None
    is_async: bool = False
    parent_class_id: str | None = None
    part_index: int | None = None
    part_total: int | None = None
    calls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parent: 'RetrievedChunk | None' = None
    related_calls: list['RetrievedChunk'] = field(default_factory=list)

    def __post_init__(self):
        self.calls = self.calls or []
        self.chunk_type = self.chunk_type or ''
        self.is_async = bool(self.is_async)
        self.name = self.name or (self.qualified_name or '').rpartition('.')[2] or None

    @classmethod
    def from_point(cls, p) -> 'RetrievedChunk':
        payload = dict(p.payload or {})
        payload = {**(payload.pop('extra', None) or {}), **payload}
        content = payload.pop('content', '')
        known = {k: payload.pop(k) for k in _PAYLOAD_FIELDS if k in payload}
        return cls(id=str(p.id), content=content, score=getattr(p, 'score', None), metadata=payload, **known)


def cross_encoder_reranker(model: str = 'BAAI/bge-reranker-v2-m3', batch_size: int = 16) -> RerankerFn:
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model)

    def rerank(query: str, chunks: list[RetrievedChunk]) -> list[float]:
        pairs = [(query, f'{c.qualified_name or ""}\n{c.content}') for c in chunks]
        return ce.predict(pairs, batch_size=batch_size).tolist()

    return rerank


def _dedup(chunks: list[RetrievedChunk], max_per_file: int | None = None) -> list[RetrievedChunk]:
    seen, per_file, out = set(), Counter(), []
    for c in chunks:
        key = (c.file_path, c.qualified_name or c.id)
        if key in seen or (max_per_file and per_file[c.file_path] >= max_per_file):
            continue
        seen.add(key)
        per_file[c.file_path] += 1
        out.append(c)
    return out


def _expand_parents(client: QdrantClient, chunks: list[RetrievedChunk]) -> None:
    parent_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_type == 'method' and chunk.parent_class_id:
            parent_ids.add(str(chunk.parent_class_id))
    ids = sorted(parent_ids)
    if not ids:
        return

    parents = {
        str(p.id): RetrievedChunk.from_point(p)
        for p in client.retrieve(VECTOR_DB_COLLECTION, ids=ids, with_payload=True)
    }
    for chunk in chunks:
        if chunk.chunk_type == 'method' and chunk.parent_class_id is not None:
            chunk.parent = parents.get(str(chunk.parent_class_id))


def _merge_parts(client: QdrantClient, chunks: list[RetrievedChunk], window: int = 1) -> None:
    parted = [c for c in chunks if (c.part_total or 0) > 1 and c.part_index is not None]
    if not parted:
        return
    flt = Filter(should=[Filter(must=[eq('file_path', c.file_path), eq('qualified_name', c.qualified_name)]) for c in parted])
    points, _ = client.scroll(VECTOR_DB_COLLECTION, scroll_filter=flt, limit=1000, with_payload=True)
    groups: dict[tuple, list[RetrievedChunk]] = {}
    for p in points:
        s = RetrievedChunk.from_point(p)
        groups.setdefault((s.file_path, s.qualified_name), []).append(s)
    for chunk in parted:
        near = sorted(
            (s for s in groups.get((chunk.file_path, chunk.qualified_name), []) if s.part_index is not None and abs(s.part_index - chunk.part_index) <= window),
            key=lambda s: s.part_index,
        )
        if len(near) > 1:
            chunk.content = '\n'.join(s.content for s in near)
            start_lines = [s.start_line for s in near if s.start_line is not None]
            if start_lines:
                chunk.start_line = min(start_lines)
            end_lines = [s.end_line for s in near if s.end_line is not None]
            if end_lines:
                chunk.end_line = max(end_lines)


def _locality(d: RetrievedChunk, c: RetrievedChunk) -> tuple[bool, bool, bool]:
    cls = lambda x: (x.qualified_name or '').rpartition('.')[0]
    folder = lambda x: (x.file_path or '').rpartition('/')[0]
    return d.file_path == c.file_path, cls(d) == cls(c), folder(d) == folder(c)


def _expand_calls(client: QdrantClient, chunks: list[RetrievedChunk], max_calls: int = 4, max_related: int = 2) -> None:
    wanted = [[call.split('(')[0].rpartition('.')[2].strip() for call in c.calls[:max_calls]] for c in chunks]
    names = sorted({n for ns in wanted for n in ns if n})
    if not names:
        return
    flt = Filter(must=[
        FieldCondition(key='name', match=MatchAny(any=names)),
        FieldCondition(key='chunk_type', match=MatchAny(any=CALLABLE_TYPES)),
    ])
    points, _ = client.scroll(VECTOR_DB_COLLECTION, scroll_filter=flt, limit=200, with_payload=True)
    seen = {c.id for c in chunks}
    by_name: dict[str, list[RetrievedChunk]] = {}
    for p in points:
        d = RetrievedChunk.from_point(p)
        if d.id not in seen:
            by_name.setdefault(d.name, []).append(d)
    for c, ns in zip(chunks, wanted):
        related = {}
        for n in ns:
            if options := by_name.get(n):
                best = max(options, key=lambda d: _locality(d, c))
                related[best.id] = best
        c.related_calls = list(related.values())[:max_related]


def retrieve(
    client: QdrantClient,
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,  # mặc định 20; 40 khi có reranker
    mode: str = 'auto',  # 'auto' | 'hybrid' | 'dense' | 'sparse'
    chunk_type: str | list[str] | None = None,
    file_path: str | None = None,
    exclude_syntax_errors: bool = False,
    max_per_file: int | None = None,
    min_score: float | None = None,  # so với rerank_score nếu có reranker, ngược lại so với score RRF
    reranker: RerankerFn | None = None,
    expand_parent: bool = False,
    merge_parts: bool = False,
    expand_calls: bool = False,
) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError('query rỗng')
    top_k = RETRIEVAL_TOP_K if top_k is None else top_k
    if top_k <= 0:
        raise ValueError('top_k phải > 0')
    candidates = max(candidates or (RERANK_CANDIDATES if reranker else 20), top_k)

    flt = build_filter(chunk_type, file_path, exclude_syntax_errors)
    points = search(client, query, limit=candidates, prefetch_limit=2 * candidates, mode=mode, flt=flt)
    exact = exact_lookup(client, query, top_k, flt) if mode == 'auto' else []

    chunks = [RetrievedChunk.from_point(p) for p in [*exact, *points]]
    for c in chunks[:len(exact)]:
        c.score = float('inf')
    chunks = _dedup(chunks)

    if reranker:
        for c, s in zip(chunks, reranker(query, chunks)):
            c.rerank_score = s
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    if min_score is not None:
        chunks = [c for c in chunks if (c.rerank_score if reranker else c.score) >= min_score]
    chunks = _dedup(chunks, max_per_file)[:top_k]

    if chunks and expand_parent:
        _expand_parents(client, chunks)
    if chunks and merge_parts:
        _merge_parts(client, chunks)
    if chunks and expand_calls:
        _expand_calls(client, chunks)
    return chunks


def _header(chunk: RetrievedChunk) -> str:
    tag = ' [async]' if chunk.is_async else ''
    return f'[{chunk.file_path}:{chunk.start_line}-{chunk.end_line}] {chunk.chunk_type}{tag} {chunk.qualified_name or ""}'.strip()


def format_context(chunks: list[RetrievedChunk], include_parent_class: bool = False, include_calls: bool = True) -> str:
    parts = []
    for chunk in chunks:
        block = f'{_header(chunk)}\n{chunk.content}'
        if include_parent_class and chunk.parent:
            block += f'\n\n# --- Parent Class Context ({chunk.parent.qualified_name}) ---\n{chunk.parent.content}'
        if include_calls:
            for r in chunk.related_calls:
                block += f'\n\n# --- Called Definition: {r.qualified_name} ---\n{r.content}'
        parts.append(block)
    return '\n\n---\n\n'.join(parts)


if __name__ == '__main__':  # python -m retrieval.retriever "hàm xử lý login" --mode auto --top-k 5 [--rerank]
    from vectorstore.qdrant import close_qdrant_client, ensure_collection, get_qdrant_client
    from vectorstore.search import ensure_payload_indexes

    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='+')
    ap.add_argument('--mode', default='auto', choices=['auto', 'hybrid', 'dense', 'sparse'])
    ap.add_argument('--top-k', type=int)
    ap.add_argument('--rerank', action='store_true')
    args = ap.parse_args()

    db = get_qdrant_client()
    try:
        ensure_collection(db)
        ensure_payload_indexes(db)
        reranker = cross_encoder_reranker() if args.rerank else None
        for c in retrieve(db, ' '.join(args.query), top_k=args.top_k, mode=args.mode, reranker=reranker):
            print(f'{c.rerank_score if reranker else c.score:.4f}  {_header(c)}')
    finally:
        close_qdrant_client()