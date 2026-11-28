import json
import logging
import os
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config.load_config import load_settings
from vectorstore.qdrant import VECTOR_DB_COLLECTION
from vectorstore.search import build_filter, eq, exact_lookup, search

_settings = load_settings()
RETRIEVAL_TOP_K = int((_settings.get('retrieval') or {}).get('top_k', 5))
_RERANK_CFG = _settings.get('reranking') or {}
RERANK_ENABLED = bool(_RERANK_CFG.get('enabled', True))
RERANK_CANDIDATES = int(_RERANK_CFG.get('candidates', 20))
RERANK_MIN_SCORE = None if _RERANK_CFG.get('min_score') is None else float(_RERANK_CFG['min_score'])
CALLABLE_TYPES = ['function', 'method']
_SCROLL_CAP = 1000

log = logging.getLogger(__name__)

# (query, chunks) -> điểm cho từng chunk, cùng thứ tự
RerankerFn = Callable[[str, list['RetrievedChunk']], list[float]]

_PAYLOAD_FIELDS = (
    'file_path', 'qualified_name', 'qualified_names', 'name', 'names', 'chunk_type', 'start_line', 'end_line',
    'is_async', 'parent_class_id', 'part_index', 'total_parts', 'calls',
)
_HEADER_PREFIXES = ('# File:', '# Class:', '# Node:', '# Imports:', '# Decorators:', '# Signature:', '# Docstring:')


@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float | None = None
    rerank_score: float | None = None
    exact: bool = False  # khớp chính xác tên (exact_lookup)
    file_path: str | None = None
    qualified_name: str | None = None
    qualified_names: list[str] = field(default_factory=list)
    name: str | None = None
    names: list[str] = field(default_factory=list)
    chunk_type: str = ''
    start_line: int | None = None
    end_line: int | None = None
    is_async: bool = False
    parent_class_id: str | None = None
    part_index: int | None = None
    total_parts: int | None = None
    calls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    related_calls: list['RetrievedChunk'] = field(default_factory=list)

    def __post_init__(self):
        self.calls = self.calls or []
        self.chunk_type = self.chunk_type or ''
        self.is_async = bool(self.is_async)
        self.name = self.name or (self.qualified_name or '').rpartition('.')[2] or None
        self.names = self.names or ([self.name] if self.name else [])
        self.qualified_names = self.qualified_names or ([self.qualified_name] if self.qualified_name else [])

    @classmethod
    def from_point(cls, p) -> 'RetrievedChunk':
        payload = dict(p.payload or {})
        payload = {**(payload.pop('extra', None) or {}), **payload}
        content = payload.pop('content', '')
        known = {k: payload.pop(k) for k in _PAYLOAD_FIELDS if k in payload}
        return cls(id=str(p.id), content=content, score=getattr(p, 'score', None), metadata=payload, **known)


class JinaReranker:
    """Gọi Jina Rerank API (POST /v1/rerank). Dùng như RerankerFn: reranker(query, chunks) -> list[float]."""

    def __init__(self, api_key: str, model: str = 'jina-reranker-v3.5', url: str = 'https://api.jina.ai/v1/rerank',
                 timeout: float = 10, max_doc_length: int | None = None, retries: int = 1):
        self.api_key, self.model, self.url = api_key, model, url
        self.timeout, self.max_doc_length, self.retries = timeout, max_doc_length, retries

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode('utf-8'), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'},
        )
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                detail = e.read().decode('utf-8', 'replace')[:300]
                last = RuntimeError(f'Jina rerank HTTP {e.code}: {detail}')
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = RuntimeError(f'Jina rerank lỗi mạng: {e}')
        raise last  # type: ignore[misc]

    def __call__(self, query: str, chunks: list['RetrievedChunk']) -> list[float]:
        if not chunks:
            return []
        body = {
            'model': self.model,
            'query': query,
            'top_n': len(chunks),
            'documents': [c.content for c in chunks],
            'return_documents': False,
        }
        if self.max_doc_length:
            body['max_doc_length'] = self.max_doc_length
        results = self._post(body).get('results') or []
        scores = [float('-inf')] * len(chunks)
        for r in results:
            i = r.get('index')
            if isinstance(i, int) and 0 <= i < len(chunks):
                scores[i] = float(r.get('relevance_score', float('-inf')))
        return scores


def default_reranker() -> JinaReranker | None:
    """Tạo JinaReranker từ settings.reranking; trả None nếu tắt hoặc thiếu API key."""
    if not RERANK_ENABLED:
        return None
    api_key = os.environ.get(_RERANK_CFG.get('api_key_env') or 'JINA_API_KEY')
    if not api_key:
        log.warning('Không có JINA_API_KEY -> bỏ qua rerank')
        return None
    kwargs = {k: _RERANK_CFG[k] for k in ('model', 'url', 'timeout', 'max_doc_length') if _RERANK_CFG.get(k) is not None}
    return JinaReranker(api_key, **kwargs)


def _dedup(chunks: list[RetrievedChunk], max_per_file: int | None = None) -> list[RetrievedChunk]:
    seen, per_file, out = set(), Counter(), []
    for c in chunks:
        if c.id in seen or (max_per_file and per_file[c.file_path] >= max_per_file):
            continue
        seen.add(c.id)
        per_file[c.file_path] += 1
        out.append(c)
    return out


def _scroll_all(client: QdrantClient, flt: Filter, cap: int = _SCROLL_CAP) -> list:
    points: list = []
    offset = None
    while len(points) < cap:
        batch, offset = client.scroll(
            VECTOR_DB_COLLECTION, scroll_filter=flt, limit=min(256, cap - len(points)),
            offset=offset, with_payload=True,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _strip_header(content: str) -> str:
    lines = content.split('\n')
    i = 0
    while i < len(lines) and lines[i].startswith(_HEADER_PREFIXES):
        i += 1
    return '\n'.join(lines[i:])


def _merge_parts(client: QdrantClient, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Chunk nhiều part -> lấy đủ các part cùng hàm/class và ghép thành một chunk."""
    out: list[RetrievedChunk] = []
    done: set[tuple] = set()
    for c in chunks:
        if (c.total_parts or 1) <= 1:
            out.append(c)
            continue
        key = (c.file_path, c.qualified_name, c.chunk_type)
        if key in done:
            continue
        done.add(key)

        flt = Filter(must=[eq('file_path', c.file_path), eq('qualified_name', c.qualified_name),
                           eq('chunk_type', c.chunk_type)])
        parts = {c.part_index or 1: c}
        for p in _scroll_all(client, flt, cap=200):
            d = RetrievedChunk.from_point(p)
            if d.total_parts == c.total_parts and d.part_index is not None:
                parts.setdefault(d.part_index, d)

        ordered = [parts[i] for i in sorted(parts)]
        c.content = '\n'.join([ordered[0].content, *(_strip_header(d.content) for d in ordered[1:])])
        c.metadata['merged_parts'] = len(ordered)
        starts = [d.start_line for d in ordered if d.start_line is not None]
        ends = [d.end_line for d in ordered if d.end_line is not None]
        if starts and ends:
            c.start_line, c.end_line = min(starts), max(ends)
        out.append(c)
    return out


def _expand_calls(client: QdrantClient, chunks: list[RetrievedChunk], max_related: int = 2) -> None:
    """Gắn định nghĩa của các hàm được gọi vào chunk.related_calls (ưu tiên cùng file)."""
    wanted = [[call.split('(')[0].rpartition('.')[2].strip() for call in c.calls] for c in chunks]
    names = sorted({n for ns in wanted for n in ns if n})
    if not names:
        return

    flt = Filter(must=[
        FieldCondition(key='names', match=MatchAny(any=names)),
        FieldCondition(key='chunk_type', match=MatchAny(any=CALLABLE_TYPES)),
    ])
    seen = {c.id for c in chunks}
    by_name: dict[str, list[RetrievedChunk]] = {}
    for p in _scroll_all(client, flt):
        d = RetrievedChunk.from_point(p)
        if d.id in seen:
            continue
        for n in d.names:
            by_name.setdefault(n, []).append(d)

    for c, ns in zip(chunks, wanted):
        related: dict[str, RetrievedChunk] = {}
        for n in ns:
            if options := by_name.get(n):
                best = max(options, key=lambda d: d.file_path == c.file_path)
                related[best.id] = best
        c.related_calls = list(related.values())[:max_related]


def retrieve(
    client: QdrantClient,
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,  # số ứng viên lấy từ search; mặc định 20, hoặc reranking.candidates khi rerank
    mode: str = 'auto',  # 'auto' | 'hybrid' | 'dense' | 'sparse'
    chunk_type: str | list[str] | None = None,
    file_path: str | None = None,
    exclude_syntax_errors: bool = False,
    max_per_file: int | None = None,
    min_score: float | None = None,  # chỉ áp dụng cho rerank_score; chunk khớp chính xác luôn được giữ
    rerank: bool = True,  # False để tắt rerank cho lần gọi này
    reranker: RerankerFn | None = None,  # mặc định: JinaReranker theo settings
    merge_parts: bool = False,
    expand_calls: bool = False,
) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError('query rỗng')
    top_k = RETRIEVAL_TOP_K if top_k is None else top_k
    if top_k <= 0:
        raise ValueError('top_k phải > 0')
    if rerank and reranker is None:
        reranker = default_reranker()
    elif not rerank:
        reranker = None
    min_score = RERANK_MIN_SCORE if min_score is None else min_score
    candidates = max(candidates or (RERANK_CANDIDATES if reranker else 20), top_k)

    # 1. Lấy ứng viên: exact (chỉ mode auto) đứng trước, rồi đến search
    flt = build_filter(chunk_type, file_path, exclude_syntax_errors)
    points = search(client, query, limit=candidates, mode=mode, flt=flt)
    exact_points = exact_lookup(client, query, top_k, flt) if mode == 'auto' else []
    exact_chunks = [RetrievedChunk.from_point(p) for p in exact_points]
    for c in exact_chunks:
        c.exact = True
    chunks = _dedup([*exact_chunks, *(RetrievedChunk.from_point(p) for p in points)])

    # 2. Rerank: chunk exact giữ đầu danh sách, phần còn lại xếp theo rerank_score.
    #    Lỗi API -> giữ nguyên thứ tự search (không làm hỏng cả truy vấn).
    if reranker and chunks:
        pinned = [c for c in chunks if c.exact]
        rest = [c for c in chunks if not c.exact]
        if rest:
            try:
                scores = reranker(query, rest)
                if len(scores) != len(rest):
                    raise ValueError('reranker trả sai số lượng điểm')
                for c, s in zip(rest, scores):
                    c.rerank_score = s
                if min_score is not None:
                    rest = [c for c in rest if c.rerank_score >= min_score]
                rest.sort(key=lambda c: c.rerank_score, reverse=True)
            except Exception as e:
                log.warning('Rerank thất bại, dùng thứ tự search: %s', e)
            chunks = pinned + rest

    # 3. Chọn kết quả cuối
    chunks = _dedup(chunks, max_per_file)[:top_k]

    # 4. Bổ sung tùy chọn
    if chunks and merge_parts:
        chunks = _merge_parts(client, chunks)
    if chunks and expand_calls:
        _expand_calls(client, chunks)
    return chunks