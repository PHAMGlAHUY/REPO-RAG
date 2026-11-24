from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config.load_config import load_settings, _resolve
from vectorstore.qdrant import VECTOR_DB_COLLECTION
from vectorstore.search import build_filter, eq, exact_lookup, search

_settings = load_settings()
RETRIEVAL_TOP_K = int((_settings.get('retrieval') or {}).get('top_k', 5))
_RERANK_CFG = _settings.get('reranking') or {}
RERANK_CANDIDATES = int(_RERANK_CFG.get('candidates', 30))
RERANK_MIN_SCORE = None if _RERANK_CFG.get('min_score') is None else float(_RERANK_CFG['min_score'])
CALLABLE_TYPES = ['function', 'method']
_SCROLL_CAP = 1000

# (query, chunks) -> điểm cho từng chunk, cùng thứ tự
RerankerFn = Callable[[str, list['RetrievedChunk']], list[float]]

# tên payload do chunker/index ghi; phần còn lại nằm trong metadata
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
    exact: bool = False  # khớp chính xác tên (exact_lookup), không có score
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
    parent: 'RetrievedChunk | None' = None
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


def onnx_reranker(model: str | None = None, batch_size: int = 16) -> RerankerFn:
    """Cross-encoder chạy bằng onnxruntime (CPU). `model` (tuỳ chọn) ghi đè reranking.onnx_file."""
    import os
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    cfg = _RERANK_CFG
    if not cfg.get('onnx_dir'):
        raise ValueError("reranking.onnx_dir chưa được cấu hình")
    base = Path(_resolve(cfg['onnx_dir']))

    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(str(base / (model or cfg['onnx_file'])), so, providers=['CPUExecutionProvider'])
    input_names = {i.name for i in sess.get_inputs()}

    tok = Tokenizer.from_file(str(base / 'tokenizer.json'))
    tok.enable_truncation(max_length=int(cfg.get('max_length', 384)))
    pad_token = next((t for t in ('[PAD]', '<pad>') if tok.token_to_id(t) is not None), '[PAD]')
    tok.enable_padding(pad_id=tok.token_to_id(pad_token) or 0, pad_token=pad_token)

    def rerank(query: str, chunks: list[RetrievedChunk]) -> list[float]:
        texts = [f'{c.qualified_name or ""}\n{c.content}' for c in chunks]
        scores: list[float] = []
        for s in range(0, len(texts), batch_size):
            enc = tok.encode_batch([(query, t) for t in texts[s:s + batch_size]])
            feed = {
                'input_ids': np.array([e.ids for e in enc], dtype=np.int64),
                'attention_mask': np.array([e.attention_mask for e in enc], dtype=np.int64),
                'token_type_ids': np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            out = sess.run(None, {k: v for k, v in feed.items() if k in input_names})[0]
            scores.extend(np.asarray(out, dtype=np.float32).reshape(len(enc), -1)[:, -1].tolist())
        return scores

    return rerank


cross_encoder_reranker = onnx_reranker  # tên cũ


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
    """Chunk nhiều part -> lấy đủ các part cùng hàm/class, ghép thành một chunk (đặt tại vị trí part tốt nhất)."""
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
        parts: dict[int, RetrievedChunk] = {}
        for p in _scroll_all(client, flt, cap=200):
            d = RetrievedChunk.from_point(p)
            if d.total_parts == c.total_parts and d.part_index is not None:
                parts.setdefault(d.part_index, d)
        parts.setdefault(c.part_index or 1, c)
        ordered = [parts[i] for i in sorted(parts)]
        c.content = '\n'.join([ordered[0].content, *(_strip_header(d.content) for d in ordered[1:])])
        c.metadata['merged_parts'] = len(ordered)
        starts = [d.start_line for d in ordered if d.start_line is not None]
        ends = [d.end_line for d in ordered if d.end_line is not None]
        if starts and ends:
            c.start_line, c.end_line = min(starts), max(ends)
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


def _locality(d: RetrievedChunk, c: RetrievedChunk) -> tuple[bool, bool, bool]:
    cls = lambda x: (x.qualified_name or '').rpartition('.')[0]
    folder = lambda x: (x.file_path or '').rpartition('/')[0]
    return d.file_path == c.file_path, cls(d) == cls(c), folder(d) == folder(c)


def _expand_calls(client: QdrantClient, chunks: list[RetrievedChunk], max_related: int = 2) -> None:
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
                best = max(options, key=lambda d: _locality(d, c))
                related[best.id] = best
        c.related_calls = list(related.values())[:max_related]


def retrieve(
    client: QdrantClient,
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,  # mặc định 20; reranking.candidates khi có reranker
    mode: str = 'auto',  # 'auto' | 'hybrid' | 'dense' | 'sparse'
    chunk_type: str | list[str] | None = None,
    file_path: str | None = None,
    exclude_syntax_errors: bool = False,
    max_per_file: int | None = None,
    min_score: float | None = None,  # chỉ áp dụng cho rerank_score (cần reranker); chunk khớp chính xác luôn được giữ
    reranker: RerankerFn | None = None,
    expand_parent: bool = False,
    merge_parts: bool = False,
    expand_calls: bool = False,
) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError('query rỗng')
    top_k = RETRIEVAL_TOP_K if top_k is None else top_k
    min_score = RERANK_MIN_SCORE if min_score is None else min_score
    if top_k <= 0:
        raise ValueError('top_k phải > 0')
    candidates = max(candidates or (RERANK_CANDIDATES if reranker else 20), top_k)

    flt = build_filter(chunk_type, file_path, exclude_syntax_errors)
    points = search(client, query, limit=candidates, prefetch_limit=2 * candidates, mode=mode, flt=flt)
    exact = exact_lookup(client, query, top_k, flt) if mode == 'auto' else []

    exact_chunks = [RetrievedChunk.from_point(p) for p in exact]
    for c in exact_chunks:
        c.exact = True
    chunks = _dedup([*exact_chunks, *(RetrievedChunk.from_point(p) for p in points)])

    if reranker and chunks:
        for c, s in zip(chunks, reranker(query, chunks)):
            c.rerank_score = s
        pinned = [c for c in chunks if c.exact]
        rest = sorted((c for c in chunks if not c.exact), key=lambda c: c.rerank_score, reverse=True)
        chunks = pinned + rest
        if min_score is not None:
            chunks = [c for c in chunks if c.exact or c.rerank_score >= min_score]
    chunks = _dedup(chunks, max_per_file)[:top_k]

    if chunks and merge_parts:
        chunks = _merge_parts(client, chunks)
    if chunks and expand_parent:
        _expand_parents(client, chunks)
    if chunks and expand_calls:
        _expand_calls(client, chunks)
    return chunks