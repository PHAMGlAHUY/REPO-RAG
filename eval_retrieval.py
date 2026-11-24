"""
Đánh giá chất lượng retrieval + reranker (Hit@k, MRR, Recall@k, nDCG@k).

cases.json: [{"query": "...", "relevant": ["vectorstore/search.py::search", ...]}, ...]
`relevant`: danh sách "file_path::qualified_name"; khớp theo ranh giới nên "x.py::search" KHÔNG khớp "x.py::search_all".
Chunk gộp (a+b) khớp với cả a lẫn b. Kết quả ghi vào logs/application.log (logger "scoring").

Ví dụ:
python eval_retrieval.py --data cases.json -k 5 -v \
  --reranker minilm=RAG/data/models/minilm-onnx:onnx/model_quint8_avx2.onnx \
  --reranker bge=hf:BAAI/bge-reranker-v2-m3:model.onnx|max_length=512,text_mode=stripped \
  --rrf both

Spec reranker:  name=<dir>:<file>  (cần <dir>/tokenizer.json)  |  name=hf:<repo>[:<file>]
Tùy chọn thêm sau dấu |:  max_length=384,text_mode=full|stripped,col=-1,sigmoid=1,tokenizer=<path>
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

_BASE = Path(__file__).resolve().parent
for _p in (_BASE / 'RAG', _BASE):  # chạy được cả khi file nằm ngoài lẫn trong thư mục RAG
    if (_p / 'vectorstore').is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config.load_config import setup_logging
from vectorstore.qdrant import close_qdrant_client, get_qdrant_client
from vectorstore.search import exact_lookup, search

log = logging.getLogger('scoring')

_HEADER_PREFIXES = ('# File:', '# Class:', '# Node:', '# Imports:', '# Decorators:', '# Signature:', '# Docstring:')


@dataclass
class Cand:
    id: str
    content: str
    file_path: str = ''
    qualified_names: list[str] = field(default_factory=list)
    exact: bool = False
    rerank_score: float | None = None
    final: float = 0.0

    @property
    def keys(self) -> list[str]:
        return [f'{self.file_path}::{n}' for n in (self.qualified_names or [''])]


def to_cand(p, exact: bool = False) -> Cand:
    pl = dict(p.payload or {})
    pl = {**(pl.pop('extra', None) or {}), **pl}
    names = pl.get('qualified_names') or [pl.get('qualified_name') or pl.get('name') or '']
    return Cand(id=str(p.id), content=pl.get('content', ''), file_path=pl.get('file_path') or '',
                qualified_names=list(names), exact=exact)


def strip_header(content: str) -> str:
    lines = content.split('\n')
    i = 0
    while i < len(lines) and lines[i].startswith(_HEADER_PREFIXES):
        i += 1
    return '\n'.join(lines[i:])


def build_text(c: Cand, mode: str) -> str:
    body = c.content if mode == 'full' else strip_header(c.content)
    return f'{c.qualified_names[0] if c.qualified_names else ""}\n{body}'


class OnnxReranker:
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 384, text_mode: str = 'full',
                 col: int = -1, sigmoid: bool = False, batch_size: int = 16):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        if text_mode not in ('full', 'stripped'):
            raise ValueError(f"text_mode phải là full hoặc stripped, nhận được {text_mode!r}")
        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_length)
        pad = next((t for t in ('[PAD]', '<pad>') if self.tok.token_to_id(t) is not None), '[PAD]')
        self.tok.enable_padding(pad_id=self.tok.token_to_id(pad) or 0, pad_token=pad)
        self.text_mode, self.col, self.sigmoid, self.batch_size = text_mode, col, sigmoid, batch_size

    def __call__(self, query: str, cands: list[Cand]) -> list[float]:
        np = self.np
        texts = [build_text(c, self.text_mode) for c in cands]
        scores: list[float] = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch([(query, t) for t in texts[s:s + self.batch_size]])
            feed = {
                'input_ids': np.array([e.ids for e in enc], dtype=np.int64),
                'attention_mask': np.array([e.attention_mask for e in enc], dtype=np.int64),
                'token_type_ids': np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            out = self.sess.run(None, {k: v for k, v in feed.items() if k in self.input_names})[0]
            arr = np.asarray(out, dtype=np.float32).reshape(len(enc), -1)[:, self.col]
            if self.sigmoid:
                arr = 1.0 / (1.0 + np.exp(-arr))
            scores.extend(arr.tolist())
        return scores


_OPT_CAST = {'max_length': int, 'text_mode': str, 'col': int, 'batch_size': int,
             'sigmoid': lambda v: str(v).lower() in ('1', 'true', 'yes')}


def parse_reranker_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition('=')
    rest, _, optstr = rest.partition('|')
    opts = dict(kv.split('=', 1) for kv in optstr.split(',') if kv)

    if rest.startswith('hf:'):
        from huggingface_hub import hf_hub_download
        repo, _, file = rest[3:].partition(':')
        file = file or 'onnx/model.onnx'
        model_path = hf_hub_download(repo, file)
        try:
            hf_hub_download(repo, file + '_data')  # external weights (nếu có)
        except Exception:
            pass
        tok_path = opts.pop('tokenizer', None) or hf_hub_download(repo, 'tokenizer.json')
    else:
        d, _, file = rest.rpartition(':')
        model_path = str(Path(d) / file)
        tok_path = opts.pop('tokenizer', None) or str(Path(d) / 'tokenizer.json')

    unknown = set(opts) - set(_OPT_CAST)
    if unknown:
        raise ValueError(f"Tùy chọn reranker không hợp lệ: {sorted(unknown)}")
    cfg = {k: _OPT_CAST[k](v) for k, v in opts.items()}
    cfg.update(model_path=model_path, tokenizer_path=tok_path)
    return name, cfg


def rank_candidates(query: str, cands: list[Cand], reranker, rrf: bool, rrf_k: int,
                    pin_exact: bool, max_per_file: int | None) -> list[Cand]:
    items = [replace(c) for c in cands]
    ret_rank = {c.id: i for i, c in enumerate(items)}

    if reranker is None:
        for c in items:
            c.final = -float(ret_rank[c.id])
    else:
        for c, s in zip(items, reranker(query, items)):
            c.rerank_score = s
        if rrf:
            order = sorted(items, key=lambda c: c.rerank_score, reverse=True)
            rr_rank = {c.id: i for i, c in enumerate(order)}
            for c in items:
                c.final = 1 / (rrf_k + ret_rank[c.id] + 1) + 1 / (rrf_k + rr_rank[c.id] + 1)
        else:
            for c in items:
                c.final = c.rerank_score

    items.sort(key=lambda c: (c.exact and pin_exact, c.final), reverse=True)

    out, per_file = [], Counter()
    for c in items:
        if max_per_file and per_file[c.file_path] >= max_per_file:
            continue
        per_file[c.file_path] += 1
        out.append(c)
    return out


def _norm(s: str) -> str:
    return s.replace('\\', '/').lower()


def matches(key: str, target: str) -> bool:
    """target khớp key nếu bắt đầu ở ranh giới thư mục/đầu chuỗi và không bị kéo dài bởi ký tự định danh."""
    t = _norm(target).strip()
    if not t:
        return False
    return re.search(r'(?:^|/)' + re.escape(t) + r'(?![a-z0-9_])', _norm(key)) is not None


def matched_targets(c: Cand, relevant: list[str]) -> set[str]:
    return {r for r in relevant if any(matches(k, r) for k in c.keys)}


def is_relevant(c: Cand, relevant: list[str]) -> bool:
    return bool(matched_targets(c, relevant))


def metrics(ranked: list[Cand], relevant: list[str], k: int) -> dict:
    top = ranked[:k]
    flags = [is_relevant(c, relevant) for c in top]
    rr = next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    credited: set[str] = set()
    gains = []
    for c in top:
        new = matched_targets(c, relevant) - credited
        gains.append(1.0 if new else 0.0)
        credited |= new
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {
        'hit': float(any(flags)),
        'mrr': rr,
        'recall': len(credited) / len(relevant) if relevant else 0.0,
        'ndcg': dcg / ideal if ideal else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--reranker', action='append', default=[],
                    help='name=<dir>:<file> | name=hf:<repo>[:<file>], thêm |max_length=..,text_mode=..,col=..,sigmoid=1')
    ap.add_argument('--no-baseline', action='store_true', help='bỏ dòng baseline "none" (không rerank)')
    ap.add_argument('--mode', default='auto', choices=['auto', 'hybrid', 'dense', 'sparse'])
    ap.add_argument('--candidates', type=int, default=30)
    ap.add_argument('--no-pin-exact', action='store_true', help='không ghim chunk khớp tên chính xác lên đầu')
    ap.add_argument('--max-per-file', type=int)
    ap.add_argument('-k', type=int, default=5)
    ap.add_argument('--rrf', choices=['off', 'on', 'both'], default='off', help='trộn thứ hạng retrieval + rerank bằng RRF')
    ap.add_argument('--rrf-k', type=int, default=60)
    ap.add_argument('-v', '--verbose', action='store_true', help='in chi tiết từng query ra console (file log luôn có)')
    args = ap.parse_args()

    setup_logging()
    log.setLevel(logging.DEBUG)  # chi tiết từng query luôn vào file; console vẫn chỉ INFO (handler level)
    log.info('=== EVAL START === args=%s', json.dumps(vars(args), ensure_ascii=False))

    cases = json.loads(Path(args.data).read_text(encoding='utf-8'))
    n = max(len(cases), 1)

    rerankers: dict[str, OnnxReranker | None] = {} if args.no_baseline else {'none': None}
    for spec in args.reranker:
        name, kw = parse_reranker_spec(spec)
        t0 = time.time()
        rerankers[name] = OnnxReranker(**kw)
        log.info('[load] %s: %.1fs', name, time.time() - t0)
    rrf_opts = {'off': [False], 'on': [True], 'both': [False, True]}[args.rrf]

    client = get_qdrant_client()
    try:
        cand_cache: list[list[Cand]] = []
        for case in cases:
            q = case['query']
            points = search(client, q, limit=args.candidates, prefetch_limit=2 * args.candidates, mode=args.mode)
            exact = exact_lookup(client, q, args.k) if args.mode == 'auto' else []
            seen, cl = set(), []
            for c in [*(to_cand(p, True) for p in exact), *(to_cand(p) for p in points)]:
                if c.id not in seen:
                    seen.add(c.id)
                    cl.append(c)
            cand_cache.append(cl)
    finally:
        close_qdrant_client()

    pool_hits = sum(any(is_relevant(c, case['relevant']) for c in cl) for case, cl in zip(cases, cand_cache))
    log.info('Pool recall@%d: %d/%d (%.1f%%)', args.candidates, pool_hits, len(cases), 100 * pool_hits / n)

    rows = []
    for name, rr in rerankers.items():
        for use_rrf in rrf_opts:
            if rr is None and use_rrf:
                continue  # baseline + rrf trùng baseline thường
            agg = Counter()
            t0 = time.time()
            details = []
            for case, cl in zip(cases, cand_cache):
                ranked = rank_candidates(case['query'], cl, rr, use_rrf, args.rrf_k,
                                         not args.no_pin_exact, args.max_per_file) if cl else []
                for k_, v in metrics(ranked, case['relevant'], args.k).items():
                    agg[k_] += v
                details.append((case, ranked))
            label = f'{name}{" +rrf" if use_rrf else ""}'
            rows.append((label, agg['hit'] / n, agg['mrr'] / n, agg['recall'] / n, agg['ndcg'] / n, time.time() - t0))

            detail = log.info if args.verbose else log.debug
            detail('=== %s ===', label)
            for case, ranked in details:
                ok = any(is_relevant(c, case['relevant']) for c in ranked[:args.k])
                detail('[%s] %s', 'OK' if ok else 'MISS', case['query'])
                if not ok:
                    detail('   expected: %s', case['relevant'])
                for i, c in enumerate(ranked[:args.k], 1):
                    mark = '*' if is_relevant(c, case['relevant']) else ' '
                    rs = '-' if c.rerank_score is None else f'{c.rerank_score:.2f}'
                    detail('   %s%d. final=%.4f rr=%s %s', mark, i, c.final, rs, c.keys[0])

    w = max((len(r[0]) for r in rows), default=10)
    log.info('%s  Hit@%d  MRR     Recall  nDCG    time(s)', 'config'.ljust(w), args.k)
    for label, hit, mrr, rec, ndcg, t in rows:
        log.info('%s  %.3f   %.3f   %.3f   %.3f   %.1f', label.ljust(w), hit, mrr, rec, ndcg, t)
    log_file = next((h.baseFilename for h in log.handlers if isinstance(h, logging.FileHandler)), None)
    log.info('=== EVAL END === (%d cases, k=%d) -> %s', len(cases), args.k, log_file)


if __name__ == '__main__':
    main()