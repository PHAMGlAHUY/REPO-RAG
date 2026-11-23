"""So RRF của Qdrant (search mode='hybrid') với RRF tự viết trên cùng hai danh sách dense/sparse.

python -m retrieval.eval_retrieval queries.jsonl --k 10
queries: mỗi dòng là text thuần hoặc {"query": "...", "expected": ["qualified_name", ...]}
"""
import argparse
import json

from vectorstore.qdrant import close_qdrant_client, ensure_collection, get_qdrant_client
from vectorstore.search import RRF_K, search


def rrf(lists: list[list[str]], k: int = RRF_K, weights: list[float] | None = None) -> list[str]:
    scores: dict[str, float] = {}
    for w, ids in zip(weights or [1.0] * len(lists), lists):
        for rank, i in enumerate(ids, 1):
            scores[i] = scores.get(i, 0.0) + w / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def load_queries(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        rows = [line.strip() for line in f if line.strip()]
    return [json.loads(r) if r.startswith('{') else {'query': r} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('queries')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--prefetch', type=int, default=50)
    args = ap.parse_args()
    k, pf = args.k, args.prefetch

    db = get_qdrant_client()
    try:
        ensure_collection(db)
        qname: dict[str, str] = {}

        def ids(query: str, mode: str, limit: int) -> list[str]:
            points = search(db, query, limit=limit, prefetch_limit=pf, mode=mode)
            qname.update({str(p.id): (p.payload or {}).get('qualified_name') for p in points})
            return [str(p.id) for p in points]

        overlap = same_order = top1 = 0.0
        hit = {'qdrant': 0, 'manual': 0}
        n_expected = 0
        rows = load_queries(args.queries)
        for row in rows:
            q = row['query']
            native = ids(q, 'hybrid', k)
            manual = rrf([ids(q, 'dense', pf), ids(q, 'sparse', pf)])[:k]
            overlap += len(set(native) & set(manual)) / max(len(native), 1)
            same_order += native == manual
            top1 += native[:1] == manual[:1]
            if row.get('expected'):
                n_expected += 1
                want = set(row['expected'])
                hit['qdrant'] += bool(want & {qname[i] for i in native})
                hit['manual'] += bool(want & {qname[i] for i in manual})
            if native != manual:
                print(f'DIFF  {q!r}\n  qdrant: {[qname[i] for i in native]}\n  manual: {[qname[i] for i in manual]}')

        n = len(rows)
        print(f'queries={n}  overlap@{k}={overlap / n:.3f}  same_order={same_order / n:.3f}  top1_agree={top1 / n:.3f}')
        if n_expected:
            print(f'hit@{k}  qdrant={hit["qdrant"] / n_expected:.3f}  manual={hit["manual"] / n_expected:.3f}  (n={n_expected})')
    finally:
        close_qdrant_client()


if __name__ == '__main__':
    main()