"""Soi nội dung collection Qdrant mà retriever.py đang dùng.

Chạy (từ thư mục gốc project):
    python inspect_collection.py
    python inspect_collection.py --url http://localhost:6333
    python inspect_collection.py --path ./qdrant_data
    python inspect_collection.py --query "load settings" --top-k 3
    python inspect_collection.py --samples 5 --scan 2000
"""
import argparse
import json
from collections import Counter

from qdrant_client import QdrantClient

from vectorstore.qdrant import VECTOR_DB_COLLECTION


def make_client(args) -> QdrantClient:
    if args.url:
        return QdrantClient(url=args.url)
    if args.path:
        return QdrantClient(path=args.path)
    # Thử dùng hàm tạo client có sẵn trong project (nếu có)
    try:
        from vectorstore import qdrant as q
        for name in ('get_client', 'create_client', 'get_qdrant_client', 'client'):
            obj = getattr(q, name, None)
            if obj is None:
                continue
            return obj() if callable(obj) else obj
    except Exception as e:
        print(f'[!] không tạo được client từ project: {e}')
    return QdrantClient(url='http://localhost:6333')


def title(text: str) -> None:
    print(f'\n{"=" * 70}\n{text}\n{"=" * 70}')


def short(v, n=120):
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    s = s.replace('\n', '\\n')
    return s if len(s) <= n else s[:n] + '...'


def show_info(client: QdrantClient) -> None:
    title(f'1. Collection: {VECTOR_DB_COLLECTION}')
    info = client.get_collection(VECTOR_DB_COLLECTION)
    print('status          :', info.status)
    print('points_count    :', info.points_count)
    print('indexed_vectors :', getattr(info, 'indexed_vectors_count', None))
    cfg = info.config.params
    print('vectors (dense) :', short(cfg.vectors, 300))
    print('sparse_vectors  :', short(getattr(cfg, 'sparse_vectors', None), 300))
    schema = info.payload_schema or {}
    print('payload index   :')
    if not schema:
        print('  (chưa có payload index nào)')
    for k, v in schema.items():
        print(f'  - {k}: {v.data_type}, points={v.points}')


def scan(client: QdrantClient, limit: int) -> list:
    points, offset = [], None
    while len(points) < limit:
        batch, offset = client.scroll(
            VECTOR_DB_COLLECTION, limit=min(256, limit - len(points)),
            offset=offset, with_payload=True, with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def show_samples(points: list, n: int) -> None:
    title(f'2. {n} point mẫu')
    for p in points[:n]:
        payload = dict(p.payload or {})
        content = payload.pop('content', '')
        print(f'\n--- id={p.id}')
        for k, v in payload.items():
            print(f'  {k:18}: {short(v)}')
        print('  content (300 ký tự đầu):')
        for line in content[:300].split('\n'):
            print('    |', line)


def show_stats(points: list) -> None:
    title(f'3. Thống kê trên {len(points)} point đã quét')
    payloads = [p.payload or {} for p in points]
    flat = [{**(pl.get('extra') or {}), **pl} for pl in payloads]

    keys = Counter(k for pl in flat for k in pl)
    print('\nField xuất hiện (số point có field đó):')
    for k, c in keys.most_common():
        print(f'  {k:20} {c}/{len(flat)}')

    print('\nchunk_type:')
    for k, c in Counter(pl.get('chunk_type') for pl in flat).most_common():
        print(f'  {str(k):20} {c}')

    print('\nTop 10 file_path:')
    for k, c in Counter(pl.get('file_path') for pl in flat).most_common(10):
        print(f'  {c:5}  {k}')

    multi = [pl for pl in flat if (pl.get('total_parts') or 1) > 1]
    print(f'\nChunk nhiều part (total_parts > 1): {len(multi)}')
    print(f'Chunk có calls                   : {sum(1 for pl in flat if pl.get("calls"))}')
    print(f'Chunk async                      : {sum(1 for pl in flat if pl.get("is_async"))}')

    lens = sorted(len(pl.get('content', '')) for pl in flat)
    if lens:
        print(f'Độ dài content: min={lens[0]}, median={lens[len(lens) // 2]}, max={lens[-1]}')

    extra_keys = Counter(k for pl in payloads for k in (pl.get('extra') or {}))
    if extra_keys:
        print('\nCác key nằm trong payload["extra"]:')
        for k, c in extra_keys.most_common():
            print(f'  {k:20} {c}')


def show_retrieval(client: QdrantClient, query: str, top_k: int, mode: str) -> None:
    from retriever import retrieve
    title(f'4. retrieve(query={query!r}, top_k={top_k}, mode={mode})')
    chunks = retrieve(client, query, top_k=top_k, mode=mode)
    if not chunks:
        print('(không có kết quả)')
    for i, c in enumerate(chunks, 1):
        print(f'\n#{i} score={c.score} exact={c.exact} type={c.chunk_type}')
        print(f'   {c.file_path}:{c.start_line}-{c.end_line}  {c.qualified_name}')
        print(f'   parts={c.part_index}/{c.total_parts}  calls={short(c.calls, 80)}')
        print(f'   metadata keys: {list(c.metadata)}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url')
    ap.add_argument('--path')
    ap.add_argument('--samples', type=int, default=3, help='số point mẫu in ra')
    ap.add_argument('--scan', type=int, default=1000, help='số point quét để thống kê')
    ap.add_argument('--query', help='nếu có, chạy thử retrieve với câu truy vấn này')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--mode', default='auto', choices=['auto', 'hybrid', 'dense', 'sparse'])
    args = ap.parse_args()

    client = make_client(args)
    show_info(client)
    points = scan(client, args.scan)
    if not points:
        print('\nCollection rỗng.')
        return
    show_samples(points, args.samples)
    show_stats(points)
    if args.query:
        show_retrieval(client, args.query, args.top_k, args.mode)


if __name__ == '__main__':
    main()