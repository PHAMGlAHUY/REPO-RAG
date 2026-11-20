import json
from pathlib import Path

from ingestion.chunking.chunker import chunk_file

SAMPLE = '''import os


class Repo:
    """Kho dữ liệu."""
    x = 1

    def a(self):
        return os.getcwd()


def helper(a,
           b):
    return a + b
'''


def print_chunks(chunks: list[dict]) -> None:
    for i, ch in enumerate(chunks, 1):
        m = ch['metadata']
        print(f"\n{'=' * 70}\n#{i} {m['chunk_type']} | {m['qualified_name']} | "
              f"dòng {m['part_start_line']}-{m['part_end_line']} | "
              f"part {m['part_index']}/{m['total_parts']} | lỗi cú pháp: {m['has_syntax_error']}")
        print('-' * 70)
        print(ch['content'])


def test_chunk_sample(tmp_path):
    f = tmp_path / 'sample.py'
    f.write_text(SAMPLE, encoding='utf-8')

    chunks = chunk_file(f, tmp_path, max_chunk_tokens=400, min_chunk_tokens=60)
    print_chunks(chunks)

    assert chunks
    assert len({c['id'] for c in chunks}) == len(chunks)


def test_chunk_real_file(tmp_path):
    """Chunk một file thật của dự án và ghi ra JSON để xem lại."""
    target = Path('ingestion/chunking/chunker.py')
    chunks = chunk_file(target, Path('.'), max_chunk_tokens=400)
    print_chunks(chunks)

    out = tmp_path / 'chunks.json'
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nĐã ghi JSON: {out}")

    assert chunks