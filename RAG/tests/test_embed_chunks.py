"""Chunk repo thật, embed rồi kiểm tra kết quả. Chạy: pytest -s tests/test_embed_chunks.py
Biến môi trường tùy chọn: EMBED_TEST_REPO (mặc định '.'), EMBED_TEST_MAX_CHUNKS (mặc định 200)."""
import os
from pathlib import Path

import numpy as np
import pytest

from embedding.embedder import embed_chunks, embed_query, get_embedding_dim, get_model
from ingestion.pipeline import iter_repository_chunks

REPO = Path(os.environ.get('EMBED_TEST_REPO', '')).resolve()
MAX_CHUNKS = int(os.environ.get('EMBED_TEST_MAX_CHUNKS', 200))


@pytest.fixture(scope='module')
def chunks() -> list[dict]:
    out: list[dict] = []
    for batch in iter_repository_chunks(REPO, workers=1):
        out.extend(batch)
        if len(out) >= MAX_CHUNKS:
            break
    out = out[:MAX_CHUNKS]
    assert out, f"Không có chunk nào từ {REPO}"
    return out


@pytest.fixture(scope='module')
def vectors(chunks) -> np.ndarray:
    return embed_chunks(chunks)


def test_shape_and_dtype(chunks, vectors):
    assert vectors.shape == (len(chunks), get_embedding_dim())
    assert vectors.dtype == np.float32


def test_finite_and_normalized(vectors):
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_no_truncation(chunks):
    m = get_model()
    lengths = np.array([len(x) for x in m.tokenizer([c['content'] for c in chunks], truncation=False)['input_ids']])
    over = int((lengths > m.max_seq_length).sum())
    print(f"\nToken/chunk: max={lengths.max()}, p95={int(np.percentile(lengths, 95))}, "
          f"max_seq_length={m.max_seq_length}, bị cắt: {over}/{len(chunks)}")
    assert over / len(chunks) <= 0.05


def test_self_retrieval(chunks, vectors):
    """Truy vấn bằng tên + docstring của hàm/method thì chunk đó phải nằm trong top 5."""
    targets = [i for i, c in enumerate(chunks) if c['metadata']['chunk_type'] in ('function', 'method')]
    assert targets, "Không có function/method nào để thử"

    hits = 0
    for i in targets:
        meta = chunks[i]['metadata']
        query = f"{meta['qualified_name'].replace('.', ' ').replace('_', ' ')} {meta['docstring']}".strip()
        top5 = np.argsort(-(vectors @ embed_query(query)))[:5]
        hits += i in top5

    rate = hits / len(targets)
    print(f"\nSelf-retrieval top-5: {hits}/{len(targets)} = {rate:.2%}")
    assert rate >= 0.5


def test_print_nearest_neighbors(chunks, vectors):
    """In 3 chunk đầu cùng các chunk gần nhất để xem bằng mắt."""
    for i in range(min(3, len(chunks))):
        scores = vectors @ vectors[i]
        scores[i] = -1
        print(f"\n{'=' * 70}\n[{chunks[i]['metadata']['chunk_type']}] {chunks[i]['metadata']['qualified_name']} "
              f"({chunks[i]['metadata']['file_path']}:{chunks[i]['metadata']['start_line']})")
        for j in np.argsort(-scores)[:3]:
            m = chunks[j]['metadata']
            print(f"   {scores[j]:.3f}  [{m['chunk_type']}] {m['qualified_name']} ({m['file_path']}:{m['start_line']})")