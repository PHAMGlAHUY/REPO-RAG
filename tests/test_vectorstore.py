import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from vectorstore import index, qdrant, search, upsert

DIM = 8


def _dense(texts, **_):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.split():
            out[i, hash(w) % DIM] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _sparse(texts):
    return [SparseVector(indices=sorted({hash(w) % 1000 for w in t.split()}), values=[1.0] * len({hash(w) % 1000 for w in t.split()})) for t in texts]


def _chunk(file_path, name, content):
    return {
        'id': str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path}:{name}')),
        'content': content,
        'metadata': {'file_path': file_path, 'qualified_name': name, 'chunk_type': 'function',
                     'content_hash': name, 'imports': ['os'], 'calls': []},
    }


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant, 'get_embedding_dim', lambda: DIM)
    monkeypatch.setattr(qdrant, 'INDEX_FINGERPRINT_PATH', str(tmp_path / 'fp.json'))
    monkeypatch.setattr(index, 'embed_texts', _dense)
    monkeypatch.setattr(index, '_embed_sparse', _sparse)
    monkeypatch.setattr(search, 'embed_query', lambda q: _dense([q])[0])
    monkeypatch.setattr(search, '_sparse_query', lambda q: _sparse([q])[0])
    client = QdrantClient(':memory:')
    qdrant.ensure_collection(client)
    return client


def _count(client):
    return client.count(qdrant.VECTOR_DB_COLLECTION).count


def test_index_and_delete(db):
    chunks = [_chunk('a.py', 'f', 'def f read file'), _chunk('a.py', 'g', 'def g parse json'), _chunk('b.py', 'h', 'def h send mail')]
    assert upsert.index_batch(db, chunks) is True
    assert _count(db) == 3

    upsert.index_batch(db, [_chunk('a.py', 'f2', 'def f2 read file')], changed_files=['a.py'])
    assert _count(db) == 2

    upsert.index_batch(db, [], deleted_files=['a.py', 'b.py'])
    assert _count(db) == 0


def test_payload_extra(db):
    upsert.index_batch(db, [_chunk('a.py', 'f', 'def f read file')])
    p = db.scroll(qdrant.VECTOR_DB_COLLECTION, limit=1)[0][0].payload
    assert p['extra']['imports'] == ['os'] and p['content'] == 'def f read file'


def test_hybrid_search(db):
    upsert.index_batch(db, [_chunk('a.py', 'f', 'read file'), _chunk('b.py', 'h', 'send mail')])
    assert search.search(db, 'read file')[0].payload['qualified_name'] == 'f'
    assert [p.payload['file_path'] for p in search.search(db, 'send mail', file_path='b.py')] == ['b.py']


def test_fingerprint_change(db, monkeypatch):
    monkeypatch.setattr(qdrant, 'CHUNKER_VERSION', 99)
    with pytest.raises(RuntimeError):
        qdrant.ensure_collection(db)