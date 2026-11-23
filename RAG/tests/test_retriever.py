import re
import uuid
import zlib
from collections import Counter

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams

from retrieval import retriever as R
from vectorstore import search as S
from vectorstore.qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, VECTOR_DB_COLLECTION

DIM = 16


def uid(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _tokens(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def fake_dense(text: str) -> tuple[float, ...]:
    v = [0.0] * DIM
    for t in _tokens(text):
        v[zlib.crc32(t.encode()) % DIM] += 1
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return tuple(x / n for x in v)


def fake_sparse(text: str) -> SparseVector:
    idx = sorted({zlib.crc32(t.encode()) % 10_000 for t in _tokens(text)})
    return SparseVector(indices=idx, values=[1.0] * len(idx))


CLS = uid('cls')
DOCS = [
    ('cls', 'class AuthService: auth entry point',
     dict(file_path='auth.py', qualified_name='AuthService', chunk_type='class_container', start_line=1, end_line=30)),
    ('login', 'def login(self, user, password): authenticate user and open session',
     dict(file_path='auth.py', qualified_name='AuthService.login', name='login', chunk_type='method',
          parent_class_id=CLS, calls=['self.verify_password(password)'], start_line=3, end_line=8)),
    ('verify', 'def verify_password(self, password): compare password hash',
     dict(file_path='auth.py', qualified_name='AuthService.verify_password', name='verify_password', chunk_type='method',
          parent_class_id=CLS, start_line=10, end_line=14)),
    ('cfg1', 'def parse_config(path): read file part one',
     dict(file_path='config.py', qualified_name='parse_config', name='parse_config', chunk_type='function',
          part_index=1, part_total=2, start_line=1, end_line=20)),
    ('cfg2', 'continue parse_config: validate keys part two',
     dict(file_path='config.py', qualified_name='parse_config', name='parse_config', chunk_type='function',
          part_index=2, part_total=2, start_line=21, end_line=40)),
    ('view', 'def login_view(request): login page handler',
     dict(file_path='views.py', qualified_name='login_view', name='login_view', chunk_type='function', start_line=1, end_line=5)),
    ('bad', 'def login(: syntax error broken login',
     dict(file_path='bad.py', qualified_name='broken_login', name='broken_login', chunk_type='function',
          has_syntax_error=True, start_line=1, end_line=2)),
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(S, '_dense', fake_dense)
    monkeypatch.setattr(S, '_sparse', fake_sparse)
    c = QdrantClient(':memory:')
    c.create_collection(
        VECTOR_DB_COLLECTION,
        vectors_config={DENSE_VECTOR_NAME: VectorParams(size=DIM, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )
    c.upsert(VECTOR_DB_COLLECTION, [
        PointStruct(
            id=uid(key),
            vector={DENSE_VECTOR_NAME: list(fake_dense(text)), SPARSE_VECTOR_NAME: fake_sparse(text)},
            payload={**payload, 'content': text},
        )
        for key, text, payload in DOCS
    ])
    return c


def names(chunks):
    return [c.qualified_name for c in chunks]


def test_identifier_query_puts_exact_match_first(client):
    assert R.retrieve(client, 'parse_config')[0].qualified_name == 'parse_config'


def test_returns_dataclass(client):
    c = R.retrieve(client, 'login session', top_k=3)[0]
    assert isinstance(c, R.RetrievedChunk) and c.content and c.file_path


def test_exclude_syntax_errors(client):
    assert 'broken_login' in names(R.retrieve(client, 'login', top_k=10))
    assert 'broken_login' not in names(R.retrieve(client, 'login', top_k=10, exclude_syntax_errors=True))


def test_chunk_type_multi_value(client):
    out = R.retrieve(client, 'password', top_k=10, chunk_type=['method', 'function'])
    assert out and {c.chunk_type for c in out} <= {'method', 'function'}


def test_expand_parent(client):
    out = R.retrieve(client, 'AuthService.login', expand_parent=True)
    login = next(c for c in out if c.qualified_name == 'AuthService.login')
    assert login.parent.qualified_name == 'AuthService'


def test_merge_parts(client):
    c = R.retrieve(client, 'parse_config', merge_parts=True)[0]
    assert 'part one' in c.content and 'part two' in c.content
    assert (c.start_line, c.end_line) == (1, 40)


def test_expand_calls(client):
    out = R.retrieve(client, 'AuthService.login', expand_calls=True)
    login = next(c for c in out if c.qualified_name == 'AuthService.login')
    assert names(login.related_calls) == ['AuthService.verify_password']


def test_dedup_and_max_per_file(client):
    out = R.retrieve(client, 'password login config part', top_k=10, max_per_file=1)
    paths = [c.file_path for c in out]
    assert len(paths) == len(set(paths))
    assert max(Counter(names(R.retrieve(client, 'parse config part', top_k=10))).values()) == 1


def test_reranker_and_threshold(client):
    def reranker(query, chunks):
        return [1.0 if 'verify' in c.content else 0.1 for c in chunks]

    out = R.retrieve(client, 'check password', reranker=reranker, min_score=0.5)
    assert names(out) == ['AuthService.verify_password']
    assert out[0].rerank_score == 1.0


def test_invalid_args(client):
    with pytest.raises(ValueError):
        R.retrieve(client, '  ')
    with pytest.raises(ValueError):
        R.retrieve(client, 'x', mode='bogus')


def test_dense_query_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(S, 'embed_query', lambda q: calls.append(q) or np.zeros(DIM))
    S._dense.cache_clear()
    S._dense('same query')
    S._dense('same query')
    assert calls == ['same query']
    S._dense.cache_clear()


def test_is_identifier_query():
    assert S.is_identifier_query('parse_config') and S.is_identifier_query('AuthService.login')
    assert not S.is_identifier_query('hàm xử lý login') and not S.is_identifier_query('login')