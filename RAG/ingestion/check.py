from __future__ import annotations

import json
from pathlib import Path

from embedding import embedder
from ingestion.pipeline import STATE_FILE, detect_changes, load_index_state
from vectorstore.qdrant import (
    CHUNKER_VERSION, INDEX_FINGERPRINT_PATH, SPARSE_DISABLE_STEMMER, SPARSE_MODEL,
    VECTOR_DB_COLLECTION, VECTOR_DB_DISTANCE, get_qdrant_client,
)


def _collection_problem(client) -> str | None:
    if not client.collection_exists(VECTOR_DB_COLLECTION):
        return "No collections found"
    points, _ = client.scroll(VECTOR_DB_COLLECTION, limit=1, with_payload=False, with_vectors=False)
    if not points:
        return "Empty collections found"
    return None


def _fingerprint_changes() -> list[str]:
    """So sánh cấu hình hiện tại với fingerprint đã lưu (không load model, không ghi file)."""
    p = Path(INDEX_FINGERPRINT_PATH)
    if not p.is_file():
        return []
    try:
        saved = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return ['fingerprint unreadable']
    current = {
        'model': embedder.EMBEDDING_MODEL,
        'backend': embedder.BACKEND,
        'onnx_file': embedder.ONNX_FILE,
        'sparse_model': SPARSE_MODEL,
        'disable_stemmer': SPARSE_DISABLE_STEMMER,
        'distance': str(VECTOR_DB_DISTANCE).lower(),
        'chunker_version': CHUNKER_VERSION,
    }
    return [k for k, v in current.items() if k in saved and saved[k] != v]


def check_index(repo_root: str | Path, state_file: str | Path = STATE_FILE) -> dict:
    reasons: list[str] = []

    problem = _collection_problem(get_qdrant_client())
    reset_state = problem is not None  # collection mất/rỗng: bỏ qua state cũ, index lại toàn bộ
    if problem:
        reasons.append(problem)

    previous_state = {} if reset_state else load_index_state(state_file)
    changed, deleted, new = detect_changes(repo_root, previous_state)
    if changed or deleted or new:
        reasons.append(f"{len(new)} new, {len(changed)} edited, {len(deleted)} deleted")

    diff = _fingerprint_changes()
    if diff:
        reasons.append(f"Index config changed: {', '.join(diff)}")

    return {
        'needs_index': bool(reasons),
        'reasons': reasons,
        'reset_state': reset_state,
        'changed': changed,
        'deleted': deleted,
        'new': new,
    }