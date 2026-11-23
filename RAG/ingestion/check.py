from __future__ import annotations

import json
from pathlib import Path

from config.load_config import PROJECT_ROOT, _resolve, load_settings
from ingestion.pipeline import detect_changes, load_index_state
from vectorstore.qdrant import VECTOR_DB_COLLECTION, get_qdrant_client

STATE_FILE = PROJECT_ROOT / "data" / "index_state.json"


def check_index(repo_root: str | Path, state_file: str | Path = STATE_FILE) -> dict:
    settings = load_settings()
    emb = settings.get("embedding", {})
    meta = _resolve(emb.get("index_meta_path", "data/index_meta.json"))
    if meta.is_file():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved.get("model") != emb.get("model"):
            raise RuntimeError(
                f"Index tạo bằng model {saved.get('model')} nhưng cấu hình là {emb.get('model')}. "
                f"Xóa collection, index_state.json và {meta} rồi chạy lại."
            )

    reasons: list[str] = []
    client = get_qdrant_client()
    if not client.collection_exists(VECTOR_DB_COLLECTION):
        reasons.append("collection chưa tồn tại")
    elif client.count(VECTOR_DB_COLLECTION, exact=False).count == 0:
        reasons.append("collection rỗng")

    state_path = Path(state_file)
    if not state_path.is_file():
        reasons.append("chưa có index_state.json")

    changed, deleted, new = detect_changes(repo_root, load_index_state(state_path))
    if changed or deleted or new:
        reasons.append(f"{len(new)} file mới, {len(changed)} file sửa, {len(deleted)} file xóa")

    return {
        "needs_index": bool(reasons),
        "reasons": reasons,
        "changed": changed,
        "deleted": deleted,
        "new": new,
    }