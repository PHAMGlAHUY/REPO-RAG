import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.load_config import PROJECT_ROOT
from vectorstore.qdrant import VECTOR_DB_COLLECTION, close_qdrant_client, get_qdrant_client

client = get_qdrant_client()
if client.collection_exists(VECTOR_DB_COLLECTION):
    client.delete_collection(VECTOR_DB_COLLECTION)
    print(f"Đã xóa collection {VECTOR_DB_COLLECTION}")
close_qdrant_client()

for name in ("index_state.json", "index_fingerprint.json", "index_meta.json"):
    p = PROJECT_ROOT / "data" / name
    if p.exists():
        p.unlink()
        print("Đã xóa", p)