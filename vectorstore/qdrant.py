import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from core.load_settings import load_settings

logger = logging.getLogger(__name__)
_cfg = load_settings()['vector_database']
client = QdrantClient(url=_cfg['url'], api_key=_cfg['api_key'], timeout=_cfg['timeout'])
COLLECTION = _cfg['collection_name']

def ensure_collection():
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=_cfg['vector_size'], distance=Distance.COSINE),
        )
        logger.info("Created collection %s", COLLECTION)

def upsert_chunks(chunks: list[dict], vectors, batch=128):
    points = [
        PointStruct(
            # id cố định theo file + vị trí -> chạy lại không bị trùng
            id=str(uuid.uuid5(uuid.NAMESPACE_URL,
                              f"{c['metadata']['file']}:{c['metadata'].get('name')}:{c['metadata'].get('start_line', 0)}")),
            vector=v.tolist(),
            payload={"text": c["text"], **c["metadata"]},
        )
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), batch):
        client.upsert(COLLECTION, points[i:i + batch])
    logger.info("Chunks upserted: %s", len(points))