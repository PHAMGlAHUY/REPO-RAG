import json
import logging
import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.models import (
    Distance, HnswConfigDiff, Modifier, PayloadSchemaType, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, SparseVectorParams, VectorParams,
)

from config.load_config import load_settings
from embedding import embedder
from embedding.embedder import get_embedding_dim

settings = load_settings()
logger = logging.getLogger('vector_database')

VECTOR_DB_SETTINGS = settings['vector_database']
VECTOR_DB_URL = VECTOR_DB_SETTINGS.get('url', f"https://{VECTOR_DB_SETTINGS.get('host', 'localhost')}:{VECTOR_DB_SETTINGS.get('port', 6333)}")
VECTOR_DB_API_KEY = VECTOR_DB_SETTINGS.get('api_key', None)
VECTOR_DB_COLLECTION = VECTOR_DB_SETTINGS.get('collection_name', 'default_collection')
VECTOR_DB_DISTANCE = VECTOR_DB_SETTINGS.get('distance', 'cosine')
VECTOR_DB_TIMEOUT = VECTOR_DB_SETTINGS.get('timeout', 30)
VECTOR_DB_BATCH_SIZE = VECTOR_DB_SETTINGS.get('upsert_batch_size', 64)
SPARSE_MODEL = VECTOR_DB_SETTINGS.get('sparse_model', 'Qdrant/bm25')
SPARSE_DISABLE_STEMMER = VECTOR_DB_SETTINGS.get('sparse_disable_stemmer', False)
CHUNKER_VERSION = VECTOR_DB_SETTINGS.get('chunker_version', 1)
INDEX_FINGERPRINT_PATH = VECTOR_DB_SETTINGS.get('fingerprint_path', 'data/index_fingerprint.json')
HNSW_M = VECTOR_DB_SETTINGS.get('hnsw_m', 16)
HNSW_EF_CONSTRUCT = VECTOR_DB_SETTINGS.get('hnsw_ef_construct', 100)

DENSE_VECTOR_NAME = 'dense'
SPARSE_VECTOR_NAME = 'sparse'

KEYWORD_FIELDS = ('file_path', 'chunk_type', 'qualified_name', 'parent_class_id', 'content_hash')
BOOL_FIELDS = ('has_syntax_error',)

client: QdrantClient | None = None
_client_lock = threading.Lock()


def _distance() -> Distance:
    try:
        return getattr(Distance, str(VECTOR_DB_DISTANCE).upper())
    except KeyError:
        raise ValueError(f"distance '{VECTOR_DB_DISTANCE}' không hợp lệ, chọn một trong {[d.name.lower() for d in Distance]}") from None


def get_qdrant_client() -> QdrantClient:
    global client
    with _client_lock:
        if client is None:
            new_client = None
            try:
                new_client = QdrantClient(url=VECTOR_DB_URL, api_key=VECTOR_DB_API_KEY, timeout=VECTOR_DB_TIMEOUT)
                new_client.get_collections()
            except Exception as e:
                if new_client is not None:
                    new_client.close()
                raise ConnectionError(f"Failed to connect to Qdrant server: {e}") from e
            client = new_client
            print(f"Connected to Qdrant at: \033[34m\033[32m {VECTOR_DB_URL} \033[0m")
            logger.info(f"Connected to Qdrant at {VECTOR_DB_URL}")

        assert client is not None
        return client


def close_qdrant_client() -> None:
    global client
    with _client_lock:
        if client is not None:
            client.close()
            client = None


def check_fingerprint() -> None:
    """Lần đầu ghi fingerprint; các lần sau báo lỗi nếu bất kỳ thành phần nào của index đã đổi."""
    current = {
        'model': embedder.EMBEDDING_MODEL,
        'dim': get_embedding_dim(),
        'sparse_model': SPARSE_MODEL,
        'disable_stemmer': SPARSE_DISABLE_STEMMER,
        'distance': str(VECTOR_DB_DISTANCE).lower(),
        'chunker_version': CHUNKER_VERSION,
    }
    p = Path(INDEX_FINGERPRINT_PATH)
    if not p.is_file():
        print("\033[33mNo fingerprint found, create new.\033[0m")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print("Created fingerprint : \033[32m Done \033[0m")
        return
    saved = json.loads(p.read_text(encoding='utf-8'))
    print(f"Fingerprint found : \033[32m {saved} \033[0m")
    if saved != current:
        print("\033[31mFingerprint đã tạo không khớp với hiện tại, raise error\033[0m")

        raise RuntimeError(
            f"Index được tạo bằng {saved}, nhưng cấu hình hiện tại là {current}. "
            f"Hãy xóa collection, file index_state và {p} rồi index lại từ đầu."
        )


def ensure_collection(db_client: QdrantClient) -> None:
    size = get_embedding_dim()
    distance = _distance()

    if not db_client.collection_exists(VECTOR_DB_COLLECTION):
        try:
            print("\033[33mNo collection found, create new.\033[0m")
            db_client.create_collection(
                collection_name=VECTOR_DB_COLLECTION,
                vectors_config={DENSE_VECTOR_NAME: VectorParams(size=size, distance=distance)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
                ),
                hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
                on_disk_payload=True,
            )
            print(f"Created new collection: \033[32m {VECTOR_DB_COLLECTION} \033[0m")
            logger.info(f"Collection {VECTOR_DB_COLLECTION} created")
        except (UnexpectedResponse, ResponseHandlingException) as e:
            logger.error(f"Lỗi khi tạo collection: {e}")
            if not db_client.collection_exists(VECTOR_DB_COLLECTION):
                raise

    for field in KEYWORD_FIELDS:
        print(f"KEYWORD_FIELDS for \033[32m {field} \033[0m")
        db_client.create_payload_index(VECTOR_DB_COLLECTION, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
    for field in BOOL_FIELDS:
        print(f"BOOL_FIELDS for \033[32m {field} \033[0m")
        db_client.create_payload_index(VECTOR_DB_COLLECTION, field_name=field, field_schema=PayloadSchemaType.BOOL)

    cfg = db_client.get_collection(VECTOR_DB_COLLECTION).config.params
    dense = cfg.vectors.get(DENSE_VECTOR_NAME) if isinstance(cfg.vectors, dict) else None
    sparse = (cfg.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
    if dense is None or sparse is None:
        raise RuntimeError(f"Collection {VECTOR_DB_COLLECTION} is missing vector 'dense'/'sparse'. Please delete old collection.")
    if dense.size != size:
        raise RuntimeError(f"Collection {VECTOR_DB_COLLECTION} has dense size {dense.size} but model dim={size}.")
    if dense.distance != distance:
        raise RuntimeError(f"Collection {VECTOR_DB_COLLECTION} has distance {dense.distance} but config distance={distance}.")
    if sparse.modifier != Modifier.IDF:
        raise RuntimeError(f"Collection {VECTOR_DB_COLLECTION} sparse vector must use modifier IDF.")

    check_fingerprint()