import logging
import time
from typing import Callable, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny

from vectorstore.index import create_qdrant_points
from vectorstore.qdrant import VECTOR_DB_BATCH_SIZE, VECTOR_DB_COLLECTION

logger = logging.getLogger('vector_database')

MAX_RETRIES = 3


def _retry(fn: Callable[[], None]) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            fn()
            return
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Qdrant lỗi ({e}), thử lại sau {wait}s")
            time.sleep(wait)



def delete_files(client: QdrantClient, file_paths: Iterable[str]) -> None:
    """Xoá toàn bộ chunk của các file trong một lần gọi."""
    file_paths = sorted(set(file_paths))
    if not file_paths:
        return

    flt = Filter(must=[FieldCondition(key='file_path', match=MatchAny(any=file_paths))])

    def _do_delete() -> None:
        client.delete(
            collection_name=VECTOR_DB_COLLECTION,
            points_selector=FilterSelector(filter=flt),
            wait=True,
        )

    _retry(_do_delete)


def index_batch(
    client: QdrantClient,
    chunks: list[dict],
    changed_files: Iterable[str] = (),
    deleted_files: Iterable[str] = (),
    summarizer=None,
    batch_size: int = VECTOR_DB_BATCH_SIZE,
) -> bool:
    """Xoá chunk cũ của file đổi/xoá, rồi embed + upsert chunk mới theo lô.
    Trả True khi tất cả đã lưu xong; bên gọi chỉ commit index_state SAU khi nhận True.
    Lỗi giữa chừng thì ném exception, state chưa commit nên lần sau index lại (id ổn định nên upsert idempotent)."""
    delete_files(client, [*changed_files, *deleted_files])
    for i in range(0, len(chunks), batch_size):
        points = create_qdrant_points(chunks[i:i + batch_size], summarizer)

        def _do_upsert() -> None:
            client.upsert(
                collection_name=VECTOR_DB_COLLECTION,
                points=points,
                wait=True,
            )

        _retry(_do_upsert)
    return True