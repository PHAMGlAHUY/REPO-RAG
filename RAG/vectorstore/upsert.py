import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, FilterSelector, HasIdCondition, MatchAny

from vectorstore.index import create_qdrant_points
from vectorstore.qdrant import VECTOR_DB_BATCH_SIZE, VECTOR_DB_COLLECTION

logger = logging.getLogger('vector_database')

MAX_RETRIES = 3


def retry(fn: Callable[[], object]) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            fn()
            return
        except (ResponseHandlingException, ConnectionError, TimeoutError, UnexpectedResponse) as e:
            if isinstance(e, UnexpectedResponse) and e.status_code < 500:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"Qdrant lỗi ({e}), thử lại sau {wait}s")
            time.sleep(wait)


_retry = retry  # tên cũ


def delete_stale(client: QdrantClient, file_paths: Iterable[str], keep_ids: list | None = None) -> None:
    """Xoá chunk của các file; bỏ qua các id trong keep_ids (chunk mới vừa upsert)."""
    file_paths = sorted(set(file_paths))
    if not file_paths:
        return
    flt = Filter(
        must=[FieldCondition(key='file_path', match=MatchAny(any=file_paths))],
        must_not=[HasIdCondition(has_id=keep_ids)] if keep_ids else None,
    )
    retry(lambda: client.delete(
        collection_name=VECTOR_DB_COLLECTION,
        points_selector=FilterSelector(filter=flt),
        wait=True,
    ))


def delete_files(client: QdrantClient, file_paths: Iterable[str]) -> None:
    """Xoá toàn bộ chunk của các file trong một lần gọi."""
    delete_stale(client, file_paths)


def index_batch(
    client: QdrantClient,
    chunks: list[dict],
    changed_files: Iterable[str] = (),
    deleted_files: Iterable[str] = (),
    batch_size: int = VECTOR_DB_BATCH_SIZE,
) -> None:
    """Upsert chunk mới trước, sau đó xoá chunk cũ của file đổi/xoá. Lỗi thì ném exception."""
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    if batches:
        with ThreadPoolExecutor(max_workers=1) as ex:  # embed batch kế tiếp trong lúc upsert batch hiện tại
            fut = ex.submit(create_qdrant_points, batches[0])
            for k in range(len(batches)):
                points = fut.result()
                if k + 1 < len(batches):
                    fut = ex.submit(create_qdrant_points, batches[k + 1])
                retry(lambda p=points: client.upsert(collection_name=VECTOR_DB_COLLECTION, points=p, wait=True))

    changed = sorted(set(changed_files))
    delete_stale(client, changed, keep_ids=[c['id'] for c in chunks])
    delete_files(client, set(deleted_files) - set(changed))