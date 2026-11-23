from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from ingestion.check import STATE_FILE
from ingestion.indexer import ensure_models, run_index

from app.services.watcher import RepoWatcher

log = logging.getLogger("ingestion")


class RagRuntime:
    def __init__(self) -> None:
        self.repo: Path | None = None
        self.mode = "background"        # background | index_first
        self.models_ready = False
        self.indexing = False
        self.ready = False
        self.last_indexed: float | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._watcher: RepoWatcher | None = None

    def start(self) -> None:
        repo = os.environ.get("RAG_REPO_ROOT")
        if not repo:
            log.warning("RAG_REPO_ROOT chưa đặt, bỏ qua nạp model/watcher")
            self.ready = True
            return
        self.repo = Path(repo).resolve()
        self.mode = os.environ.get("RAG_BOOT_MODE", "background")
        threading.Thread(target=self._boot, name="rag-boot", daemon=True).start()

    def _boot(self) -> None:
        try:
            ensure_models()
            self.models_ready = True
            if self.mode == "index_first":
                self.reindex()
            self._watcher = RepoWatcher(self.repo, self.reindex)
            self._watcher.start()
        except Exception as e:
            log.exception("Boot lỗi")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.ready = True
            log.info("RAG runtime ready")

    def reindex(self) -> bool:
        """False nếu đang có lần index khác chạy."""
        if self.repo is None or not self._lock.acquire(blocking=False):
            return False
        self.indexing = True
        try:
            run_index(self.repo, STATE_FILE, close_client=False)
            self.last_indexed = time.time()
            self.error = None
        except Exception as e:
            log.exception("Index lỗi")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.indexing = False
            self._lock.release()
        return True

    def stop(self) -> None:
        if self._watcher:
            self._watcher.stop()

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "models_ready": self.models_ready,
            "indexing": self.indexing,
            "last_indexed": self.last_indexed,
            "error": self.error,
        }


runtime = RagRuntime()