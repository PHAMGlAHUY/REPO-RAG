from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("ingestion")

DEBOUNCE = 3.0
IGNORED = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".idea", "logs", "data"}


class RepoWatcher(FileSystemEventHandler):
    def __init__(self, repo: Path, on_change: Callable[[], bool]) -> None:
        self.repo = repo
        self.on_change = on_change
        self._timer: threading.Timer | None = None
        self._obs = Observer()

    def on_any_event(self, event) -> None:
        if event.is_directory or IGNORED.intersection(Path(event.src_path).parts):
            return
        if not str(event.src_path).endswith(".py"):
            return
        self._schedule()

    def _schedule(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        if not self.on_change():   # đang index -> hẹn lại
            self._schedule()

    def start(self) -> None:
        self._obs.schedule(self, str(self.repo), recursive=True)
        self._obs.start()
        log.info("Watching %s", self.repo)

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._obs.stop()
        self._obs.join(timeout=5)