import logging.config
import os
from pathlib import Path

import yaml

# Thư mục gốc dự án (config/ nằm ngay dưới gốc) -> không phụ thuộc thư mục chạy hiện tại
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    with open(_resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}  # file rỗng -> {} thay vì None

    # Biến môi trường ghi đè (dùng trong Docker)
    vdb = cfg.setdefault("vector_database", {})
    if os.environ.get("QDRANT_URL"):
        vdb["url"] = os.environ["QDRANT_URL"]
    if os.environ.get("QDRANT_API_KEY"):
        vdb["api_key"] = os.environ["QDRANT_API_KEY"]
    return cfg


def setup_logging(path: str | Path = "config/logging.yaml") -> None:
    with open(_resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # FileHandler sẽ crash nếu thư mục logs/ chưa tồn tại
    for h in cfg.get("handlers", {}).values():
        if "filename" in h:
            h["filename"] = str(_resolve(h["filename"]))
            Path(h["filename"]).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(cfg)