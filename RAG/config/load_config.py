import copy
import logging.config
import os
from functools import lru_cache
from pathlib import Path

import yaml

# Thư mục gốc dự án (config/ nằm ngay dưới gốc) -> không phụ thuộc thư mục chạy hiện tại
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Nạp file .env (KEY=VALUE) vào os.environ; không ghi đè biến đã có sẵn trong môi trường.
    Tìm ở thư mục gốc dự án (RAG/) rồi thư mục cha của nó."""
    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


_load_dotenv()


def _resolve(path: str | Path) -> Path:
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=None)
def _read_settings(path: str) -> dict:
    with open(_resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}  # file rỗng -> {} thay vì None
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: nội dung phải là một mapping YAML")

    # Biến môi trường ghi đè (dùng trong Docker)
    vdb = cfg["vector_database"] = cfg.get("vector_database") or {}
    if os.environ.get("QDRANT_URL"):
        vdb["url"] = os.environ["QDRANT_URL"]
    if os.environ.get("QDRANT_API_KEY"):
        vdb["api_key"] = os.environ["QDRANT_API_KEY"]
    if os.environ.get("QDRANT_COLLECTION"):
        vdb["collection_name"] = os.environ["QDRANT_COLLECTION"]
    return cfg


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    """Đọc + cache file cấu hình; trả về bản sao để nơi gọi sửa thoải mái."""
    return copy.deepcopy(_read_settings(str(path)))


def setup_logging(path: str | Path = "config/logging.yaml") -> None:
    with open(_resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # FileHandler sẽ crash nếu thư mục logs/ chưa tồn tại
    for h in (cfg.get("handlers") or {}).values():
        if "filename" in h:
            h["filename"] = str(_resolve(h["filename"]))
            Path(h["filename"]).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(cfg)