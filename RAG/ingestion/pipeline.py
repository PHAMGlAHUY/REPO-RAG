"""Quét repository: đọc .gitignore bằng pathspec, lọc file, chạy song song chunker, re-index tăng dần, yield batch."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterator

from config.load_config import load_settings, setup_logging, _resolve
from ingestion.chunking.chunker import chunk_file
from ingestion.chunking.parsers import PARSER_FACTORIES, get_parser_for_extension
from ingestion.chunking.splitter import get_tokenizer

logger = logging.getLogger('ingestion')

cfg = load_settings().get('ingestion') or {}


def _cfg(key: str, default):
    """Đọc key dạng chữ thường, nếu không có thì thử dạng chữ hoa (vd: ignore_dirs / IGNORE_DIRS)."""
    return cfg.get(key, cfg.get(key.upper(), default))


IGNORE_DIRS: set[str] = set(_cfg('ignore_dirs', [
    '.git', '__pycache__', '.venv', 'venv', 'node_modules',
    '.pytest_cache', '.mypy_cache', '.tox', 'dist', 'build', '.idea', 'site-packages',
]) or [])
IGNORE_PATH_PARTS: set[str] = set(_cfg('ignore_path_parts', ['migrations']) or [])
SUPPORTED_EXTS: set[str] = {e.lower() for e in _cfg('supported_extensions', ['.py']) or []}
MAX_CHUNK_TOKENS: int = int(_cfg('max_chunk_tokens', 400) or 400)
MIN_CHUNK_TOKENS: int = int(_cfg('min_chunk_tokens', 60))
MAX_FILE_SIZE_MB: float = float(_cfg('max_file_size_mb', 2))  # Giới hạn kích thước file để tránh nghẽn RAM
STATE_FILE: Path = Path(_resolve(_cfg('state_file', 'data/index_state.json')))

_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def should_ignore_file(file_path: Path, rel_path: str) -> bool:
    name = file_path.name
    # Bỏ qua các file sinh tự động phổ biến
    if name.endswith("_pb2.py") or name.endswith("_pb2_grpc.py"):
        return True
    # Chỉ xét các thành phần của đường dẫn tương đối (không tính thư mục cha của repo)
    if IGNORE_PATH_PARTS & set(Path(rel_path).parts[:-1]):
        return True

    # Kiểm tra kích thước file (bỏ qua file quá lớn)
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return True
    except OSError:
        return True

    return False


def load_gitignore_spec(repo_path: Path):
    """Đọc .gitignore (nếu có) bằng pathspec để loại trừ file/thư mục theo chuẩn Git."""
    gitignore_path = repo_path / '.gitignore'
    if not gitignore_path.is_file():
        return None
    try:
        import pathspec  # type: ignore
        lines = gitignore_path.read_text(encoding='utf-8', errors='replace').splitlines()
        return pathspec.PathSpec.from_lines('gitwildmatch', lines)
    except ImportError:
        logger.warning("Chưa cài 'pathspec', bỏ qua bước đọc .gitignore. Cài bằng: pip install pathspec")
        return None
    except Exception as e:
        logger.warning(f"Không thể đọc .gitignore tại {gitignore_path}: {e}")
        return None


def compute_file_hash(file_path: Path) -> str:
    """Tính SHA-256 của file để phục vụ re-index tăng dần (cache theo đường dẫn + mtime + kích thước)."""
    st = file_path.stat()
    key = (str(file_path), st.st_mtime_ns, st.st_size)
    cached = _HASH_CACHE.get(key)
    if cached:
        return cached
    h = hashlib.sha256()
    with file_path.open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    _HASH_CACHE[key] = digest = h.hexdigest()
    return digest


def load_index_state(state_file: str | Path | None) -> dict[str, str]:
    """Đọc mapping {relative_path: file_sha256} từ lần index trước."""
    if not state_file:
        return {}
    p = Path(state_file)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Không đọc được index state {p}, chạy full scan: {e}")
        return {}


def save_index_state(state_file: str | Path | None, state: dict[str, str]) -> None:
    """Lưu trạng thái hash file (ghi nguyên tử: ghi file tạm rồi thay thế)."""
    if not state_file:
        return
    p = Path(state_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def _iter_candidate_files(repo_path: Path, gitignore_spec) -> Iterator[tuple[Path, str]]:
    """Duyệt file hợp lệ theo IGNORE_DIRS, .gitignore (pathspec), kích thước và extension hỗ trợ."""
    repo_resolved = repo_path.resolve()

    for root, dirs, files in os.walk(repo_resolved, followlinks=False):
        root_path = Path(root)
        try:
            rel_root = root_path.relative_to(repo_resolved).as_posix()
        except ValueError:
            dirs[:] = []
            continue
        rel_root = '' if rel_root == '.' else rel_root

        # Cắt nhánh: không đi vào thư mục bị ignore (so khớp theo tên, không theo đường dẫn tuyệt đối)
        kept_dirs: list[str] = []
        for d in sorted(dirs):
            if d in IGNORE_DIRS:
                continue
            if gitignore_spec is not None:
                rel_dir = f"{rel_root}/{d}" if rel_root else d
                if gitignore_spec.match_file(rel_dir + '/'):
                    continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for name in sorted(files):
            file_path = root_path / name
            rel_path = f"{rel_root}/{name}" if rel_root else name

            if file_path.is_symlink() or not file_path.is_file():
                continue
            if should_ignore_file(file_path, rel_path):
                continue
            if gitignore_spec is not None and gitignore_spec.match_file(rel_path):
                continue

            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTS or ext not in PARSER_FACTORIES:
                logger.debug(f"Skipping file for unsupported parser {ext} : {file_path}")
                continue

            yield file_path, rel_path


def _scan(repo_path: Path, gitignore_spec) -> tuple[dict[str, tuple[Path, str]], set[str]]:
    """Quét + hash. Trả về ({rel_path: (path, sha256)}, tập rel_path không tính được hash)."""
    files: dict[str, tuple[Path, str]] = {}
    failed: set[str] = set()
    for file_path, rel_path in _iter_candidate_files(repo_path, gitignore_spec):
        try:
            files[rel_path] = (file_path, compute_file_hash(file_path))
        except Exception as e:
            logger.error(f"Lỗi tính hash file {file_path}: {e}")
            failed.add(rel_path)
    return files, failed


def detect_changes(repo_root: str | Path, previous_state: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Chỉ quét + so hash. Trả về (changed, deleted, new)."""
    repo_path = Path(repo_root).resolve()
    files, failed = _scan(repo_path, load_gitignore_spec(repo_path))
    changed: list[str] = []
    new: list[str] = []
    for rel, (_, h) in files.items():
        old = previous_state.get(rel)
        if old is None:
            new.append(rel)
        elif old != h:
            changed.append(rel)
    deleted = sorted(set(previous_state) - set(files) - failed)
    return changed, deleted, new


def _init_worker_process() -> None:
    """Khởi tạo logging, tokenizer và parser riêng cho từng worker process."""
    setup_logging()
    get_tokenizer()
    for ext in PARSER_FACTORIES:
        get_parser_for_extension(ext)


def _chunk_single_file_worker(task: tuple[str, str, str, int, int, str]) -> tuple[str, str, list[dict]]:
    """Hàm chạy trong worker process: chunk 1 file và trả về (rel_path, file_hash, chunks)."""
    file_str, repo_str, rel_path, max_chunk_tokens, min_chunk_tokens, file_hash = task
    chunks = chunk_file(
        file_str,
        repo_str,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
    for ch in chunks:
        ch.setdefault('metadata', {})['file_hash'] = file_hash
    return rel_path, file_hash, chunks


def iter_repository_chunks(
    repo_root: str | Path,
    max_chunk_tokens: int | None = None,
    min_chunk_tokens: int | None = None,
    batch_size: int = 100,
    workers: int | None = None,
    state_file: str | Path | None = None,
    state_out: dict[str, str] | None = None,
    previous_state: dict[str, str] | None = None,
    deleted_files_out: list[str] | None = None,
    changed_files_out: list[str] | None = None,
) -> Iterator[list[dict]]:
    """Generator: quét song song & re-index tăng dần theo hash file, yield từng batch chunk để giải phóng RAM.

    - workers: số process chạy song song (mặc định os.cpu_count() or 1; đặt <= 1 để chạy tuần tự).
    - previous_state: dict {rel_path: sha256} đã load sẵn từ bên ngoài (ưu tiên hơn state_file).
      Truyền {} để buộc index lại toàn bộ (vd: collection đã mất / rỗng).
    - state_file: đường dẫn file JSON lưu {rel_path: sha256}. Chỉ dùng để load khi previous_state is None
      và chỉ ghi lại khi được truyền (caller tự quản lý commit thì để None).
    - deleted_files_out: nhận relative_path của file đã bị xóa / không còn được index.
    - changed_files_out: nhận relative_path của file ĐÃ ĐƯỢC index trước đó nhưng nội dung đã đổi.
      Bên lưu trữ phải xóa chunk cũ theo file_path của các file này (và của deleted_files_out)
      sau khi upsert các chunk mới yield ra.
    """
    max_chunk_tokens = max_chunk_tokens or MAX_CHUNK_TOKENS
    min_chunk_tokens = MIN_CHUNK_TOKENS if min_chunk_tokens is None else min_chunk_tokens
    repo_path = Path(repo_root).resolve()
    prev_state = previous_state if previous_state is not None else load_index_state(state_file)
    new_state: dict[str, str] = {}

    files, failed = _scan(repo_path, load_gitignore_spec(repo_path))
    for rel in failed:  # không tính được hash: giữ nguyên trạng thái cũ, chunk cũ vẫn còn trong index
        if rel in prev_state:
            new_state[rel] = prev_state[rel]

    tasks: list[tuple[str, str, str, int, int, str]] = []
    changed_rel_paths: list[str] = []
    for rel_path, (file_path, current_hash) in files.items():
        # Re-index tăng dần: nếu hash không đổi so với lần trước thì giữ hash cũ và bỏ qua chunk lại
        if prev_state.get(rel_path) == current_hash:
            new_state[rel_path] = current_hash
            continue
        if rel_path in prev_state:  # file đã từng index nhưng nội dung đổi -> chunk cũ phải bị xóa
            changed_rel_paths.append(rel_path)
        tasks.append((str(file_path), str(repo_path), rel_path, max_chunk_tokens, min_chunk_tokens, current_hash))

    if deleted_files_out is not None:
        deleted_files_out.clear()
        deleted_files_out.extend(sorted(set(prev_state) - set(files) - failed))

    if changed_files_out is not None:
        changed_files_out.clear()
        changed_files_out.extend(sorted(changed_rel_paths))

    num_workers = workers if workers is not None else (os.cpu_count() or 1)
    _batch: list[dict] = []

    def _consume_result(_rel_path: str, file_hash: str, _chunks: list[dict]) -> Iterator[list[dict]]:
        nonlocal _batch
        new_state[_rel_path] = file_hash
        for chunk in _chunks:
            _batch.append(chunk)
            if len(_batch) >= batch_size:
                yield _batch
                _batch = []

    if num_workers > 1 and len(tasks) > 1:
        window = num_workers * 2  # chỉ giữ một số task đang chạy để kết quả không dồn trong RAM
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker_process) as executor:
            task_iter = iter(tasks)
            pending = {}
            while True:
                while len(pending) < window:
                    t = next(task_iter, None)
                    if t is None:
                        break
                    pending[executor.submit(_chunk_single_file_worker, t)] = t[2]
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    _rel_path = pending.pop(future)
                    try:
                        r_path, f_hash, _chunks = future.result()
                    except BrokenProcessPool:
                        logger.error("Worker process bị dừng đột ngột, hủy lần index này (state không được lưu)")
                        raise
                    except Exception as e:
                        logger.error(f"Lỗi xử lý song song file {_rel_path}: {e}")
                        continue
                    yield from _consume_result(r_path, f_hash, _chunks)
    else:
        _init_worker_process()
        for t in tasks:
            try:
                r_path, f_hash, _chunks = _chunk_single_file_worker(t)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý file {t[2]}: {e}")
                continue
            yield from _consume_result(r_path, f_hash, _chunks)

    if _batch:  # phần dư cuối cùng
        yield _batch

    if state_out is not None:
        state_out.clear()
        state_out.update(new_state)  # gồm cả file không sinh chunk (vd __init__.py rỗng)

    save_index_state(state_file, new_state)