import uuid
import logging
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from config.load_config import load_settings, _resolve

logger = logging.getLogger('ingestion')

_cfg = load_settings()['embedding']
MAX_SEQ_LENGTH: int = _cfg.get('max_seq_length') or 512


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer | None:
    """Dùng đúng tokenizer.json của model ONNX để số token khớp với embedder."""
    try:
        return Tokenizer.from_file(str(Path(_resolve(_cfg['onnx_dir'])) / 'tokenizer.json'))
    except Exception as e:
        logger.warning(f"Unable to load tokenizer, using len//4 fallback: {e}")
        return None


@lru_cache(maxsize=4096)
def count_tokens_approx(text: str) -> int:
    if not text:
        return 0

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return len(text) // 4


def generate_chunk_id(file_path: str, identity: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path}:{identity}'))


def split_long_line(line: str, max_tokens: int) -> list[str]:
    total = count_tokens_approx(line)
    if total <= max_tokens:
        return [line]

    chunks: list[str] = []

    step = max(1, len(line) // ((total // max_tokens) + 1))
    start = 0
    while start < len(line):
        end = min(start + step, len(line))
        while end > start + 1 and count_tokens_approx(line[start:end]) > max_tokens:
            end = start + (end - start) // 2
        chunks.append(line[start:end])
        start = end
    return chunks


def split_large_text_ranges(text: str, max_tokens: int = 400) -> list[tuple[str, int, int]]:
    """Trả về (nội dung, chỉ số dòng đầu, chỉ số dòng cuối) của từng phần,
    đếm từ 0 trên text.split('\n'), để biết mỗi phần nằm ở dòng nào."""
    lines = text.split('\n')
    if count_tokens_approx(text) <= max_tokens:
        return [(text, 0, max(len(lines) - 1, 0))]

    parts: list[tuple[str, int, int]] = []
    current: list[tuple[int, str]] = []  # (chỉ số dòng, nội dung dòng)
    current_tokens = 0

    for idx, line in enumerate(lines):
        # Một dòng đơn lẻ có thể đã vượt quá max_tokens
        for sub_line in split_long_line(line, max_tokens):
            line_tokens = count_tokens_approx(sub_line) + 1
            if current and current_tokens + line_tokens > max_tokens:
                parts.append(('\n'.join(l for _, l in current), current[0][0], current[-1][0]))
                current, current_tokens = [], 0

            current.append((idx, sub_line))
            current_tokens += line_tokens

    if current:
        parts.append(('\n'.join(l for _, l in current), current[0][0], current[-1][0]))

    return parts