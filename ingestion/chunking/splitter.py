import uuid
import logging
from functools import lru_cache
from config.load_config import load_settings
from transformers import PreTrainedTokenizer

settings = load_settings()
logger = logging.getLogger('ingestion')

_cfg = settings['embedding']
EMBEDDING_MODEL = _cfg['model']

@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizer | None:
    try:
        from transformers import AutoTokenizer
        model = load_settings()['embedding']['model']
        return AutoTokenizer.from_pretrained(model)
    except Exception as e:
        logger.warning(f"Unable to load tokenizer, using len//4 fallback: {e}")
        return None

def count_tokens_approx(text: str) -> int:
    if not text:
        return 0

    tokenizer = get_tokenizer()

    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return len(text) // 4

def generate_chunk_id(file_path: str, identity: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path}:{identity}'))

def split_long_line(line: str, max_tokens: int) -> list[str]:
    if count_tokens_approx(line) <= max_tokens:
        return [line]

    chunks: list[str] = []

    step = max(1, len(line) // max(1, (count_tokens_approx(line) // max_tokens) + 1))
    start = 0
    while start < len(line):
        end = min(start + step, len(line))
        while end > start + 1 and count_tokens_approx(line[start:end]) > max_tokens:
            end = start + (end - start) // 2
        chunks.append(line[start:end])
        start = end
    return chunks

def split_large_text_ranges(text: str, max_tokens: int = 400, overlap_lines: int = 5) -> list[tuple[str, int, int]]:
    """Như split_large_text nhưng trả thêm (chỉ số dòng đầu, chỉ số dòng cuối) của từng phần,
    đếm từ 0 trên text.split('\n'), để biết mỗi phần nằm ở dòng nào."""
    lines = text.split('\n')
    if count_tokens_approx(text) <= max_tokens:
        return [(text, 0, max(len(lines) - 1, 0))]

    parts: list[tuple[str, int, int]] = []
    current: list[tuple[int, str]] = []  # (chỉ số dòng, nội dung dòng)
    current_tokens = 0

    for idx, line in enumerate(lines):
        # Xử lý nếu bản thân 1 dòng đơn lẻ đã vượt quá max_tokens
        sub_lines = split_long_line(line, max_tokens)

        for sub_line in sub_lines:
            line_tokens = count_tokens_approx(sub_line) + 1
            if current and current_tokens + line_tokens > max_tokens:
                parts.append(('\n'.join(l for _, l in current), current[0][0], current[-1][0]))
                current = current[-overlap_lines:] if overlap_lines > 0 else []
                current_tokens = sum(count_tokens_approx(l) + 1 for _, l in current)
                while current and current_tokens + line_tokens > max_tokens:
                    current_tokens -= count_tokens_approx(current.pop(0)[1]) + 1

            current.append((idx, sub_line))
            current_tokens += line_tokens

    if current:
        parts.append(('\n'.join(l for _, l in current), current[0][0], current[-1][0]))

    return parts


def split_large_text(text: str, max_tokens: int = 400, overlap_lines: int = 5) -> list[str]:
    return [part for part, _, _ in split_large_text_ranges(text, max_tokens, overlap_lines)]