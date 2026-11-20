"""Test bất biến cho AST Chunker:
1. Mọi dòng không trống trong file đều thuộc ít nhất 1 chunk.
2. Khoảng part_start_line..part_end_line khớp với nội dung thực tế của chunk.
"""
from pathlib import Path
import pytest
import re

from ingestion.chunking.chunker import chunk_file
from ingestion.pipeline import _iter_candidate_files, load_gitignore_spec

_PART_COMMENT_RE = re.compile(r"^#\s+.*\(part\s+\d+/\d+\)\s*$")


_SKIP_PARTS = {".venv", "venv", ".git", "__pycache__", "site-packages", "dist", "build"}

# Các tiền tố dòng metadata do chunker tự chèn thêm vào content
_SYNTHETIC_PREFIXES = (
    "# File:",
    "# Class:",
    "# Node:",
    "# Imports:",
    "# Decorators:",
    "# Signature:",
    "# Docstring:",
    "# [Continuation of",
    "# Context Summary:",
)


def _strip_synthetic_headers(chunk_content: str, file_path: str) -> list[str]:
    """Loại bỏ các dòng header và nhãn part tự sinh để lấy lại đúng các dòng code gốc."""
    code_lines: list[str] = []
    for line in chunk_content.split('\n'):
        stripped = line.strip()
        if any(part in _SKIP_PARTS for part in file_path.parts):
            continue
        if line.startswith(_SYNTHETIC_PREFIXES):
            continue
        if _PART_COMMENT_RE.match(stripped):
            continue
        code_lines.append(line)
    return code_lines


def verify_file_chunk_invariants(file_path: Path, repo_root: Path, max_chunk_tokens: int = 400) -> None:
    raw_text = file_path.read_text(encoding='utf-8', errors='replace')
    file_lines = raw_text.split('\n')

    # Nếu file rỗng / chỉ toàn khoảng trắng thì bỏ qua
    if not raw_text.strip():
        return

    chunks = chunk_file(file_path, repo_root, max_chunk_tokens=max_chunk_tokens)
    assert chunks, f"File không rỗng nhưng không sinh ra chunk nào: {file_path}"

    covered_lines: set[int] = set()

    for ch in chunks:
        meta = ch['metadata']
        p_start = meta.get('part_start_line', meta['start_line'])
        p_end = meta.get('part_end_line', meta['end_line'])

        assert 1 <= p_start <= p_end <= len(file_lines), (
            f"Khoảng dòng không hợp lệ [{p_start}..{p_end}] trên file {len(file_lines)} dòng: {file_path}"
        )

        for ln in range(p_start, p_end + 1):
            covered_lines.add(ln)

        # Kiểm tra độ khớp nội dung: mọi dòng code không trống trong chunk phải xuất hiện
        # trong cửa sổ dòng [p_start..p_end] (cho phép lùi thêm vài dòng signature phía trên)
        sig_start = meta.get('start_line', p_start)
        window_start = max(1, min(sig_start, p_start))
        source_slice_stripped = {
            line.strip() for line in file_lines[window_start - 1 : p_end] if line.strip()
        }

        chunk_code_lines = [l.strip() for l in _strip_synthetic_headers(ch['content'], file_path=file_path) if l.strip()]
        for c_line in chunk_code_lines:
            assert c_line in source_slice_stripped, (
                f"Lệch nội dung tại {file_path}:{p_start}-{p_end} (node={meta.get('qualified_name')}):\n"
                f"  Dòng trong chunk: {c_line!r}\n"
                f"  Không tìm thấy trong file gốc dòng [{window_start}..{p_end}]"
            )

    # Bất biến 1: Mọi dòng không trống trong file phải được phủ bởi ít nhất 1 chunk
    uncovered_non_empty: list[tuple[int, str]] = []
    for idx, line in enumerate(file_lines, start=1):
        if line.strip() and idx not in covered_lines:
            uncovered_non_empty.append((idx, line))

    assert not uncovered_non_empty, (
        f"Có {len(uncovered_non_empty)} dòng không trống bị bỏ sót trong {file_path}:\n"
        + "\n".join(f"  Dòng {ln}: {txt}" for ln, txt in uncovered_non_empty[:10])
    )


@pytest.mark.parametrize("max_tokens", [150, 400, 800])
def test_repository_chunk_invariants(max_tokens: int):
    repo_root = Path(__file__).resolve().parents[1]
    gitignore_spec = load_gitignore_spec(repo_root)
    files = list(_iter_candidate_files(repo_root, gitignore_spec))
    assert files, "Không tìm thấy file mã nguồn nào để kiểm thử."

    for file_path, _ in files:
        verify_file_chunk_invariants(file_path, repo_root, max_chunk_tokens=max_tokens)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    spec = load_gitignore_spec(root)
    checked = 0
    for f_path, _ in _iter_candidate_files(root, spec):
        verify_file_chunk_invariants(f_path, root, max_chunk_tokens=400)
        checked += 1
    print(f"PASS: Tất cả {checked} file đều thỏa mãn 100% bất biến chunking.")