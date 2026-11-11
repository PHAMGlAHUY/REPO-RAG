from pathlib import Path
from .python_chunker import chunk_python
from .text_chunker import chunk_text_file


def chunk_file(path: Path, root: Path) -> list[dict]:
    return chunk_python(path, root) if path.suffix == ".py" else chunk_text_file(path, root)