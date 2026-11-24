import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "RAG"))  # để import config, embedding, vectorstore, retrieval

from fastapi import FastAPI
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    from config.load_config import setup_logging
    from embedding.embedder import warmup
    from vectorstore.index import warmup_sparse
    from vectorstore.qdrant import get_qdrant_client, close_qdrant_client

    setup_logging()
    app.state.client = get_qdrant_client()
    warmup()          # nạp model dense một lần khi khởi động
    warmup_sparse()   # nạp model sparse
    yield
    close_qdrant_client()


app = FastAPI(lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    mode: Literal["auto", "hybrid", "dense", "sparse"] = "auto"
    file_path: str | None = None
    chunk_type: list[str] | None = None
    expand_parent: bool = False
    merge_parts: bool = False


@app.post("/search")
def search_endpoint(req: SearchRequest):  # dùng def (không async): FastAPI tự chạy trong threadpool
    from retrieval.retriever import retrieve

    chunks = retrieve(
        app.state.client,
        req.query,
        top_k=req.top_k,
        mode=req.mode,
        file_path=req.file_path,
        chunk_type=req.chunk_type,
        expand_parent=req.expand_parent,
        merge_parts=req.merge_parts,
    )
    return [asdict(c) for c in chunks]