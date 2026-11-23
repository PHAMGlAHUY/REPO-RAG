import threading

from fastapi import APIRouter, HTTPException

from app.services.rag_runtime import runtime

router = APIRouter(prefix="/api/index", tags=["index"])


@router.get("/status")
def status() -> dict:
    return runtime.status()


@router.post("")
def trigger() -> dict:
    if not runtime.models_ready:
        raise HTTPException(409, "Model đang được nạp")
    if runtime.indexing:
        raise HTTPException(409, "Đang index")
    threading.Thread(target=runtime.reindex, daemon=True).start()
    return {"started": True}