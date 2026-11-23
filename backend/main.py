from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import index
from app.core.config import settings
from app.services.rag_runtime import runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.start()  # nạp model + watcher ở thread nền
    yield
    runtime.stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(index.router)