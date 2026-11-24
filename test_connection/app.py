import uuid
from typing import Any

import gradio as gr
import spaces
from fastapi import Body, HTTPException

DB: dict[str, Any] = {}  # lưu trong RAM, Space restart thì mất


@spaces.GPU
def ping():  # chỉ để ZeroGPU không báo lỗi
    return "pong"


with gr.Blocks() as repo:
    btn = gr.Button("Ping")
    out = gr.Textbox()
    btn.click(ping, None, out)


def health():
    return {"status": "ok"}


def post_json(payload: Any = Body(...)):
    key = uuid.uuid4().hex[:8]
    DB[key] = payload
    return {"id": key}


def get_json(key: str):
    if key not in DB:
        raise HTTPException(404, "Không tìm thấy")
    return DB[key]


def add_route(path, fn, method, status_code=200):
    app = repo.app
    app.add_api_route(path, fn, methods=[method], status_code=status_code)
    # đưa route mới lên đầu để không bị route của Gradio chặn
    app.router.routes.insert(0, app.router.routes.pop())


if __name__ == "__main__":
    # launch trước (Gradio tạo app), rồi mới thêm route vào app đang chạy
    repo.launch(ssr_mode=False, prevent_thread_lock=True)

    add_route("/health", health, "GET")
    add_route("/json", post_json, "POST", status_code=201)
    add_route("/json/{key}", get_json, "GET")

    repo.block_thread()