"""Client Ollama cho RAG: viết lại truy vấn và sinh câu trả lời."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import httpx

from .prompt import (
    CHARS_PER_TOKEN,
    Message,
    build_messages,
    build_rewrite_messages,
    invalid_citations,
    is_vietnamese,
    no_answer_text,
)

logger = logging.getLogger(__name__)

MAX_REWRITE_CHARS = 300
WARN_RATIO = 0.95
_RETRYABLE = (ConnectionError, TimeoutError, httpx.TransportError)

_REWRITE_PREFIX_RE = re.compile(r"^\s*(?:truy vấn(?: tiếng anh| độc lập)?|câu hỏi(?: độc lập)?|(?:english |standalone )?query)\s*:\s*", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_IDENT_RE = re.compile(r"\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b|\b\w+\.\w+\b")
_PRONOUN_RE = re.compile(r"\b(nó|này|đó|kia|trên|vậy|it|this|that|them)\b", re.I)


def needs_rewrite(question: str, history: Sequence[Message]) -> bool:
    """Câu hỏi tiếng Việt luôn được chuyển sang truy vấn tiếng Anh (BM25 + reranker chỉ mạnh với tiếng Anh).
    Câu hỏi khác chỉ rewrite khi thiếu ngữ cảnh (đại từ mơ hồ, quá ngắn, không có định danh)."""
    if is_vietnamese(question):
        return True
    if not history:
        return False
    if _PRONOUN_RE.search(question):
        return True
    return not (_IDENT_RE.search(question) or len(question.split()) > 8)


def clean_rewritten_query(raw: str, original: str) -> str:
    """Lấy dòng truy vấn thật đầu tiên từ đầu ra của model."""
    for line in _THINK_RE.sub("", raw).splitlines():
        q = _REWRITE_PREFIX_RE.sub("", line).strip().strip("\"“”")
        if q:
            return q if len(q) <= MAX_REWRITE_CHARS else original
    return original


@dataclass
class AnswerResult:
    text: str
    used_chunks: list[Any] = field(default_factory=list)
    truncated: bool = False
    prompt_tokens: int = 0
    bad_citations: list[int] = field(default_factory=list)


class LLMClient:
    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str | None = None,
        num_ctx: int = 8192,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        seed: int | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        client: Any = None,
        check_model: bool = True,
        keep_alive: str | None = None,
    ):
        self.model = model
        self.retries = retries
        self.keep_alive = keep_alive
        self._cpt = float(CHARS_PER_TOKEN)  # ký tự/token, tự hiệu chỉnh theo prompt_eval_count
        self._params = dict(
            num_ctx=num_ctx, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, repeat_penalty=repeat_penalty, seed=seed,
        )

        if client is None:
            import ollama

            client = ollama.Client(host=base_url or os.getenv("OLLAMA_BASE_URL"), timeout=timeout)
        self._client = client
        if check_model:
            try:
                client.show(model)
            except Exception as e:
                raise RuntimeError(f"Không dùng được model '{model}' (đã `ollama pull {model}` và bật server chưa?): {e}") from e

    @classmethod
    def from_config(cls, settings: Mapping[str, Any], **kwargs: Any) -> "LLMClient":
        cfg = dict(settings.get("llm") or {})
        if not cfg:
            raise KeyError("Thiếu mục 'llm' trong config")
        cfg.pop("provider", None)
        if "model_name" in cfg:
            cfg["model"] = cfg.pop("model_name")
        return cls(**{**cfg, **kwargs})

    # ------------------------------------------------------------------ #

    def _merge(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        bad = set(overrides) - set(self._params)
        if bad:
            raise TypeError(f"Tham số không hợp lệ: {sorted(bad)}; hợp lệ: {sorted(self._params)}")
        return {**self._params, **overrides}

    def _kwargs(self, messages: list[Message], overrides: Mapping[str, Any], stream: bool, think: bool | None) -> dict[str, Any]:
        p = self._merge(overrides)
        options = {
            "num_ctx": p["num_ctx"], "num_predict": p["max_tokens"], "temperature": p["temperature"],
            "top_p": p["top_p"], "repeat_penalty": p["repeat_penalty"],
        }
        if p["seed"] is not None:
            options["seed"] = p["seed"]
        kw: dict[str, Any] = {"model": self.model, "messages": messages, "options": options, "stream": stream}
        if think is not None:
            kw["think"] = think

        if self.keep_alive:
            kw["keep_alive"] = self.keep_alive

        return kw

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        for attempt in range(self.retries):
            try:
                return fn()
            except _RETRYABLE:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)

    def _observe(self, messages: list[Message], resp: Any, num_ctx: int) -> int:
        """Đọc prompt_eval_count: cảnh báo khi gần đầy num_ctx và hiệu chỉnh ký tự/token."""
        tokens = resp.get("prompt_eval_count") or 0
        if tokens:
            sent = sum(len(m["content"]) for m in messages)
            self._cpt = min(self._cpt, max(sent / tokens, 1.0))
            if tokens >= WARN_RATIO * num_ctx:
                logger.warning("Prompt %d/%d token: Ollama có thể đã cắt prompt", tokens, num_ctx)
        return tokens

    def _chat(self, messages: list[Message], overrides: Mapping[str, Any], think: bool | None) -> tuple[str, int]:
        resp = self._with_retry(lambda: self._client.chat(**self._kwargs(messages, overrides, False, think)))
        tokens = self._observe(messages, resp, self._merge(overrides)["num_ctx"])
        return resp["message"]["content"], tokens

    # ------------------------------------------------------------------ #

    def generate(self, messages: list[Message], think: bool | None = None, **overrides: Any) -> str:
        return self._chat(messages, overrides, think)[0]

    def stream(self, messages: list[Message], think: bool | None = None, **overrides: Any) -> Iterator[str]:
        num_ctx = self._merge(overrides)["num_ctx"]
        it = self._client.chat(**self._kwargs(messages, overrides, True, think))
        try:
            for part in it:
                if text := part["message"]["content"]:
                    yield text
                if part.get("done"):
                    self._observe(messages, part, num_ctx)
        finally:
            getattr(it, "close", lambda: None)()

    def prepare(
        self, question: str, chunks: Sequence[Any], history: Sequence[Message] | None = None, **overrides: Any
    ) -> tuple[list[Message], list[Any]]:
        """Dựng messages theo ngân sách num_ctx. Trả về (messages, chunk thực sự đưa vào prompt)."""
        p = self._merge(overrides)
        budget = int((p["num_ctx"] - p["max_tokens"]) * self._cpt)
        return build_messages(question, chunks, history, max_prompt_chars=budget)

    def rewrite_query(self, question: str, history: Sequence[Message]) -> str:
        if not needs_rewrite(question, history):
            return question
        messages = build_rewrite_messages(question, history)
        try:
            raw = self.generate(messages, think=False, temperature=0.0, max_tokens=128)
        except _RETRYABLE:
            raise
        except Exception:  # model không hỗ trợ tham số `think`
            raw = self.generate(messages, temperature=0.0, max_tokens=128)
        return clean_rewritten_query(raw, question)

    def answer(
        self, question: str, chunks: Sequence[Any], history: Sequence[Message] | None = None, **overrides: Any
    ) -> AnswerResult:
        if not chunks:
            return AnswerResult(no_answer_text(question))
        messages, used = self.prepare(question, chunks, history, **overrides)
        if not used:
            return AnswerResult(no_answer_text(question))
        text, tokens = self._chat(messages, overrides, None)
        num_ctx = self._merge(overrides)["num_ctx"]
        return AnswerResult(
            text=text,
            used_chunks=used,
            truncated=tokens >= WARN_RATIO * num_ctx or len(used) < len(chunks),
            prompt_tokens=tokens,
            bad_citations=invalid_citations(text, len(used)),
        )