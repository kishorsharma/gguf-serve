"""OpenAI-compatible API routes.

Originally cell "12 - Build the Gradio Server application".

Streaming responses carry the model's raw output, `</think>` reasoning tags
included, and the client separates them. That is what the bundled web UI does,
and it is the behaviour third-party OpenAI clients were tested against. Send
`extra_body.chat_template_kwargs.enable_thinking = true` to have the server
strip the reasoning instead and return only the final answer.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from qwenk import config
from qwenk.chat import generate, split_response


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = config.MODEL_ID
    messages: List[Dict[str, Any]]
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    top_k: int = config.TOP_K
    max_tokens: int = config.MAX_TOKENS
    stream: bool = False
    extra_body: Optional[Dict[str, Any]] = None


def _strip_reasoning_requested(request: ChatCompletionRequest) -> bool:
    """Read the vLLM-style `enable_thinking` flag out of `extra_body`."""
    extra = request.extra_body or {}
    kwargs = extra.get("chat_template_kwargs") or {}
    return bool(kwargs.get("enable_thinking", False))


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _chunk(request_id: str, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": config.MODEL_ID,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def register(app, llm) -> None:
    """Attach `/health`, `/v1/models` and `/v1/chat/completions` to `app`."""

    @app.get("/health", tags=["status"])
    async def health():
        return {
            "status": "ok",
            "model": config.MODEL_ID,
            "model_file": config.MODEL_FILE,
            "context": config.CTX_SIZE,
        }

    @app.get("/v1/models", tags=["openai"])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": config.MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen-k",
                }
            ],
        }

    # Defined with `def`, not `async def`, on purpose: generation is a blocking
    # call, so FastAPI must run it in a worker thread. Declaring it async would
    # stall the event loop for the whole request and make every other route,
    # including /health, hang until generation finished.
    @app.post("/v1/chat/completions", tags=["openai"])
    def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            return JSONResponse(
                {
                    "error": {
                        "message": "messages is required",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )

        strip_reasoning = _strip_reasoning_requested(request)
        request_id = "chatcmpl-" + uuid.uuid4().hex

        stream = generate(
            llm,
            request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )

        if not request.stream:
            raw = "".join(stream)
            reasoning, answer = split_response(raw)

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                # Zeroed rather than omitted: streaming llama.cpp does not
                # report token counts, and clients that reach for
                # `response.usage.total_tokens` crash on a missing field.
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                # Non-standard, and only populated when the caller asked for the
                # reasoning to be separated out.
                "reasoning_content": reasoning if strip_reasoning else None,
            }

        def event_stream():
            buffer = ""
            try:
                for piece in stream:
                    buffer += piece
                    if not strip_reasoning:
                        yield _sse(
                            _chunk(request_id, {"role": "assistant", "content": piece})
                        )

                if strip_reasoning:
                    _, answer = split_response(buffer)
                    yield _sse(
                        _chunk(request_id, {"role": "assistant", "content": answer})
                    )

                yield _sse(_chunk(request_id, {}, finish_reason="stop"))
                yield "data: [DONE]\n\n"

            except Exception as error:  # surfaced to the client, not swallowed
                yield _sse(
                    {"error": {"message": str(error), "type": type(error).__name__}}
                )
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Stops the share tunnel from buffering the stream into one blob.
                "X-Accel-Buffering": "no",
            },
        )
