"""OpenAI-compatible API routes.

Streaming responses carry the model's raw output, `</think>` reasoning tags
included, and the client separates them. That is what the bundled web UI does,
and it is the behaviour third-party OpenAI clients were tested against. Send
`extra_body.chat_template_kwargs.enable_thinking = true` to have the server
strip the reasoning instead and return only the final answer.

When the request includes `tools`, the server forwards them to llama.cpp and
returns OpenAI `tool_calls` — including Qwen `<tool_call>` markup that
llama-cpp-python would otherwise leave in `content`.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ggufserve import config
from ggufserve.chat import collect, generate, separate


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    top_k: int = config.TOP_K
    max_tokens: int = config.MAX_TOKENS
    stream: bool = False
    extra_body: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Dict[str, Any]]] = None


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
        "model": config.model_id(),
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _usage() -> dict:
    # Zeroed rather than omitted: streaming llama.cpp does not report token
    # counts, and clients that reach for `response.usage.total_tokens` crash
    # on a missing field.
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _assistant_message(
    raw: str,
    tool_calls: list[dict] | None,
    strip_reasoning: bool,
) -> tuple[dict, str | None]:
    reasoning, answer = separate(raw)
    message: dict[str, Any] = {
        "role": "assistant",
        # Hermes treats a non-null content field as a finished answer even
        # when tool_calls are also present, so keep content empty on a call.
        "content": None if tool_calls else answer,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, (reasoning if strip_reasoning else None)


def _buffered_stream(request_id: str, message: dict, finish_reason: str):
    """Turn a finished completion into OpenAI chat.completion.chunk events.

    Used when `tools` were sent: the full generation is collected first so
    Qwen XML can be rewritten as `delta.tool_calls` instead of being streamed
    as ordinary text that Hermes would treat as a finished answer.
    """
    yield _sse(_chunk(request_id, {"role": "assistant"}))

    tool_calls = message.get("tool_calls")
    content = message.get("content")
    if tool_calls:
        for index, call in enumerate(tool_calls):
            yield _sse(
                _chunk(
                    request_id,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call.get("id"),
                                "type": call.get("type") or "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": call["function"].get("arguments") or "",
                                },
                            }
                        ]
                    },
                )
            )
    elif content:
        yield _sse(_chunk(request_id, {"content": content}))

    yield _sse(_chunk(request_id, {}, finish_reason=finish_reason))
    yield "data: [DONE]\n\n"


def register(app, llm) -> None:
    """Attach `/health`, `/v1/models` and `/v1/chat/completions` to `app`."""

    @app.get("/health", tags=["status"])
    async def health():
        return {
            "status": "ok",
            "model": config.model_id(),
            "model_file": config.MODEL_FILE,
            "context": config.CTX_SIZE,
            "parse_reasoning": config.PARSE_REASONING,
        }

    @app.get("/v1/models", tags=["openai"])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id(),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "gguf-serve",
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
        parse_markup = bool(request.tools or request.functions)
        # Live token streaming is only safe when we are not going to rewrite
        # the output as tool_calls. With tools present, collect first.
        live_stream = request.stream and not parse_markup

        if live_stream:
            stream = generate(
                llm,
                request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
            )

            def event_stream():
                buffer = ""
                try:
                    for piece in stream:
                        buffer += piece
                        if not strip_reasoning:
                            yield _sse(
                                _chunk(
                                    request_id, {"role": "assistant", "content": piece}
                                )
                            )

                    if strip_reasoning:
                        _, answer = separate(buffer)
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
                    "X-Accel-Buffering": "no",
                },
            )

        raw, tool_calls, finish_reason = collect(
            llm,
            request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice,
            functions=request.functions,
            parse_markup=parse_markup,
        )
        message, reasoning = _assistant_message(raw, tool_calls, strip_reasoning)
        if not tool_calls:
            finish_reason = "stop"

        if request.stream:
            def buffered_events():
                try:
                    yield from _buffered_stream(request_id, message, finish_reason)
                except Exception as error:
                    yield _sse(
                        {"error": {"message": str(error), "type": type(error).__name__}}
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                buffered_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.model_id(),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(),
            "reasoning_content": reasoning,
        }
