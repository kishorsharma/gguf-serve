"""OpenAI-compatible API routes.

Streaming responses carry the model's raw output, `</think>` reasoning tags
included, and the client separates them. That is what the bundled web UI does,
and it is the behaviour third-party OpenAI clients were tested against. Send
`extra_body.chat_template_kwargs.enable_thinking = true` to have the server
strip the reasoning instead and return only the final answer.

When the model writes Qwen `<tool_call>` / `<function=` markup — with or
without a `tools` array on the request — the server rewrites it into OpenAI
`tool_calls` instead of leaving the tags in `content`.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ggufserve import config
from ggufserve.chat import close_iterator, collect, iter_client_events, separate
from ggufserve.system import warn


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
        "content": answer or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, (reasoning if strip_reasoning else None)


def _stream_tool_calls(calls: list[dict]) -> list[dict]:
    streamed = []
    for index, call in enumerate(calls):
        streamed.append(
            {
                "index": index,
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": call["function"].get("arguments") or "",
                },
            }
        )
    return streamed


def _sse_response(iterator) -> StreamingResponse:
    return StreamingResponse(
        iterator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _abort_on_disconnect(
    http_request: Request, iterator: Iterator[str]
) -> AsyncIterator[str]:
    """Stop llama.cpp when Hermes hits stop / the client drops the stream.

    A sync StreamingResponse keeps sampling after the socket dies, holding the
    inference lock. Later requests then queue behind a dead client until they
    time out — which looks like the model 'rejecting everything' after cancel.
    """
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gguf-chat")

    def pull() -> str | None:
        try:
            return next(iterator)
        except StopIteration:
            return None

    try:
        while True:
            if await http_request.is_disconnected():
                if config.CHAT_LOG:
                    warn("client disconnected; aborting generation")
                break
            chunk = await loop.run_in_executor(pool, pull)
            if chunk is None:
                break
            yield chunk
            if await http_request.is_disconnected():
                if config.CHAT_LOG:
                    warn("client disconnected; aborting generation")
                break
    finally:
        try:
            await loop.run_in_executor(pool, close_iterator, iterator)
        finally:
            pool.shutdown(wait=True)


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
    def chat_completions(request: ChatCompletionRequest, http_request: Request):
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

        if request.stream:
            if strip_reasoning:
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
                )
                message, _reasoning = _assistant_message(raw, tool_calls, True)
                if not tool_calls:
                    finish_reason = "stop"

                def buffered_events():
                    try:
                        yield _sse(_chunk(request_id, {"role": "assistant"}))
                        content = message.get("content")
                        if content:
                            yield _sse(_chunk(request_id, {"content": content}))
                        if message.get("tool_calls"):
                            yield _sse(
                                _chunk(
                                    request_id,
                                    {
                                        "tool_calls": _stream_tool_calls(
                                            message["tool_calls"]
                                        )
                                    },
                                )
                            )
                        yield _sse(
                            _chunk(request_id, {}, finish_reason=finish_reason)
                        )
                        yield "data: [DONE]\n\n"
                    except Exception as error:
                        if config.CHAT_LOG:
                            warn(f"chat {request_id}: {type(error).__name__}: {error}")
                            traceback.print_exc()
                        yield _sse(
                            {
                                "error": {
                                    "message": str(error),
                                    "type": type(error).__name__,
                                }
                            }
                        )
                        yield "data: [DONE]\n\n"

                return _sse_response(buffered_events())

            def event_stream():
                events = None
                try:
                    yield _sse(_chunk(request_id, {"role": "assistant"}))
                    events = iter_client_events(
                        llm,
                        request.messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        max_tokens=request.max_tokens,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        functions=request.functions,
                    )
                    for event in events:
                        delta: dict[str, Any] = {}
                        if event.content:
                            delta["content"] = event.content
                        if event.tool_calls:
                            delta["tool_calls"] = _stream_tool_calls(event.tool_calls)
                        if delta:
                            yield _sse(_chunk(request_id, delta))
                        if event.finish_reason:
                            yield _sse(
                                _chunk(
                                    request_id, {}, finish_reason=event.finish_reason
                                )
                            )
                    yield "data: [DONE]\n\n"
                except Exception as error:
                    if config.CHAT_LOG:
                        warn(f"chat {request_id}: {type(error).__name__}: {error}")
                        traceback.print_exc()
                    yield _sse(
                        {"error": {"message": str(error), "type": type(error).__name__}}
                    )
                    yield "data: [DONE]\n\n"
                finally:
                    close_iterator(events)

            return _sse_response(_abort_on_disconnect(http_request, event_stream()))

        try:
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
            )
        except Exception as error:
            if config.CHAT_LOG:
                warn(f"chat {request_id}: {type(error).__name__}: {error}")
                traceback.print_exc()
            return JSONResponse(
                {
                    "error": {
                        "message": str(error),
                        "type": type(error).__name__,
                    }
                },
                status_code=500,
            )
        message, reasoning = _assistant_message(raw, tool_calls, strip_reasoning)
        if not tool_calls:
            finish_reason = "stop"

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
