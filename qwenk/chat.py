"""Generation and Qwen reasoning-tag handling.

Originally cells "06 - Smoke test", "07 - Qwen reasoning parser" and the
`generate_sync` helper from cell 12.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from qwenk import config
from qwenk.system import ok, step

THINK_END = "</think>"
THINK_START = "<think>"

# This 2xT4 setup is configured for one generation at a time: a single llama.cpp
# context cannot serve concurrent requests, and letting two overlap corrupts the
# KV cache and interleaves tokens between responses. Requests queue here.
_lock = threading.Lock()


def split_response(content: str) -> tuple[str, str]:
    """Split raw Qwen output into (reasoning, answer).

    Qwen3.8 emits its reasoning first and closes it with `</think>`. In practice
    the opening `<think>` is often absent, so the closing tag is what we key on.
    Text with no closing tag is treated as a plain answer.
    """
    if not content:
        return "", ""

    if THINK_END not in content:
        return "", content.strip()

    reasoning, answer = content.split(THINK_END, 1)
    reasoning = reasoning.strip()

    if reasoning.startswith(THINK_START):
        reasoning = reasoning[len(THINK_START):].strip()

    return reasoning, answer.strip()


def generate(
    llm,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Stream generated text fragments, holding the inference lock throughout.

    Yields the model's raw output, reasoning tags included. Callers decide what
    to do with the `</think>` boundary.
    """
    params = {
        "messages": messages,
        "temperature": float(config.TEMPERATURE if temperature is None else temperature),
        "top_p": float(config.TOP_P if top_p is None else top_p),
        "top_k": int(config.TOP_K if top_k is None else top_k),
        "max_tokens": int(config.MAX_TOKENS if max_tokens is None else max_tokens),
        "stream": True,
    }

    # Note: llama-cpp-python 0.3.35 does not accept `chat_template_kwargs` in
    # create_chat_completion, so Qwen's `enable_thinking` template switch cannot
    # be forwarded. Reasoning is always generated and stripped afterwards.
    with _lock:
        for chunk in llm.create_chat_completion(**params):
            piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if piece:
                yield piece


def smoke_test(llm) -> None:
    """Prove the model actually produces tokens before we expose a public URL."""
    step("Running smoke test")

    raw = "".join(
        generate(
            llm,
            [{"role": "user", "content": "What is 17 x 23? Answer briefly."}],
            temperature=0.2,
            max_tokens=128,
        )
    )

    _, answer = split_response(raw)
    print(f"   prompt : What is 17 x 23?")
    print(f"   answer : {answer.replace(chr(10), ' ')[:160]}")

    if not answer.strip():
        raise SystemExit(
            "The model loaded but produced no answer. Refusing to start the "
            "server. See docs/troubleshoot.md."
        )

    ok("model is generating")
