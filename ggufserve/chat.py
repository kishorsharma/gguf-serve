"""Generation and reasoning-tag handling."""

from __future__ import annotations

import threading
from typing import Any, Iterator

from ggufserve import config
from ggufserve.system import heartbeat, info, ok, step

THINK_END = "</think>"
THINK_START = "<think>"

# One llama.cpp context cannot serve concurrent requests: letting two overlap
# corrupts the KV cache and interleaves tokens between responses. Requests queue
# here instead.
_lock = threading.Lock()


def split_response(content: str) -> tuple[str, str]:
    """Split raw output into (reasoning, answer).

    Reasoning models emit their scratchpad first and close it with `</think>`.
    The opening `<think>` is often absent, so the closing tag is what we key on.
    Text with no closing tag is treated as a plain answer, which is what makes
    this safe for ordinary models.
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


def separate(content: str) -> tuple[str, str]:
    """Like `split_response`, but honours the PARSE_REASONING setting."""
    if not config.PARSE_REASONING:
        return "", content.strip()
    return split_response(content)


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
    # create_chat_completion, so template switches such as Qwen's
    # `enable_thinking` cannot be forwarded. Reasoning is always generated and
    # stripped afterwards.
    with _lock:
        for chunk in llm.create_chat_completion(**params):
            piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if piece:
                yield piece


def smoke_test(llm) -> None:
    """Prove the model actually produces tokens before we expose a public URL."""
    step("Running smoke test")

    prompt = "In one short sentence, what is a large language model?"
    info(f"prompt : {prompt}")

    with heartbeat("generating"):
        raw = "".join(
            generate(
                llm,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=192,
            )
        )

    _, answer = separate(raw)
    info(f"answer : {answer.replace(chr(10), ' ')[:160]}")

    if not answer.strip():
        raise SystemExit(
            "The model loaded but produced no answer, so the server was not "
            "started. See docs/troubleshoot.md."
        )

    ok("model is generating")
