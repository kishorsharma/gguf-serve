"""Generation, tool-call parsing, and reasoning-tag handling."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from ggufserve import config
from ggufserve.system import heartbeat, info, ok, step

THINK_END = "</think>"
THINK_START = "<think>"

# One llama.cpp context cannot serve concurrent requests: letting two overlap
# corrupts the KV cache and interleaves tokens between responses. Requests queue
# here instead.
_lock = threading.Lock()

_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK = re.compile(
    r"<function=(?P<name>[\w.:-]+)>(?P<body>.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_BLOCK = re.compile(
    r"<parameter=(?P<key>[\w.:-]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_ARG_PAIR = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatDelta:
    """One piece of a streamed llama.cpp chat completion."""

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


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


def extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Pull Qwen/Hermes tool-call markup out of model text.

    llama-cpp-python's Jinja handler puts tools into the prompt, but it does
    not parse the `<tool_call>` XML Qwen emits — that comes back as ordinary
    `delta.content`. Hermes then sees `tool_calls: null` and `finish_reason:
    stop`. This converts the markup into OpenAI `tool_calls` and returns the
    leftover prose.

    Returns `(tool_calls, remaining_text)`. `tool_calls` is empty when nothing
    parsed, and `remaining_text` is then the original text.
    """
    if not text:
        return [], text

    lowered = text.lower()
    if "<tool_call" not in lowered and "<function=" not in lowered:
        return [], text

    calls: list[dict[str, Any]] = []
    pieces: list[str] = []
    pos = 0
    for match in _TOOL_BLOCK.finditer(text):
        found = _tool_dicts_from_body(match.group(1))
        if not found:
            continue
        pieces.append(text[pos:match.start()])
        calls.extend(found)
        pos = match.end()

    remaining = text
    if calls:
        remaining = "".join(pieces) + text[pos:]
    else:
        found = _tool_dicts_from_function_xml(text)
        if found:
            calls = found
            remaining = _FUNCTION_BLOCK.sub("", text)

    if not calls:
        return [], text

    remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
    return [_openai_tool_call(call) for call in calls], remaining


def complete(
    llm,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    functions: list[dict[str, Any]] | None = None,
) -> Iterator[ChatDelta]:
    """Stream chat deltas, holding the inference lock throughout."""
    params: dict[str, Any] = {
        "messages": messages,
        "temperature": float(config.TEMPERATURE if temperature is None else temperature),
        "top_p": float(config.TOP_P if top_p is None else top_p),
        "top_k": int(config.TOP_K if top_k is None else top_k),
        "max_tokens": int(config.MAX_TOKENS if max_tokens is None else max_tokens),
        "stream": True,
    }
    if tools:
        params["tools"] = tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if functions:
        params["functions"] = functions

    # Note: llama-cpp-python 0.3.35 does not accept `chat_template_kwargs` in
    # create_chat_completion, so template switches such as Qwen's
    # `enable_thinking` cannot be forwarded. Reasoning is always generated and
    # stripped afterwards.
    with _lock:
        for chunk in llm.create_chat_completion(**params):
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            tool_calls = delta.get("tool_calls") or None
            finish_reason = choice.get("finish_reason")
            if not content and not tool_calls and not finish_reason:
                continue
            yield ChatDelta(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )


def collect(
    llm,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    functions: list[dict[str, Any]] | None = None,
    parse_markup: bool = False,
) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Run a completion to the end.

    Returns `(raw_text, tool_calls, finish_reason)`. `tool_calls` comes from
    llama.cpp when it already emits OpenAI deltas, and from `extract_tool_calls`
    when `parse_markup` is set and the model wrote Qwen XML instead.
    """
    parts: list[str] = []
    tool_deltas: list[list[dict[str, Any]]] = []
    finish = "stop"

    for event in complete(
        llm,
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        functions=functions,
    ):
        if event.content:
            parts.append(event.content)
        if event.tool_calls:
            tool_deltas.append(event.tool_calls)
        if event.finish_reason:
            finish = event.finish_reason

    raw = "".join(parts)
    merged = _merge_tool_call_deltas(tool_deltas)
    if merged:
        return raw, merged, "tool_calls"

    if parse_markup:
        parsed, remainder = extract_tool_calls(raw)
        if parsed:
            return remainder, parsed, "tool_calls"

    return raw, None, finish or "stop"


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
    for event in complete(
        llm,
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    ):
        if event.content:
            yield event.content


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


def _tool_dicts_from_body(body: str) -> list[dict[str, str]]:
    return (
        _tool_dicts_from_json(body)
        or _tool_dicts_from_function_xml(body)
        or _tool_dicts_from_arg_keys(body)
    )


def _tool_dicts_from_json(text: str) -> list[dict[str, str]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, str]] = []
    index = 0
    while index < len(text):
        brace = text.find("{", index)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            index = brace + 1
            continue
        found.extend(_tool_dicts_from_value(obj))
        index = end
    return found


def _tool_dicts_from_value(obj: Any) -> list[dict[str, str]]:
    if isinstance(obj, list):
        found: list[dict[str, str]] = []
        for item in obj:
            found.extend(_tool_dicts_from_value(item))
        return found
    if not isinstance(obj, dict):
        return []

    name: Any = obj.get("name")
    args: Any = obj.get("arguments", obj.get("parameters", {}))
    function = obj.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
        args = function.get("arguments", function.get("parameters", args))
    elif isinstance(function, str) and not name:
        name = function

    if not name:
        return []
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return [{"name": str(name), "arguments": args}]


def _tool_dicts_from_function_xml(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for match in _FUNCTION_BLOCK.finditer(text):
        params = {
            key: value
            for key, value in _PARAM_BLOCK.findall(match.group("body"))
        }
        found.append(
            {
                "name": match.group("name"),
                "arguments": json.dumps(params, ensure_ascii=False),
            }
        )
    return found


def _tool_dicts_from_arg_keys(text: str) -> list[dict[str, str]]:
    pairs = _ARG_PAIR.findall(text)
    if not pairs:
        return []
    name = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<"):
            name = stripped
            break
    if not name:
        return []
    return [{"name": name, "arguments": json.dumps(dict(pairs), ensure_ascii=False)}]


def _openai_tool_call(call: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
    }


def _merge_tool_call_deltas(
    groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    slots: dict[int, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            index = int(item.get("index", 0))
            slot = slots.get(index)
            if slot is None:
                slot = {
                    "id": item.get("id") or "",
                    "type": item.get("type") or "function",
                    "function": {"name": "", "arguments": ""},
                }
                slots[index] = slot
            elif item.get("id"):
                slot["id"] = item["id"]
            if item.get("type"):
                slot["type"] = item["type"]
            function = item.get("function") or {}
            name = function.get("name")
            if name:
                slot["function"]["name"] += name
            arguments = function.get("arguments")
            if arguments:
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                slot["function"]["arguments"] += arguments

    calls: list[dict[str, Any]] = []
    for index in sorted(slots):
        slot = slots[index]
        if not slot["function"]["name"]:
            continue
        if not slot["id"]:
            slot["id"] = "call_" + uuid.uuid4().hex[:24]
        calls.append(slot)
    return calls
