"""Generation, tool-call parsing, and reasoning-tag handling."""

from __future__ import annotations

import json
import re
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from ggufserve import config
from ggufserve.system import heartbeat, info, ok, step, warn

THINK_END = "</think>"
THINK_START = "<think>"

# One llama.cpp context cannot serve concurrent requests: letting two overlap
# corrupts the KV cache and interleaves tokens between responses. Requests queue
# here instead.
_lock = threading.Lock()

_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_TOOL_MARK = re.compile(r"<tool_call|<function=", re.IGNORECASE)
# Hold back a short suffix while streaming so a tag split across tokens is not
# flushed as ordinary text before we can see it is a tool call.
_HOLD = 16
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


def _as_mapping(value: Any) -> Any:
    """JSON-decode OpenAI tool argument strings so Qwen's template can |items them.

    Qwen3.8's chat template does `tool_call.arguments|items`. That needs a
    mapping. Hermes/OpenAI send `function.arguments` as a JSON string, and
    llama-cpp-python renders the GGUF template as-is — which raises
    `Can only get item pairs from a mapping.`
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, dict) else value
    return value


def prepare_tools_for_template(tools: Any) -> Any:
    """Make sure `tools` is a list of dicts with mapping `parameters`."""
    if not tools:
        return tools
    if isinstance(tools, str):
        tools = _as_mapping(tools)
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        return tools
    prepared = []
    for tool in tools:
        if isinstance(tool, str):
            tool = _as_mapping(tool)
        if not isinstance(tool, dict):
            continue
        tool = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            function = dict(function)
            parameters = function.get("parameters")
            mapped = _as_mapping(parameters) if parameters is not None else parameters
            if isinstance(mapped, dict):
                function["parameters"] = mapped
            tool["function"] = function
        prepared.append(tool)
    return prepared


def prepare_messages_for_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn OpenAI JSON-string tool arguments into mappings for the chat template."""
    prepared: list[dict[str, Any]] = []
    for original in messages:
        message = dict(original)
        calls = message.get("tool_calls")
        if not calls:
            prepared.append(message)
            continue
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except json.JSONDecodeError:
                prepared.append(message)
                continue
        if not isinstance(calls, list):
            prepared.append(message)
            continue
        rewritten = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            call = dict(call)
            function = call.get("function")
            if isinstance(function, dict):
                function = dict(function)
                mapped = _as_mapping(function.get("arguments"))
                function["arguments"] = mapped if isinstance(mapped, dict) else {}
                call["function"] = function
            elif "arguments" in call:
                mapped = _as_mapping(call.get("arguments"))
                call["arguments"] = mapped if isinstance(mapped, dict) else {}
            rewritten.append(call)
        message["tool_calls"] = rewritten
        prepared.append(message)
    return prepared


def _kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def describe_arg_types(messages: list[dict[str, Any]]) -> list[str]:
    """Compact `msg[i].name.arguments=str|dict` labels for Kaggle logs."""
    labels: list[str] = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if not calls:
            continue
        if not isinstance(calls, list):
            labels.append(f"msg[{index}].tool_calls={_kind(calls)}")
            continue
        role = message.get("role") or "?"
        for call in calls:
            if not isinstance(call, dict):
                labels.append(f"msg[{index}].tool_calls={_kind(call)}")
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name") or "?"
                labels.append(
                    f"msg[{index}].{role}.{name}.arguments="
                    f"{_kind(function.get('arguments'))}"
                )
            else:
                labels.append(
                    f"msg[{index}].{role}.arguments={_kind(call.get('arguments'))}"
                )
    return labels


def _tool_names(tools: Any) -> list[str]:
    if not tools:
        return []
    if isinstance(tools, dict):
        tools = [tools]
    names: list[str] = []
    if not isinstance(tools, list):
        return [f"tools={_kind(tools)}"]
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                names.append(str(function.get("name") or "?"))
            else:
                names.append(str(tool.get("name") or "?"))
        else:
            names.append(_kind(tool))
    return names


def _role_labels(messages: list[dict[str, Any]]) -> str:
    labels = []
    for message in messages:
        role = str(message.get("role") or "?")
        if message.get("tool_calls"):
            role += "+tools"
        labels.append(role)
    return ",".join(labels)


def _preview(text: str, limit: int = 160) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _call_names(calls: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for call in calls or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            names.append(str(function.get("name") or "?"))
        elif isinstance(call, dict):
            names.append(str(call.get("name") or "?"))
    return names


def _chat_info(text: str) -> None:
    if config.CHAT_LOG:
        info(text)


def _chat_warn(text: str) -> None:
    if config.CHAT_LOG:
        warn(text)


def close_iterator(iterator: Any) -> None:
    """Close a generator so its `finally` runs (lock release, llama.cpp reset)."""
    if iterator is None:
        return
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


def _reset_llama(llm) -> None:
    reset = getattr(llm, "reset", None)
    if not callable(reset):
        return
    try:
        reset()
    except Exception as error:
        _chat_warn(f"llama.cpp reset failed: {type(error).__name__}: {error}")


def _log_llama_error(error: BaseException, messages: list[dict[str, Any]]) -> None:
    if not config.CHAT_LOG:
        return
    warn(f"llama.cpp failed: {type(error).__name__}: {error}")
    leftover = [item for item in describe_arg_types(messages) if item.endswith("=str")]
    if leftover:
        warn(f"template still has string arguments: {', '.join(leftover)}")
    text = str(error).lower()
    if "item pairs" in text or "mapping" in text:
        warn(
            "Jinja |items needs dict arguments, not JSON strings "
            "(Qwen chat template vs OpenAI tool_calls)"
        )
    traceback.print_exc()


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
    prepared_messages = prepare_messages_for_template(messages)
    prepared_tools = prepare_tools_for_template(tools) if tools else tools
    if config.CHAT_LOG:
        incoming = describe_arg_types(messages)
        outgoing = describe_arg_types(prepared_messages)
        _chat_info(
            f"chat in : {len(messages)} msgs [{_role_labels(messages)}] "
            f"tools={_tool_names(tools) or '-'} choice={tool_choice!r}"
        )
        if incoming or outgoing:
            _chat_info(
                f"chat args in : {incoming or '-'} -> template {outgoing or '-'}"
            )
            leftover = [item for item in outgoing if item.endswith("=str")]
            if leftover:
                _chat_warn(
                    f"string arguments still present; Jinja |items will fail: {leftover}"
                )

    params: dict[str, Any] = {
        "messages": prepared_messages,
        "temperature": float(config.TEMPERATURE if temperature is None else temperature),
        "top_p": float(config.TOP_P if top_p is None else top_p),
        "top_k": int(config.TOP_K if top_k is None else top_k),
        "max_tokens": int(config.MAX_TOKENS if max_tokens is None else max_tokens),
        "stream": True,
    }
    if prepared_tools:
        params["tools"] = prepared_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if functions:
        params["functions"] = functions

    # Note: llama-cpp-python 0.3.35 does not accept `chat_template_kwargs` in
    # create_chat_completion, so template switches such as Qwen's
    # `enable_thinking` cannot be forwarded. Reasoning is always generated and
    # stripped afterwards.
    with _lock:
        stream = None
        finished = False
        try:
            stream = llm.create_chat_completion(**params)
            for chunk in stream:
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
            finished = True
        except Exception as error:
            _log_llama_error(error, prepared_messages)
            raise
        finally:
            close_iterator(stream)
            if not finished:
                _chat_warn("chat aborted; resetting llama.cpp context")
                _reset_llama(llm)


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
    parse_markup: bool = True,
) -> tuple[str, list[dict[str, Any]] | None, str]:
    """Run a completion to the end.

    Returns `(raw_text, tool_calls, finish_reason)`. `tool_calls` comes from
    llama.cpp when it already emits OpenAI deltas, and from `extract_tool_calls`
    when the model wrote Qwen XML into `content` instead. Markup is parsed even
    if the client omitted `tools`, because coder GGUFs emit it anyway.
    """
    parts: list[str] = []
    tool_deltas: list[list[dict[str, Any]]] = []
    finish = "stop"

    stream = complete(
        llm,
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        functions=functions,
    )
    try:
        for event in stream:
            if event.content:
                parts.append(event.content)
            if event.tool_calls:
                tool_deltas.append(event.tool_calls)
            if event.finish_reason:
                finish = event.finish_reason
    finally:
        close_iterator(stream)

    raw = "".join(parts)
    merged = _merge_tool_call_deltas(tool_deltas)
    if merged:
        _chat_info(
            f"chat out: finish=tool_calls names={_call_names(merged)} "
            f"via=openai-delta content={_preview(raw)!r}"
        )
        return raw, merged, "tool_calls"

    if parse_markup:
        parsed, remainder = extract_tool_calls(raw)
        if parsed:
            _chat_info(
                f"chat out: finish=tool_calls names={_call_names(parsed)} "
                f"via=qwen-xml content={_preview(remainder)!r}"
            )
            return remainder, parsed, "tool_calls"

    _chat_info(
        f"chat out: finish={finish or 'stop'} names=[] via=text "
        f"content={_preview(raw)!r}"
    )
    return raw, None, finish or "stop"


def iter_client_events(
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
    """Stream content, withholding Qwen tool-call markup for the final delta.

    Tokens before `<tool_call` / `<function=` are forwarded live. The markup
    itself is rewritten as OpenAI `tool_calls` so clients never see the tags
    as a finished answer.
    """
    raw = ""
    emitted = 0
    tool_deltas: list[list[dict[str, Any]]] = []
    finish = "stop"

    def take(up_to: int) -> str:
        nonlocal emitted
        if up_to <= emitted:
            return ""
        piece = raw[emitted:up_to]
        emitted = up_to
        return piece

    stream = complete(
        llm,
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        functions=functions,
    )
    try:
        for event in stream:
            if event.tool_calls:
                tool_deltas.append(event.tool_calls)
            if event.finish_reason:
                finish = event.finish_reason
            if not event.content:
                continue
            raw += event.content
            mark = _TOOL_MARK.search(raw)
            if mark:
                piece = take(mark.start())
            else:
                piece = take(max(emitted, len(raw) - _HOLD))
            if piece:
                yield ChatDelta(content=piece)
    finally:
        close_iterator(stream)

    merged = _merge_tool_call_deltas(tool_deltas)
    parsed, _remainder = extract_tool_calls(raw)
    calls = merged or parsed
    if calls:
        via = "openai-delta" if merged else "qwen-xml"
        _chat_info(
            f"chat out: finish=tool_calls names={_call_names(calls)} "
            f"via={via} content={_preview(raw)!r}"
        )
        yield ChatDelta(tool_calls=calls, finish_reason="tool_calls")
        return
    leftover = take(len(raw))
    if leftover:
        yield ChatDelta(content=leftover)
    _chat_info(
        f"chat out: finish={finish or 'stop'} names=[] via=text "
        f"content={_preview(raw)!r}"
    )
    yield ChatDelta(finish_reason=finish or "stop")


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
    stream = complete(
        llm,
        messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    try:
        for event in stream:
            if event.content:
                yield event.content
    finally:
        close_iterator(stream)


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
