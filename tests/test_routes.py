#!/usr/bin/env python3
"""Tests that run without a GPU, using a stub in place of the model.

These cover the behaviour the notebook was verified against: reasoning stripped
from non-streaming responses, raw text preserved in streams, sampling parameters
forwarded, and generation serialized behind the inference lock.

    pip install fastapi httpx
    python tests/test_routes.py

Needs neither llama-cpp-python nor Gradio, so it is safe to run on a laptop.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ggufserve import api, chat, config, installer, model, server, system, webui
from ggufserve.chat import (
    describe_arg_types,
    extract_tool_calls,
    prepare_messages_for_template,
    prepare_tools_for_template,
    split_response,
)
import launch
from launch import _count_steps, _parse_split

REASONING = "Let me think. 17*23 = 391."
ANSWER = "17 x 23 = **391**."
RAW = f"{REASONING}\n</think>\n\n{ANSWER}"

# The closing tag is deliberately torn across two fragments, because that is
# what a real token stream does and it is the case a naive parser gets wrong.
FRAGMENTS = [f"{REASONING}\n</thi", "nk>\n\n17 x 23 ", "= **391**."]


SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

SHELL_XML = (
    '<tool_call>\n'
    '{"name": "shell", "arguments": {"command": "pwd"}}\n'
    "</tool_call>"
)
QWEN_CODER_XML = """I'll search for information about a Cluley AI assistant.

<tool_call>
<function=web_search>
<parameter=query>
Cluley AI assistant tool</parameter>
</function>
</tool_call>"""


class StubLlama:
    """Stands in for llama_cpp.Llama.create_chat_completion(stream=True)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._active = 0
        self.max_concurrent = 0
        self.chunks: list[dict] | None = None
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["stream"] is True, "generation must always stream"

        script = self.chunks
        self.chunks = None

        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)

        def chunks():
            try:
                sequence = script
                if sequence is None:
                    sequence = [
                        {"choices": [{"delta": {"content": fragment}}]}
                        for fragment in FRAGMENTS
                    ]
                for item in sequence:
                    time.sleep(0.02)  # widen the window for a lock race
                    yield item
            finally:
                self._active -= 1

        return chunks()


def _notebook_settings() -> dict:
    """The constants a reader edits at the top of the serve cell.

    Read by executing the assignments only: the rest of the cell shells out to
    launch.py, which is not something a test should do.
    """
    notebook = json.loads(
        (REPO_ROOT / "notebook" / "gguf-serve.ipynb").read_text(encoding="utf-8")
    )
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "KV_CACHE" in "".join(c["source"])
    )

    settings: dict = {}
    for line in cell.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*) = (.+?)(?:\s+#.*)?$", line)
        if match:
            settings[match.group(1)] = ast.literal_eval(match.group(2))
    return settings


def _raises(exception, fn, *args) -> bool:
    try:
        fn(*args)
    except exception:
        return True
    except Exception:
        return False
    return False


def _captured(fn) -> str:
    """Run `fn` and return everything it printed, including from its threads."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


def _sleep_inside_heartbeat(interval: float, duration: float) -> None:
    with system.heartbeat("working", interval=interval):
        time.sleep(duration)


def _step_call_sites() -> int:
    """Count step() calls in the package, ignoring its definition in system.py.

    Ground truth for launch._count_steps: if someone adds a step() somewhere in
    the pipeline without updating the arithmetic, this is what notices.
    """
    total = 0
    for source in sorted((REPO_ROOT / "ggufserve").glob("*.py")):
        for line in source.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s+(system\.)?step\(", line):
                total += 1
    return total


def _with_config(fn, **overrides):
    """Call `fn` with config values temporarily replaced, then restore them."""
    previous = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        return fn()
    finally:
        for key, value in previous.items():
            setattr(config, key, value)


def _with_fake_smi(stdout: str, fn):
    """Run `fn` with nvidia-smi stubbed out, so GPU parsing is testable."""
    real_run, real_which = system.run, system.shutil.which
    try:
        system.shutil.which = lambda name: "/usr/bin/nvidia-smi"
        system.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        return fn()
    finally:
        system.run, system.shutil.which = real_run, real_which


def build_app(llm) -> FastAPI:
    app = FastAPI()

    # gradio.Server claims `/` when it is constructed. Registering it here too
    # is what makes this a real test of webui's route promotion.
    @app.get("/")
    async def main():
        return {"gradio": "placeholder"}

    api.register(app, llm)
    webui.register(app)
    return app


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  pass  {name}")
        else:
            print(f"  FAIL  {name}  {detail}")
            self.failures.append(name)

    def section(self, name: str) -> None:
        print(f"\n{name}")


def main() -> int:
    llm = StubLlama()
    client = TestClient(build_app(llm))
    check = Checker()

    check.section("model identity")
    check(
        "id derived from a GGUF filename",
        config.derive_model_id("Qwen3.8-27B-UD-Q5_K_XL.gguf") == "qwen3.8-27b-ud-q5-k-xl",
        config.derive_model_id("Qwen3.8-27B-UD-Q5_K_XL.gguf"),
    )
    check(
        "works for an unrelated model",
        config.derive_model_id("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
        == "deepseek-r1-distill-qwen-7b-q4-k-m",
        config.derive_model_id("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
    )
    check(
        "an explicit MODEL_ID wins",
        _with_config(MODEL_ID="custom-name", fn=config.model_id) == "custom-name",
    )
    check(
        "MODEL_URL overrides the Hugging Face URL",
        _with_config(MODEL_URL="https://example.com/m.gguf", fn=config.model_url)
        == "https://example.com/m.gguf",
    )

    check.section("status routes")
    response = client.get("/health")
    check("/health returns 200", response.status_code == 200, response.text)
    check(
        "/health names the loaded model",
        response.json()["model"] == config.model_id(),
        response.text,
    )
    check(
        "/v1/models advertises the same id",
        client.get("/v1/models").json()["data"][0]["id"] == config.model_id(),
    )
    check(
        "/v1/models lists exactly one model",
        len(client.get("/v1/models").json()["data"]) == 1,
    )

    check.section("web ui")
    check(
        "/ serves the UI rather than gradio's root",
        "<title>gguf-serve</title>" in client.get("/").text,
    )
    check(
        "/chat serves the UI",
        "<title>gguf-serve</title>" in client.get("/chat").text,
    )
    check(
        "stylesheet has the right content type",
        client.get("/assets/style.css").headers["content-type"].startswith("text/css"),
    )
    check(
        "script has the right content type",
        "text/javascript" in client.get("/assets/app.js").headers["content-type"],
    )
    check("unknown asset is 404", client.get("/assets/nope.css").status_code == 404)

    check.section("request validation")
    response = client.post("/v1/chat/completions", json={"messages": []})
    check("empty messages is rejected", response.status_code == 400, response.text)

    check.section("non-streaming completion")
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    content = body["choices"][0]["message"]["content"]
    check("reasoning is stripped from content", content == ANSWER, repr(content))
    check("no reasoning tag leaks through", "</think>" not in content)
    check("finish_reason is set", body["choices"][0]["finish_reason"] == "stop")
    check("usage is present for clients that read it", "usage" in body)
    check("reasoning_content is absent by default", body["reasoning_content"] is None)

    check.section("non-streaming with enable_thinking")
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
    ).json()
    check(
        "reasoning_content is returned separately",
        body["reasoning_content"] == REASONING,
        repr(body["reasoning_content"]),
    )

    check.section("sampling parameters")
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": 33,
        },
    )
    call = llm.calls[-1]
    check("temperature reaches the model", call["temperature"] == 0.2, str(call))
    check("top_p reaches the model", call["top_p"] == 0.8, str(call))
    check("top_k reaches the model", call["top_k"] == 5, str(call))
    check("max_tokens reaches the model", call["max_tokens"] == 33, str(call))

    check.section("tool calling")
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "pwd"}],
            "tools": [SHELL_TOOL],
            "tool_choice": "auto",
        },
    )
    call = llm.calls[-1]
    check("tools reach the model", call.get("tools") == [SHELL_TOOL], str(call.get("tools")))
    check("tool_choice reaches the model", call.get("tool_choice") == "auto", str(call.get("tool_choice")))

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "pwd"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_shell",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": '{"command": "pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_shell",
                    "content": "/kaggle/working",
                },
            ],
            "tools": [SHELL_TOOL],
        },
    )
    forwarded = (
        llm.calls[-1]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    )
    check(
        "OpenAI argument strings become mappings before llama.cpp",
        forwarded == {"command": "pwd"},
        repr(forwarded),
    )

    def exploding():
        raise TypeError("Can only get item pairs from a mapping.")
        yield {}

    llm.chunks = exploding()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        boom = _with_config(
            CHAT_LOG=True,
            fn=lambda: client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "pwd"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_shell",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": '{"command": "pwd"}',
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": [SHELL_TOOL],
                },
            ),
        )
    logged = buffer.getvalue()
    check("mapping errors return 500 JSON", boom.status_code == 500, boom.text)
    check(
        "mapping errors are printed for Kaggle",
        "item pairs from a mapping" in logged,
        logged[:500],
    )
    check(
        "Jinja hint is printed",
        "Jinja |items" in logged,
        logged[:500],
    )
    check(
        "incoming string arguments are labeled",
        "arguments=str" in logged,
        logged[:500],
    )
    quiet = _captured(
        lambda: client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    check("chat traces are off by default", "chat in :" not in quiet, quiet[:400])

    llm.chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_shell",
                                "type": "function",
                                "function": {"name": "shell", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"command": "pwd"}'}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "pwd"}],
            "tools": [SHELL_TOOL],
        },
    ).json()
    message = body["choices"][0]["message"]
    check(
        "structured tool_calls are returned",
        (message.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "shell",
        repr(message.get("tool_calls")),
    )
    check(
        "arguments stay a JSON string",
        (message.get("tool_calls") or [{}])[0].get("function", {}).get("arguments")
        == '{"command": "pwd"}',
        repr(message.get("tool_calls")),
    )
    check(
        "finish_reason is tool_calls",
        body["choices"][0]["finish_reason"] == "tool_calls",
        body["choices"][0]["finish_reason"],
    )
    check("content is null when only a tool call was produced", message.get("content") is None)

    llm.chunks = [{"choices": [{"delta": {"content": SHELL_XML}}]}]
    body = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "pwd"}],
            "tools": [SHELL_TOOL],
        },
    ).json()
    message = body["choices"][0]["message"]
    parsed_args = (message.get("tool_calls") or [{}])[0].get("function", {}).get("arguments")
    check(
        "Qwen <tool_call> XML is converted to tool_calls",
        (message.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "shell",
        repr(message.get("tool_calls")),
    )
    check(
        "XML arguments are stringified JSON",
        parsed_args == '{"command": "pwd"}',
        repr(parsed_args),
    )
    check(
        "XML conversion sets finish_reason tool_calls",
        body["choices"][0]["finish_reason"] == "tool_calls",
    )

    llm.chunks = [{"choices": [{"delta": {"content": SHELL_XML}}]}]
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "pwd"}]},
    ).json()
    message = body["choices"][0]["message"]
    check(
        "without tools, Qwen XML still becomes tool_calls",
        (message.get("tool_calls") or [{}])[0].get("function", {}).get("name") == "shell",
        repr(message.get("tool_calls")),
    )
    check(
        "without tools, finish_reason is still tool_calls",
        body["choices"][0]["finish_reason"] == "tool_calls",
    )
    check("XML is stripped out of content", "<tool_call>" not in (message.get("content") or ""))

    llm.chunks = [{"choices": [{"delta": {"content": QWEN_CODER_XML}}]}]
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "check cluley"}]},
    ).json()
    message = body["choices"][0]["message"]
    check(
        "nested <function=name> inside <tool_call> parses",
        (message.get("tool_calls") or [{}])[0].get("function", {}).get("name")
        == "web_search",
        repr(message.get("tool_calls")),
    )
    check(
        "nested parameter values survive",
        json.loads(
            (message.get("tool_calls") or [{}])[0]
            .get("function", {})
            .get("arguments")
            or "{}"
        )
        == {"query": "Cluley AI assistant tool"},
    )
    check(
        "preface is kept as content",
        "I'll search" in (message.get("content") or ""),
        repr(message.get("content")),
    )

    llm.chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command": "pwd"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    streamed_calls, stream_finish, stream_done = [], None, False
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "pwd"}],
            "tools": [SHELL_TOOL],
            "stream": True,
        },
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                stream_done = True
                continue
            chunk = json.loads(payload)
            streamed_calls.extend(
                chunk["choices"][0]["delta"].get("tool_calls") or []
            )
            stream_finish = chunk["choices"][0]["finish_reason"] or stream_finish
    check(
        "streaming emits delta.tool_calls",
        any(call.get("function", {}).get("name") == "shell" for call in streamed_calls),
        repr(streamed_calls),
    )
    check("streaming finish_reason is tool_calls", stream_finish == "tool_calls")
    check("tool-call stream ends with [DONE]", stream_done)

    llm.chunks = [{"choices": [{"delta": {"content": QWEN_CODER_XML}}]}]
    streamed_calls, stream_finish, streamed_text = [], None, ""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "check cluley"}], "stream": True},
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            delta = chunk["choices"][0]["delta"]
            streamed_text += delta.get("content") or ""
            streamed_calls.extend(delta.get("tool_calls") or [])
            stream_finish = chunk["choices"][0]["finish_reason"] or stream_finish
    check(
        "streaming without tools still emits delta.tool_calls",
        any(call.get("function", {}).get("name") == "web_search" for call in streamed_calls),
        repr(streamed_calls),
    )
    check("streaming withholds the XML tags from content", "<tool_call>" not in streamed_text)
    check("streaming still keeps the preface", "I'll search" in streamed_text)
    check("streaming nested XML finish_reason is tool_calls", stream_finish == "tool_calls")

    check.section("streaming completion")
    streamed, events, done, finish = "", 0, False, None
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        check("stream returns 200", response.status_code == 200)
        check(
            "content type is event-stream",
            response.headers["content-type"].startswith("text/event-stream"),
        )
        check(
            "tunnel buffering is disabled",
            response.headers.get("x-accel-buffering") == "no",
        )

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                continue
            chunk = json.loads(payload)
            events += 1
            streamed += chunk["choices"][0]["delta"].get("content", "")
            finish = chunk["choices"][0]["finish_reason"] or finish

    check("arrives as several chunks", events >= len(FRAGMENTS), str(events))
    check("raw text is streamed verbatim", streamed == RAW, repr(streamed))
    check("stream keeps the reasoning tags", "</think>" in streamed)
    check("final chunk carries finish_reason", finish == "stop")
    check("stream is terminated with [DONE]", done)

    _, answer = split_response(streamed)
    check("a client can recover the answer", answer == ANSWER, repr(answer))

    check.section("reasoning parsing")
    check(
        "reasoning split off when enabled",
        chat.separate(RAW) == (REASONING, ANSWER),
        repr(chat.separate(RAW)),
    )
    check(
        "output passed through untouched when disabled",
        _with_config(lambda: chat.separate(RAW), PARSE_REASONING=False)
        == ("", RAW.strip()),
    )
    check(
        "a model with no reasoning tags is unaffected",
        chat.separate("Just an answer.") == ("", "Just an answer."),
    )

    parsed, leftover = extract_tool_calls(SHELL_XML)
    check("JSON inside <tool_call> parses", parsed and parsed[0]["function"]["name"] == "shell")
    check("parsed arguments are a JSON string", parsed[0]["function"]["arguments"] == '{"command": "pwd"}')
    check("markup is removed from leftover text", leftover == "")

    doubled = '<tool_call>\n{{"name": "shell", "arguments": {"command": "ls"}}}\n</tool_call>'
    parsed, _ = extract_tool_calls(doubled)
    check(
        "doubled JSON braces still parse",
        parsed and json.loads(parsed[0]["function"]["arguments"]) == {"command": "ls"},
        repr(parsed),
    )

    qwen_xml = (
        "<function=shell>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "</function>"
    )
    parsed, leftover = extract_tool_calls(qwen_xml)
    check(
        "Qwen <function=name> XML parses",
        parsed and parsed[0]["function"]["name"] == "shell",
        repr(parsed),
    )
    check(
        "function XML arguments round-trip",
        parsed and json.loads(parsed[0]["function"]["arguments"]) == {"command": "pwd"},
    )

    parsed, leftover = extract_tool_calls(QWEN_CODER_XML)
    check(
        "coder-style nested XML parses",
        parsed and parsed[0]["function"]["name"] == "web_search",
        repr(parsed),
    )
    check(
        "coder-style leftover is the preface",
        leftover.startswith("I'll search"),
        repr(leftover),
    )

    captured = (
        "I'll search for that term for you. It might be a typo - let me check a few possibilities.\n\n"
        "<tool_call>\n<function=web_search>\n<parameter=query>\n"
        '"cluley" AI assistant</parameter>\n</function>\n</tool_call>\n'
        "<tool_call>\n<function=web_search>\n<parameter=query>\n"
        '"Cluey" AI assistant</parameter>\n</function>\n</tool_call>'
    )
    parsed, leftover = extract_tool_calls(captured)
    check("captured dump yields two web_search calls", len(parsed) == 2, str(len(parsed)))
    check(
        "first query is cluley",
        parsed and json.loads(parsed[0]["function"]["arguments"])["query"]
        == '"cluley" AI assistant',
        repr(parsed[0]["function"]["arguments"] if parsed else None),
    )
    check(
        "second query is Cluey",
        len(parsed) == 2
        and json.loads(parsed[1]["function"]["arguments"])["query"]
        == '"Cluey" AI assistant',
    )
    check("captured dump XML is stripped", "<tool_call>" not in leftover)
    none, original = extract_tool_calls("I will inspect the repository.")
    check("prose without markup is left alone", none == [] and original == "I will inspect the repository.")

    prepared = prepare_messages_for_template(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            }
        ]
    )
    check(
        "string arguments become a dict for Jinja |items",
        prepared[0]["tool_calls"][0]["function"]["arguments"] == {"query": "x"},
        repr(prepared[0]["tool_calls"][0]["function"]["arguments"]),
    )
    check(
        "empty argument string becomes {}",
        prepare_messages_for_template(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "shell", "arguments": ""}}
                    ],
                }
            ]
        )[0]["tool_calls"][0]["function"]["arguments"]
        == {},
    )
    already = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "shell", "arguments": {"command": "ls"}}}
            ],
        }
    ]
    check(
        "dict arguments pass through",
        prepare_messages_for_template(already)[0]["tool_calls"][0]["function"][
            "arguments"
        ]
        == {"command": "ls"},
    )
    string_tools = prepare_tools_for_template(
        [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": '{"type": "object"}',
                },
            }
        ]
    )
    check(
        "tool parameters JSON string becomes a mapping",
        string_tools[0]["function"]["parameters"] == {"type": "object"},
        repr(string_tools[0]["function"]["parameters"]),
    )
    check(
        "string arguments are labeled str",
        describe_arg_types(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "x"}',
                            }
                        }
                    ],
                }
            ]
        )
        == ["msg[0].assistant.web_search.arguments=str"],
    )

    check.section("model file validation")
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.gguf"
        good.write_bytes(b"GGUF" + b"\0" * 1020)

        check("a well-formed file passes", model.validate(good)[0])
        check(
            "exact expected size passes",
            model.validate(good, expected_bytes=1024)[0],
        )
        check(
            "a short file is rejected as incomplete",
            model.validate(good, expected_bytes=999_999)[0] is False,
        )

        wrong = Path(tmp) / "wrong.gguf"
        wrong.write_bytes(b"NOPE" + b"\0" * 1020)
        check("bad magic bytes are rejected", model.validate(wrong)[0] is False)

        empty = Path(tmp) / "empty.gguf"
        empty.write_bytes(b"")
        check("an empty file is rejected", model.validate(empty)[0] is False)

        check(
            "a missing file is rejected",
            model.validate(Path(tmp) / "nope.gguf")[0] is False,
        )

        dataset = Path(tmp) / "input" / "my-gguf"
        dataset.mkdir(parents=True)
        attached = dataset / "kept.gguf"
        attached.write_bytes(b"GGUF" + b"\0" * 1020)
        check(
            "an attached Kaggle dataset copy is found",
            model.find_attached_model("kept.gguf", root=Path(tmp) / "input") == attached,
        )
        check(
            "a missing attached copy is None",
            model.find_attached_model("nope.gguf", root=Path(tmp) / "input") is None,
        )
        check(
            "a dataset slug pins the search",
            model.find_attached_model(
                "kept.gguf", root=Path(tmp) / "input", dataset="my-gguf"
            )
            == attached,
        )
        renamed = dataset / "other-name.gguf"
        attached.replace(renamed)
        check(
            "a named dataset with one GGUF is used even if the filename differs",
            model.find_attached_model(
                "kept.gguf", root=Path(tmp) / "input", dataset="my-gguf"
            )
            == renamed,
        )
        check(
            "a normal model path is not redirected",
            model._download_dest(Path(tmp) / "model.gguf") == Path(tmp) / "model.gguf",
        )

    check.section("wheel cache")
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "wheels"
        previous = os.environ.get("GGUF_SERVE_WHEEL_DIR")
        os.environ["GGUF_SERVE_WHEEL_DIR"] = str(cache)
        try:
            check(
                "GGUF_SERVE_WHEEL_DIR selects the cache directory",
                installer.persistent_cache_dir() == cache,
            )
        finally:
            if previous is None:
                os.environ.pop("GGUF_SERVE_WHEEL_DIR", None)
            else:
                os.environ["GGUF_SERVE_WHEEL_DIR"] = previous

    check.section("kv cache type")
    check("f16 and q8_0 map to ggml type ids", model.KV_CACHE_TYPES == {"f16": 1, "q8_0": 8})

    # Must be rejected before the llama_cpp import, so a bad config value gives
    # a clear message rather than an ImportError on a machine without CUDA.
    try:
        _with_config(
            lambda: model.load(Path("/nonexistent.gguf")),
            KV_CACHE_TYPE="bogus",
        )
        check("a bad KV_CACHE_TYPE is rejected", False, "no error raised")
    except SystemExit as error:
        check("a bad KV_CACHE_TYPE is rejected early", "bogus" in str(error), str(error))
    except ImportError as error:
        check(
            "a bad KV_CACHE_TYPE is rejected early",
            False,
            f"reached llama_cpp import first: {error}",
        )

    check.section("host stats")
    # nvidia-smi output as it really looks with --noheader --nounits, including
    # the [N/A] readings some cards and VMs return.
    smi = (
        "0, Tesla T4, 10379, 15360, 2, 49\n"
        "1, Tesla T4, 11597, 15360, [N/A], [N/A]\n"
        "garbage line\n"
    )
    gpus = _with_fake_smi(smi, system.gpu_stats)
    check("both GPUs parsed, junk skipped", len(gpus) == 2, str(len(gpus)))
    check("memory converted to GiB", round(gpus[0].used_gib, 2) == 10.14, str(gpus[0]))
    check("utilisation parsed", gpus[0].util_pct == 2)
    check("[N/A] becomes None, not a crash", gpus[1].util_pct is None)
    check(
        "description reads sensibly",
        gpus[0].describe() == "Tesla T4  10.1 / 15.0 GiB (68%)  util 2%  49C",
        gpus[0].describe(),
    )
    check("no GPU means an empty list", _with_fake_smi("", system.gpu_stats) == [])

    with tempfile.TemporaryDirectory() as tmp:
        meminfo = Path(tmp) / "meminfo"
        meminfo.write_text(
            "MemTotal:       32873252 kB\n"
            "MemFree:        27000000 kB\n"
            "MemAvailable:   30000000 kB\n"
            "Buffers:          100000 kB\n"
        )
        ram = system.ram_stats(meminfo)
        check("total RAM parsed", ram and round(ram[1], 1) == 31.4, str(ram))
        check(
            "used derived from MemAvailable",
            ram and round(ram[0], 1) == 2.7,
            str(ram),
        )
        check(
            "a missing meminfo returns None",
            system.ram_stats(Path(tmp) / "nope") is None,
        )

    check("large sizes read as GiB", system.human_size(20876938144) == "19.44 GiB")
    check("small sizes drop to MiB", system.human_size(2048) == "0 MiB")

    check.section("share url extraction")
    # Gradio returns (app, local_url, share_url) inside a TupleNoPrint.
    check(
        "public URL pulled from the launch result",
        server._share_url((object(), "http://0.0.0.0:7860/", "https://abc.gradio.live"))
        == "https://abc.gradio.live",
    )
    check(
        "None when sharing is off",
        server._share_url((object(), "http://0.0.0.0:7860/", None)) is None,
    )
    for odd in (None, (), (object(), "x"), (object(), "x", 123), "nope"):
        if server._share_url(odd) is not None:
            check(f"unexpected shape {odd!r} handled", False)
            break
    else:
        check("unexpected shapes degrade to None instead of raising", True)

    check.section("inference lock")

    def hit() -> None:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    check(
        "only one generation runs at a time",
        llm.max_concurrent == 1,
        f"peak concurrency was {llm.max_concurrent}",
    )

    llm.resets = 0
    gen = chat.complete(llm, [{"role": "user", "content": "hi"}])
    next(gen)
    gen.close()
    check("closing a stream resets llama.cpp", llm.resets == 1, str(llm.resets))
    check("closing a stream stops llama.cpp", llm._active == 0, str(llm._active))
    check("closing a stream releases the lock", chat._lock.acquire(blocking=False))
    chat._lock.release()
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    check(
        "a request after cancel still completes",
        body["choices"][0]["message"]["content"] == ANSWER,
        repr(body["choices"][0]["message"].get("content")),
    )

    llm.resets = 0
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        for line in response.iter_lines():
            if line:
                break
    check("http cancel stops llama.cpp", llm._active == 0, str(llm._active))
    check("http cancel releases the lock", chat._lock.acquire(blocking=False))
    chat._lock.release()
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    check(
        "a request after http cancel still completes",
        body["choices"][0]["message"]["content"] == ANSWER,
        repr(body["choices"][0]["message"].get("content")),
    )

    check.section("model source parsing")
    HF = "https://huggingface.co/unsloth/Qwen3-8B-GGUF"

    # The address bar of a file page is what people actually paste, and Hugging
    # Face serves /blob/ as HTML with status 200 — so getting this wrong means
    # downloading a web page and failing on GGUF magic bytes.
    blob = config.parse_source(f"{HF}/blob/main/Qwen3-8B-Q4_K_M.gguf")
    check("a file page URL yields the repo", blob.repo == "unsloth/Qwen3-8B-GGUF")
    check("a file page URL yields the filename", blob.filename == "Qwen3-8B-Q4_K_M.gguf")
    check("/blob/ is rewritten to /resolve/", "/resolve/main/" in blob.url)
    check("and never left as /blob/", "/blob/" not in blob.url)

    direct = config.parse_source(f"{HF}/resolve/main/Qwen3-8B-Q4_K_M.gguf")
    check("a resolve URL survives intact", direct.url.startswith(f"{HF}/resolve/main/"))

    nested = config.parse_source(f"{HF}/blob/main/BF16/split.gguf")
    check("a subdirectory keeps the full remote path", "/main/BF16/split.gguf" in nested.url)
    check("but stores only the basename locally", nested.filename == "split.gguf")

    for label, text in (
        ("a repo page URL", HF),
        ("a tree URL", f"{HF}/tree/main"),
        ("a bare owner/repo id", "unsloth/Qwen3-8B-GGUF"),
    ):
        parsed = config.parse_source(text)
        check(f"{label} gives a repo and no file", parsed.repo and not parsed.filename)

    other = config.parse_source("https://example.com/models/custom.gguf")
    check("a non-HF URL is used verbatim", other.url == "https://example.com/models/custom.gguf")
    check("with its filename taken from the path", other.filename == "custom.gguf")

    plain = config.parse_source("Qwen3-8B-Q4_K_M.gguf")
    check("a bare filename is just a filename", plain.filename and not plain.repo)

    for bad in ("", "   ", "not a url", "///"):
        check(f"{bad!r} is rejected", _raises(ValueError, config.parse_source, bad))

    # A repo names no file, and guessing one wrong wastes a 20 GiB download, so
    # this has to be a refusal rather than a default.
    args = launch.parse_args(["--model", "unsloth/Qwen3-8B-GGUF"])
    check(
        "a bare repo is refused, not guessed at",
        _raises(SystemExit, launch._apply_model_arg, args),
    )

    check.section("notebook settings mirror config.py")
    # The cell presents these as "the shipped defaults", so they have to be the
    # shipped defaults. Left unchecked they would quietly become a second,
    # stale source of truth that silently overrides the first.
    settings = _notebook_settings()
    expected = {
        "MODEL_REPO": config.MODEL_REPO,
        "MODEL_FILE": config.MODEL_FILE,
        "MODEL_DIR": str(config.MODEL_DIR),
        "CTX": config.CTX_SIZE,
        "KV_CACHE": config.KV_CACHE_TYPE,
        "GPU_LAYERS": config.N_GPU_LAYERS,
        "PORT": config.SERVER_PORT,
        "SHARE": config.SHARE,
        "PARSE_REASONING": config.PARSE_REASONING,
        "VERBOSE": config.VERBOSE,
        "CHAT_LOG": config.CHAT_LOG,
    }
    check("every setting was found in the cell", set(expected) <= set(settings))
    for name, value in expected.items():
        check(f"{name} matches config.py", settings.get(name) == value)

    check(
        "MODEL_URL is blank so the repo and file above apply",
        settings.get("MODEL_URL") == "",
    )
    check(
        "TENSOR_SPLIT matches config.py once parsed",
        _parse_split(settings.get("TENSOR_SPLIT", "")) == config.TENSOR_SPLIT,
    )

    check.section("tensor split parsing")
    check("a matched pair", _parse_split("1,1") == [1.0, 1.0])
    check("a weighted pair", _parse_split("1,2") == [1.0, 2.0])
    check("whitespace is tolerated", _parse_split(" 1.0 , 1.0 ") == [1.0, 1.0])
    check("a single GPU is spelled none", _parse_split("none") is None)
    check("as is an empty value", _parse_split("") is None)
    for bad in ("1,x", "0,1", "-1,2", "abc"):
        check(f"{bad!r} is rejected", _raises(SystemExit, _parse_split, bad))

    check.section("progress reporting")
    system.set_total_steps(3)
    labels = _captured(lambda: [system.step("one"), system.step("two")])
    check("steps are numbered against the total", "[1/3] one" in labels)
    check("the counter advances", "[2/3] two" in labels)

    system.set_total_steps(0)
    check(
        "an unset total falls back to a plain marker",
        ">> solo" in _captured(lambda: system.step("solo")),
    )

    threads_before = threading.active_count()
    ticks = _captured(lambda: _sleep_inside_heartbeat(0.05, 0.16))
    check("a long wait reports elapsed time", "elapsed" in ticks)
    check("the duration is printed on the way out", "took" in ticks)
    check("the ticker thread is cleaned up", threading.active_count() == threads_before)

    # The [n/total] labels are only honest while this arithmetic tracks the
    # step() calls in the pipeline, and nothing else would catch them drifting.
    full = SimpleNamespace(skip_install=False, download_only=False, skip_smoke_test=False)
    check(
        "a full run's total matches the step() calls in the package",
        _count_steps(full) == _step_call_sites(),
    )
    for flag, expected in (
        ("skip_install", _count_steps(full) - 2),
        ("skip_smoke_test", _count_steps(full) - 1),
        ("download_only", _count_steps(full) - 4),
    ):
        args = SimpleNamespace(**{**vars(full), flag: True})
        check(f"--{flag.replace('_', '-')} shortens the total", _count_steps(args) == expected)

    check.section("openapi schema")
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/health", "/v1/models", "/v1/chat/completions"):
        check(f"{path} is documented", path in paths)
    check("UI routes stay out of the schema", "/chat" not in paths)

    print()
    if check.failures:
        print(f"{len(check.failures)} failed: {', '.join(check.failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
