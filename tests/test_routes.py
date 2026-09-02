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

import contextlib
import io
import json
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

from ggufserve import api, chat, config, model, server, system, webui
from ggufserve.chat import split_response
from launch import _count_steps

REASONING = "Let me think. 17*23 = 391."
ANSWER = "17 x 23 = **391**."
RAW = f"{REASONING}\n</think>\n\n{ANSWER}"

# The closing tag is deliberately torn across two fragments, because that is
# what a real token stream does and it is the case a naive parser gets wrong.
FRAGMENTS = [f"{REASONING}\n</thi", "nk>\n\n17 x 23 ", "= **391**."]


class StubLlama:
    """Stands in for llama_cpp.Llama.create_chat_completion(stream=True)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._active = 0
        self.max_concurrent = 0

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["stream"] is True, "generation must always stream"

        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)

        def chunks():
            try:
                for fragment in FRAGMENTS:
                    time.sleep(0.02)  # widen the window for a lock race
                    yield {"choices": [{"delta": {"content": fragment}}]}
            finally:
                self._active -= 1

        return chunks()


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
