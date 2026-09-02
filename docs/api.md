# API reference

The server is OpenAI-compatible. Point any OpenAI client at the server origin plus `/v1`, and use the model id that `GET /health` reports.

There is no authentication. `api_key` is required by most client libraries but ignored here, so any non-empty string works.

Interactive docs are at `/docs`, generated from the live routes.

## Model ids

The id is derived from the GGUF filename: `Qwen3.8-27B-UD-Q5_K_XL.gguf` becomes `qwen3.8-27b-ud-q5-k-xl`. Set `MODEL_ID` in `ggufserve/config.py` to override it.

The `model` field in a request is accepted but not enforced — only one model is loaded, so every request goes to it regardless. That makes it easy to drop this server in behind a client that has a model name hardcoded.

## Reasoning output

Reasoning models (Qwen3, DeepSeek-R1, QwQ) think first, close the section with `</think>`, then answer. Raw output looks like this:

```
We need 17 times 23. 17*(20+3) = 340+51 = 391.
</think>

17 × 23 = **391**.
```

The opening `<think>` is frequently absent, so gguf-serve keys on the closing tag. Ordinary models never emit it and their output passes through untouched.

How this reaches you depends on the mode:

| Mode | What you receive |
| --- | --- |
| Non-streaming | Reasoning stripped. `content` is the answer only. |
| Streaming | Raw text, reasoning included. Split it client-side on `</think>`. |
| Streaming with `enable_thinking` | Reasoning stripped, but nothing is emitted until generation finishes. |

Streaming defaults to raw output because stripping it server-side means buffering the whole response — you lose token-by-token output, which is the reason to stream in the first place. The bundled web UI splits the stream as it arrives; [`web/app.js`](../web/app.js) has a working implementation worth copying.

To opt into server-side stripping, send the vLLM-style flag:

```json
{
  "extra_body": {
    "chat_template_kwargs": { "enable_thinking": true }
  }
}
```

Non-streaming responses then also carry the reasoning separately in a non-standard `reasoning_content` field.

Two caveats:

- **Do not feed reasoning back into the next turn.** Send only final answers in `messages`, or the model conditions on its own scratchpad and quality degrades.
- Server-side stripping can be disabled entirely with `--no-reasoning`, in which case every mode returns raw text.

## `POST /v1/chat/completions`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `messages` | array | required | Standard OpenAI role/content objects |
| `model` | string | — | Accepted but not enforced |
| `temperature` | float | `1.0` | |
| `top_p` | float | `0.95` | |
| `top_k` | int | `20` | Not in the OpenAI spec; llama.cpp supports it |
| `max_tokens` | int | `2048` | |
| `stream` | bool | `false` | Server-sent events when true |
| `extra_body` | object | `null` | Carries `chat_template_kwargs.enable_thinking` |

Defaults come from `ggufserve/config.py`, so they follow whatever you set there.

Streaming responses are `text/event-stream`, one `data:` line per chunk, terminated by `data: [DONE]`. Errors mid-stream arrive as a chunk containing an `error` object rather than a dropped connection, because the HTTP status has already been sent by then.

Token usage is reported as zeros. Streaming llama.cpp does not return counts, and inventing estimates would be worse than an obvious placeholder. The field is present rather than omitted because clients that read `response.usage.total_tokens` crash on a missing one.

### Examples

Non-streaming:

```bash
curl https://YOUR-ID.gradio.live/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 17 x 23?"}],
    "max_tokens": 64
  }'
```

Streaming, with the reasoning split off client-side:

```python
from openai import OpenAI

client = OpenAI(base_url="https://YOUR-ID.gradio.live/v1", api_key="not-used")

stream = client.chat.completions.create(
    model="qwen3.8-27b-ud-q5-k-xl",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)

raw = ""
for chunk in stream:
    raw += chunk.choices[0].delta.content or ""

answer = raw.split("</think>", 1)[-1].strip()
print(answer)
```

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "qwen3.8-27b-ud-q5-k-xl",
    "object": "model",
    "created": 1787860707,
    "owned_by": "gguf-serve"
  }]
}
```

## `GET /health`

Confirms the server is up and reports what it actually loaded — the quickest way to check that a config change took effect.

```json
{
  "status": "ok",
  "model": "qwen3.8-27b-ud-q5-k-xl",
  "model_file": "Qwen3.8-27B-UD-Q5_K_XL.gguf",
  "context": 16384,
  "parse_reasoning": true
}
```

## Concurrency

Requests are serialized behind one lock. A single llama.cpp context cannot serve overlapping requests — doing so corrupts the KV cache and interleaves tokens between responses — so a second request waits rather than failing. Set client timeouts accordingly: a queued request can sit for as long as the one ahead of it takes.
