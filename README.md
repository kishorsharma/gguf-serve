# gguf-serve

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kishorsharma/gguf-serve/blob/main/notebook/gguf-serve.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishorsharma/gguf-serve/blob/main/notebook/gguf-serve.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Turn any GGUF model into a **public, OpenAI-compatible API** with a browser chat UI — from one command, on a free notebook GPU.

```
git clone https://github.com/kishorsharma/gguf-serve.git
cd gguf-serve
python launch.py
```

That is the whole setup. It installs a CUDA build of llama.cpp that actually uses the GPU, downloads and verifies the model, splits it across however many GPUs you have, and serves everything on one public URL. Re-running skips whatever is already done.

When it is ready you get everything you need in one block:

```
======================================================================
  gguf-serve is running
======================================================================

  PUBLIC URL   https://fffdf9bc9bf4c2fc71.gradio.live

  chat UI      https://fffdf9bc9bf4c2fc71.gradio.live/
  API base     https://fffdf9bc9bf4c2fc71.gradio.live/v1
  API docs     https://fffdf9bc9bf4c2fc71.gradio.live/docs
  local        http://127.0.0.1:7860/

  model        qwen3.8-27b-ud-q5-k-xl
  file         Qwen3.8-27B-UD-Q5_K_XL.gguf  (19.44 GiB)
  context      16,384 tokens
  reasoning    </think> sections split off

  GPU 0        Tesla T4  10.1 / 15.0 GiB (68%)  util 2%  49C
  GPU 1        Tesla T4  11.3 / 15.0 GiB (76%)  util 1%  52C
  RAM          8.2 / 31.0 GiB (26%)

  Point any OpenAI client at:
    base_url = "https://fffdf9bc9bf4c2fc71.gradio.live/v1"
    model    = "qwen3.8-27b-ud-q5-k-xl"
    api_key  = "not-used"
======================================================================
```

The GPU line is worth a glance: if VRAM use is far below your card's capacity, llama.cpp quietly kept layers on the CPU and generation will be slow.

Anything [llama.cpp](https://github.com/ggerganov/llama.cpp) can load works — Qwen, Llama, Mistral, Gemma, DeepSeek-R1, Phi. The only real constraint is that the model plus its KV cache fits your VRAM. The default is Qwen3.8-27B at Q5\_K\_XL, verified on Kaggle's free 2 × Tesla T4.

## What you get

One server, one origin, six endpoints:

| Path | What it is |
| --- | --- |
| `/` | Browser chat UI with streaming and live reasoning |
| `/docs` | Interactive OpenAPI reference |
| `/v1/models` | OpenAI model listing |
| `/v1/chat/completions` | OpenAI chat completions, streaming and non-streaming |
| `/health` | Health check reporting what is actually loaded |
| `/openapi.json` | OpenAPI schema |

All of them are also reachable on a public `https://….gradio.live` URL, so a client on your laptop can talk to a model running in a notebook.

## Serving a different model

Two flags, no editing:

```
python launch.py \
  --model-repo unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  --model-file DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
```

Everything else follows. The API model id is derived from the filename (`deepseek-r1-distill-qwen-7b-q4-k-m`), the download size is read from the server so nothing needs to be looked up, and the chat UI labels itself from `/health`.

For a permanent change, edit [`ggufserve/config.py`](ggufserve/config.py) instead. Single-GPU setups and mismatched GPU pairs are covered in [docs/configuration.md](docs/configuration.md).

## Running it

### Kaggle

Open [`notebook/gguf-serve.ipynb`](notebook/gguf-serve.ipynb) and run its two cells. Set the accelerator to **GPU T4 x2** first (*Settings → Accelerator*) — the default model needs about 23 GiB of VRAM, which is two T4s.

Leave the second cell running. The server lives inside it, so stopping the cell takes the public URL down with it.

### Colab

The same notebook works, with one caveat: **free Colab gives you a single 16 GB T4**, and the default model does not fit. Either use an L4 or A100 runtime, or pick a model that fits one card:

```
python launch.py --model-file Qwen3.8-27B-UD-Q3_K_XL.gguf
```

### Your own machine

```
python launch.py --no-share
```

`--no-share` skips the public tunnel and serves on `http://127.0.0.1:7860` only.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--model-file` | `Qwen3.8-27B-UD-Q5_K_XL.gguf` | The `.gguf` to serve |
| `--model-repo` | `unsloth/Qwen3.8-27B-GGUF` | Hugging Face repo holding it |
| `--model-url` | — | Download from here instead of Hugging Face |
| `--model-dir` | `/tmp/gguf-serve/models` | Where the `.gguf` is stored |
| `--ctx` | `16384` | Context window; lower it if the model will not fit |
| `--gpu-layers` | `-1` | Layers on the GPU; `-1` means all |
| `--port` | `7860` | Falls forward to the next free port if taken |
| `--no-share` | off | Local access only, no public URL |
| `--no-reasoning` | off | Pass output through without splitting off `</think>` |
| `--skip-install` | off | Trust the environment, skip dependency checks |
| `--skip-smoke-test` | off | Serve without checking that the model generates |
| `--download-only` | off | Fetch and verify the model, then exit |

## Using the API

Any OpenAI client works. Point `base_url` at the server plus `/v1`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-ID.gradio.live/v1",
    api_key="not-used",  # this server does not check keys
)

response = client.chat.completions.create(
    model="qwen3.8-27b-ud-q5-k-xl",
    messages=[{"role": "user", "content": "Explain tensor parallelism briefly."}],
)

print(response.choices[0].message.content)
```

`GET /health` tells you the exact model id to use if you are unsure.

Reasoning models wrap their scratchpad in `</think>` tags. Non-streaming responses have it stripped for you; streaming responses do not, by default. [docs/api.md](docs/api.md) explains why and how to change it. Ordinary models never emit the tag, so nothing special happens for them.

## Requirements

| | Minimum | Notes |
| --- | --- | --- |
| GPU | NVIDIA, CUDA 12.1+ | Verified on Kaggle 2 × Tesla T4 (CUDA 12.8) |
| VRAM | model size + ~3 GiB | The `+3` is the 16K KV cache; scales with `--ctx` |
| System RAM | 16 GB | 32 GB on Kaggle |
| Disk | model size + 5% | Checked before downloading |
| Python | 3.10+ | 3.12 on Kaggle |

Multiple GPUs are pooled, so 2 × 16 GB serves a 19 GiB model fine. Below the VRAM figure nothing fails outright — llama.cpp spills layers to system RAM and generation gets dramatically slower.

## How it is put together

Each module maps to a stage of the launch sequence, in the order `launch.py` runs them:

```
launch.py                entry point and command-line flags
ggufserve/config.py      all tunable settings
ggufserve/system.py      host inspection, console helpers
ggufserve/installer.py   CUDA llama.cpp + Gradio setup, skipped when correct
ggufserve/model.py       download, verify, load the GGUF
ggufserve/chat.py        generation, inference lock, reasoning-tag parsing
ggufserve/api.py         OpenAI-compatible routes
ggufserve/webui.py       routes serving the chat UI
ggufserve/server.py      assembles and launches the one public server
web/                     chat UI (html, css, js)
tests/                   tests that run without a GPU
```

The whole thing is a single `gradio.Server`, which is a FastAPI app. Because the API routes, the chat UI, and the share tunnel all belong to that one app, they share one public origin — that is why there is no second process and no second tunnel to expose.

The tests stub the model out, so they need neither a GPU nor llama.cpp and run in a second on a laptop:

```
pip install fastapi httpx
python tests/test_routes.py
```

Docs: [configuration](docs/configuration.md) · [API reference](docs/api.md) · [troubleshooting](docs/troubleshoot.md)

## Limits worth knowing

- **The public URL is unauthenticated.** Anyone with the link can use the model and spend your GPU quota. Treat it as temporary and share it carefully.
- **One request at a time.** A single llama.cpp context cannot serve concurrent requests, so requests queue. This is a deliberate fit for a small GPU box, not an oversight.
- **The share URL is temporary.** Gradio gives it up to a week, but in practice the notebook's own session limit ends it sooner, and it dies with the process either way.
- **On a hosted notebook the model is re-downloaded after a restart**, because it lands in scratch space. [docs/configuration.md](docs/configuration.md) covers how to avoid that.
- **Multi-part GGUF splits are not handled.** The model has to be one self-contained file.

## Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for inference
- [Gradio](https://github.com/gradio-app/gradio) for the app server and share tunnel
- [Unsloth](https://github.com/unslothai/unsloth) for the GGUF quantizations used by default
- Clone-and-run packaging inspired by [Fooocus](https://github.com/lllyasviel/Fooocus)

## License

MIT — see [LICENSE](LICENSE). Model weights carry their own licenses.
