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

The first run takes roughly 15 minutes, so it reports itself as eight numbered steps, and the two slow ones — the download and loading the weights — print a progress bar and an elapsed-time line respectively. Silence never means it is working.

```
[4/8] Locating model
   Qwen3.8-27B-UD-Q5_K_XL.gguf
   expected size 19.44 GiB (from the server)
   [ok] already present (19.44 GiB)

[5/8] Loading model
   ... loading weights onto the GPU (15s elapsed)
   ... loading weights onto the GPU (30s elapsed)
   took 41s
   [ok] model loaded
```

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
  context      16,384 tokens (KV cache f16)
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

Find the model's `.gguf` on Hugging Face, copy the URL out of your browser, and pass it to `--model`:

```
python launch.py --model https://huggingface.co/unsloth/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf
```

That is the whole change. The `/blob/` page URL is the one you get from the address bar, and it is translated to a real download URL for you.

In the notebook, put the same URL in `MODEL_URL` at the top of cell 2 instead.

Everything downstream follows automatically: the API model id is derived from the filename (`qwen3-8b-q4-k-m`), the download size is read from the server so nothing needs looking up, and the chat UI labels itself from `/health`.

### Finding that URL on Hugging Face

1. Search for the model with **GGUF** in the name — `Qwen3-8B-GGUF`, not `Qwen3-8B`. The plain repo holds the full-precision weights, which llama.cpp cannot load.
2. Open the **Files** tab. A quantization repo usually holds a dozen `.gguf` files that differ only in size and quality.
3. Pick one that fits your VRAM, with about 3 GiB spare for the KV cache. Hugging Face shows each file's size next to it. Higher numbers in the name are better quality and larger: `Q4_K_M` is a good default, `Q8_0` is near-lossless, `Q2_K` is small but noticeably degraded.
4. Click the file, then copy the URL from your browser's address bar. That is what `--model` wants.

Two kinds of file cannot be served, so skip them: **multi-part splits** ending `-00001-of-00002.gguf`, which have to be merged first, and helper files like `mmproj-*.gguf` or `imatrix*.gguf`, which are not weights.

For a permanent change, edit [`ggufserve/config.py`](ggufserve/config.py) instead. Single-GPU setups and mismatched GPU pairs are covered in [docs/configuration.md](docs/configuration.md).

## Running it

### Kaggle

Open [`notebook/gguf-serve.ipynb`](notebook/gguf-serve.ipynb) and run cell 1, then cell 2. Set the accelerator to **GPU T4 x2** first (*Settings → Accelerator*) — the default model needs about 23 GiB of VRAM, which is two T4s.

You do not need the flags below in the notebook. Cell 2 opens with every setting as a named constant, already holding its shipped default, so changing the model or the context is a matter of editing a value in place:

```python
MODEL_REPO = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE = "Qwen3.8-27B-UD-Q5_K_XL.gguf"
CTX = 16384
KV_CACHE = "f16"
TENSOR_SPLIT = "1,1"
```

Leave cell 2 running. The server lives inside it, so stopping the cell takes the public URL down with it.

### Colab

The same notebook works, with one caveat: **free Colab gives you a single 16 GB T4**, and the default model does not fit. Either use an L4 or A100 runtime, or pick a model that fits one card — in cell 2:

```python
MODEL_FILE = "Qwen3.8-27B-UD-Q3_K_XL.gguf"
TENSOR_SPLIT = "none"
```

### Your own machine

```
python launch.py --no-share
```

`--no-share` skips the public tunnel and serves on `http://127.0.0.1:7860` only.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--model` | — | A Hugging Face URL or `.gguf` filename; sets the three below at once |
| `--model-file` | `Qwen3.8-27B-UD-Q5_K_XL.gguf` | The `.gguf` to serve |
| `--model-repo` | `unsloth/Qwen3.8-27B-GGUF` | Hugging Face repo holding it |
| `--model-url` | — | Download from here instead of Hugging Face |
| `--model-dir` | `/tmp/gguf-serve/models` | Where the `.gguf` is stored |
| `--ctx` | `16384` | Context window in tokens; see the table below |
| `--gpu-layers` | `-1` | Layers on the GPU; `-1` means all |
| `--kv-cache-type` | `f16` | `q8_0` halves what a long context costs |
| `--tensor-split` | `1.0,1.0` | GPU split; `1,2` for 16 GB + 32 GB, `none` for one GPU |
| `--port` | `7860` | Falls forward to the next free port if taken |
| `--no-share` | off | Local access only, no public URL |
| `--no-reasoning` | off | Pass output through without splitting off `</think>` |
| `--skip-install` | off | Trust the environment, skip dependency checks |
| `--skip-smoke-test` | off | Serve without checking that the model generates |
| `--download-only` | off | Fetch and verify the model, then exit |

### Context length quick reference

Context is measured in **tokens**, and every model has a ceiling baked into its weights — for Qwen3.8-27B that is 262,144 (256K). Going above it degrades output rather than extending memory, so 512K and 1M are not available for this model.

What fits 2 × 16 GB at the default Q5\_K\_XL, including weights:

| `--ctx` | | Total VRAM | On 2 × T4 |
| --- | --- | --- | --- |
| `16384` | 16K | 21.5 GiB | **default**, comfortable |
| `65536` | 64K | 24.5 GiB | comfortable |
| `131072` | 128K | 28.5 GiB | tight; use `--kv-cache-type q8_0` |
| `262144` | 256K | 28.5 GiB with `q8_0` | needs `q8_0` and smaller weights |

```
python launch.py --ctx 131072 --kv-cache-type q8_0
```

This model stretches unusually far because it is a hybrid: only 16 of its 64 layers use full attention, so only those grow with context. A conventional dense model of the same size needs roughly 4× the KV cache — scale down accordingly if you switch. Full numbers and the reasoning are in [docs/configuration.md](docs/configuration.md).

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
