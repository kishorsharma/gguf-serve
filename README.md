# Qwen-K

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/<your-username>/qwen-k/blob/main/notebook/qwen-k.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/qwen-k/blob/main/notebook/qwen-k.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Run **Qwen3.8-27B** on a free Kaggle notebook and get a **public, OpenAI-compatible API** plus a browser chat UI — from one command.

Qwen-K handles the parts that are normally fiddly: picking a CUDA build of llama.cpp that actually uses the GPU, fetching the 19.4 GiB GGUF without stalling, splitting a 27B model across two 16 GB T4s, and exposing everything through a single public URL instead of two tunnels.

```
git clone https://github.com/<your-username>/qwen-k.git
cd qwen-k
python launch.py
```

That is the whole setup. The first run installs dependencies, downloads the model, loads it, and starts serving — about 15 minutes, mostly download. Re-running skips whatever is already done.

## What you get

One server, one origin, six endpoints:

| Path | What it is |
| --- | --- |
| `/` | Browser chat UI with live reasoning and streaming |
| `/docs` | Interactive OpenAPI reference |
| `/v1/models` | OpenAI model listing |
| `/v1/chat/completions` | OpenAI chat completions, streaming and non-streaming |
| `/health` | Health and configuration check |
| `/openapi.json` | OpenAPI schema |

With `--share` (the default) all of them are also reachable on a public `https://….gradio.live` URL, so you can point a client on your laptop at a model running in a Kaggle notebook.

## Running it

### Kaggle

Open [`notebook/qwen-k.ipynb`](notebook/qwen-k.ipynb) and run its two cells. Set the accelerator to **GPU T4 x2** first (*Settings → Accelerator*) — the default quantization needs about 23 GiB of VRAM, which is two T4s.

Leave the second cell running. The server lives inside it, so stopping the cell takes the public URL down with it.

### Colab

The same notebook works, with one caveat: **free Colab gives you a single 16 GB T4**, and the default quantization does not fit. Either use a runtime with more VRAM (L4 or A100), or switch to a smaller quantization first — set `MODEL_FILE` to `Qwen3.8-27B-UD-Q3_K_XL.gguf` and `TENSOR_SPLIT` to `None` in `qwenk/config.py`. [docs/configuration.md](docs/configuration.md) covers both.

### Your own machine

```
git clone https://github.com/<your-username>/qwen-k.git
cd qwen-k
python launch.py --no-share
```

`--no-share` skips the public tunnel and serves on `http://127.0.0.1:7860` only.

### Options

| Flag | Default | Effect |
| --- | --- | --- |
| `--port` | `7860` | Port to listen on; falls forward to the next free port if taken |
| `--no-share` | off | Local access only, no public URL |
| `--model-dir` | `/tmp/qwen-k/models` | Where the `.gguf` lives |
| `--ctx` | `16384` | Context window; lower this if the model will not fit |
| `--skip-install` | off | Trust the current environment and skip dependency checks |
| `--skip-smoke-test` | off | Start serving without first checking that the model generates |
| `--download-only` | off | Fetch and validate the model, then exit |

## Using the API

Any OpenAI client works. Point `base_url` at the server plus `/v1`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-ID.gradio.live/v1",
    api_key="not-used",  # this server does not check keys
)

response = client.chat.completions.create(
    model="qwen3.8-27b-q5-k-xl",
    messages=[{"role": "user", "content": "Explain tensor parallelism briefly."}],
)

print(response.choices[0].message.content)
```

Qwen3.8 is a reasoning model, so responses contain a `</think>`-delimited reasoning section. Non-streaming responses have it stripped for you; streaming responses do not, by default. See [docs/api.md](docs/api.md) for how that works and how to change it.

## Requirements

| | Minimum | Notes |
| --- | --- | --- |
| GPU | 2 x 16 GB NVIDIA | Verified on Kaggle 2 x Tesla T4. One 16 GB GPU is not enough for the default quantization. |
| VRAM | ~23 GiB total | Weights plus a 16K KV cache |
| System RAM | 16 GB | 32 GB on Kaggle |
| Disk | 21 GiB free | For the model file |
| Python | 3.10+ | 3.12 on Kaggle |
| CUDA | 12.1+ | 12.8 on Kaggle |

Below these numbers llama.cpp spills layers to system RAM and generation slows to a crawl rather than failing outright.

## Configuration

Everything tunable lives in [`qwenk/config.py`](qwenk/config.py) — model file, context size, GPU split, sampling defaults, dependency pins. It is plain Python with comments explaining what each value does and when to change it.

For a different quantization, persistent model storage, or single-GPU use, see [docs/configuration.md](docs/configuration.md).

## How it is put together

Each module maps to a stage of the launch sequence, in the order `launch.py` runs them:

```
launch.py            entry point and command-line flags
qwenk/config.py      all tunable settings
qwenk/system.py      host inspection, console helpers
qwenk/installer.py   CUDA llama.cpp + Gradio setup, skipped when already correct
qwenk/model.py       download, validate, load the GGUF
qwenk/chat.py        generation, inference lock, reasoning-tag parsing
qwenk/api.py         OpenAI-compatible routes
qwenk/webui.py       routes serving the chat UI
qwenk/server.py      assembles and launches the one public server
web/                 chat UI (html, css, js)
tests/               route tests that run without a GPU
```

The whole thing is a single `gradio.Server`, which is a FastAPI app. Because the API routes, the chat UI, and the share tunnel all belong to that one app, they share one public origin — that is why there is no second process and no second tunnel to expose.

The route tests stub out the model, so they need neither a GPU nor llama.cpp and run in a second on a laptop:

```
pip install fastapi httpx
python tests/test_routes.py
```

Docs: [configuration](docs/configuration.md) · [API reference](docs/api.md) · [troubleshooting](docs/troubleshoot.md)

## Limits worth knowing

- **The public URL is unauthenticated.** Anyone with the link can use the model and spend your GPU quota. Treat it as temporary and share it carefully.
- **One request at a time.** A single llama.cpp context cannot serve concurrent requests, so requests queue. This is a deliberate fit for a 2 x T4 box, not an oversight.
- **The share URL is temporary.** It lasts up to 72 hours and dies with the process.
- **On Kaggle the model is re-downloaded after a restart**, because it is stored in scratch space. See [docs/configuration.md](docs/configuration.md) to avoid that.

## Credits

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Alibaba's Qwen team
- [GGUF quantizations](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) by [Unsloth](https://github.com/unslothai/unsloth)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Gradio](https://github.com/gradio-app/gradio) for the app server and share tunnel
- Clone-and-run packaging inspired by [Fooocus](https://github.com/lllyasviel/Fooocus)

## License

MIT — see [LICENSE](LICENSE). The model weights are covered by their own license (Apache 2.0).
