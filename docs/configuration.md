# Configuration

Everything tunable lives in [`ggufserve/config.py`](../ggufserve/config.py). Most settings also have a command-line flag, which is the better choice while you are still experimenting.

| Setting | Flag | Default |
| --- | --- | --- |
| `MODEL_FILE` | `--model-file` | `Qwen3.8-27B-UD-Q5_K_XL.gguf` |
| `MODEL_REPO` | `--model-repo` | `unsloth/Qwen3.8-27B-GGUF` |
| `MODEL_URL` | `--model-url` | derived from repo + file |
| `MODEL_DIR` | `--model-dir` | `/tmp/gguf-serve/models` |
| `CTX_SIZE` | `--ctx` | `16384` |
| `N_GPU_LAYERS` | `--gpu-layers` | `-1` (all) |
| `SERVER_PORT` | `--port` | `7860` |
| `SHARE` | `--no-share` | on |
| `PARSE_REASONING` | `--no-reasoning` | on |

`MODEL_DIR` also reads the `GGUF_SERVE_MODEL_DIR` environment variable, which is handy in notebooks where editing a file is awkward.

## Choosing a model

Any single-file GGUF that llama.cpp can load. Point at a Hugging Face repo and a filename inside it:

```
python launch.py \
  --model-repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --model-file Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
```

Nothing else needs configuring. The expected download size is read from the server with a `HEAD` request, so a truncated transfer is caught without you having to look up a byte count, and the API model id is derived from the filename — `meta-llama-3.1-8b-instruct-q5-k-m` here. Override it with `MODEL_ID` if you want a specific name.

Multi-part splits (`…-00001-of-00002.gguf`) are **not** supported; the model has to be one file.

To fetch from somewhere other than Hugging Face, give a direct URL:

```
python launch.py --model-url https://example.com/models/my-model.gguf
```

`MODEL_FILE` is still used as the local filename, so set both.

### Picking a size that fits

Budget roughly **model file size + 3 GiB** of VRAM at the default 16K context. Multiple GPUs are pooled, so 2 × 16 GB gives you a ~29.8 GiB budget.

For the default Qwen3.8-27B repo:

| File | Size | VRAM needed | Notes |
| --- | --- | --- | --- |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 6.8 GiB | ~10 GiB | Fits one 16 GB card easily; quality clearly degraded |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.2 GiB | ~12.5 GiB | Fits one 16 GB card; noticeable quality loss |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 12.2 GiB | ~15.5 GiB | Tight on one 16 GB card |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.4 GiB | ~19.5 GiB | Faster than the default, room for a bigger context |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 19.4 GiB | ~23 GiB | **Default.** Verified on 2 × T4 |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 23.6 GiB | ~27 GiB | Near-lossless; only just fits 2 × 16 GB |

Going under the VRAM figure does not fail — llama.cpp moves layers to system RAM and generation gets much slower. If you are close, `--gpu-layers` lets you control the spill explicitly instead of leaving it to chance.

## Keeping the model across restarts

By default the GGUF goes to `/tmp`, which a Kaggle or Colab restart wipes — costing you the download again. Kaggle's persistent `/kaggle/working` is capped at 20 GB and counts against your output quota, so it is not a workable home for a large file.

The reliable fix is to store the model outside the notebook.

**Kaggle** — fetch the file once with `--download-only`, upload it as a Kaggle Dataset, attach the dataset, then point at it:

```
python launch.py --model-dir /kaggle/input/your-dataset-name
```

Attached datasets are read-only, which is fine: the file is already there, so it is verified and the download skipped.

**Colab** — mount Drive and use a folder on it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```
python launch.py --model-dir /content/drive/MyDrive/gguf-serve/models
```

Drive is slow to read, so loading takes noticeably longer than from local disk. It still beats re-downloading.

## GPUs

`TENSOR_SPLIT` decides how the model is divided.

```python
TENSOR_SPLIT = [1.0, 1.0]   # even split across two GPUs (default)
TENSOR_SPLIT = None         # single GPU
TENSOR_SPLIT = [1.0, 2.0]   # 16 GB paired with a 32 GB
```

If the entry count does not match the number of visible GPUs, gguf-serve warns and splits evenly rather than letting llama.cpp address a device that is not there. So a single-GPU machine works without editing anything — but you still need a model that fits one card.

## Context size

Context is measured in **tokens, not bytes** — roughly 0.75 words or 4 characters each, so 16K tokens is about 12,000 words. A "1 MB context" is not a thing you can ask for.

Every model has a hard ceiling built into its weights. For Qwen3.8-27B that is **262,144 tokens (256K)**. Setting `--ctx` above a model's ceiling does not extend its memory; it degrades output, because the model is being used beyond the positions it was trained for. So 512K and 1M are not available for this model — you would need a model trained for them, such as the Qwen2.5-1M series.

### Quick reference for Qwen3.8-27B on 2 × 16 GB

The KV cache is allocated for the whole context up front, and its size is what decides whether a context length fits. For this model it costs **64 KiB per token** at `f16`, or **32 KiB per token** at `q8_0`.

Totals below include the Q5\_K\_XL weights and about 1 GiB of buffers. Usable VRAM on 2 × T4 is roughly 29.1 GiB, since the driver reserves a little of each card.

| `--ctx` | Tokens | KV f16 | Total f16 | KV q8_0 | Total q8_0 | On 2 × T4 |
| --- | --- | --- | --- | --- | --- | --- |
| `8192` | 8K | 0.5 GiB | 21.0 GiB | 0.2 GiB | 20.7 GiB | comfortable |
| `16384` | 16K | 1.0 GiB | 21.5 GiB | 0.5 GiB | 21.0 GiB | **default**, comfortable |
| `32768` | 32K | 2.0 GiB | 22.5 GiB | 1.0 GiB | 21.5 GiB | comfortable |
| `65536` | 64K | 4.0 GiB | 24.5 GiB | 2.0 GiB | 22.5 GiB | comfortable |
| `131072` | 128K | 8.0 GiB | 28.5 GiB | 4.0 GiB | 24.5 GiB | tight at f16; fine at q8\_0 |
| `262144` | 256K | 16.0 GiB | 36.5 GiB | 8.0 GiB | 28.5 GiB | q8\_0 only, and tight |

So the practical picks are **64K comfortably**, **128K with `--kv-cache-type q8_0`**, and the model's full **256K only by also dropping to a smaller quantization**:

```
# 128K, comfortable
python launch.py --ctx 131072 --kv-cache-type q8_0

# 256K, the model's maximum — needs the smaller weights to leave room
python launch.py --ctx 262144 --kv-cache-type q8_0 \
                --model-file Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The startup summary prints actual VRAM use per GPU, which is the real check — a number well below capacity means llama.cpp spilled layers to system RAM and you should reduce the context.

### Why this model stretches so far

A dense 27B model would need roughly 4 × this much KV cache. Qwen3.8-27B is a **hybrid**: of its 64 layers only 16 use full attention, and the other 48 use linear attention whose state is a fixed size regardless of how long the context gets. Only those 16 layers grow with `--ctx`, which is what makes a 256K context conceivable on consumer hardware at all.

If you switch to a conventional dense model, expect these numbers to be several times worse and scale your context down accordingly.

### KV cache precision

`KV_CACHE_TYPE` is `f16`, llama.cpp's default. `q8_0` halves the cache for a small quality cost, and is the difference between a long context fitting and not.

Selecting it also turns on flash attention, because llama.cpp can only read a quantized cache through it. That needs a build with flash-attention support for your GPU; T4 and newer qualify. If loading fails right after you switch, this is the reason — go back to `f16` and use a shorter context instead.

## Reasoning models

`PARSE_REASONING` is on. It splits off the `</think>`-delimited scratchpad that reasoning models (Qwen3, DeepSeek-R1, QwQ) emit before their answer.

Leaving it on is harmless for ordinary models, which never emit the tag — the parser treats untagged output as a plain answer. Turn it off with `--no-reasoning` if you want the raw text untouched, for example when a model legitimately produces `</think>` as content.

## Sampling defaults

`TEMPERATURE`, `TOP_P`, `TOP_K` and `MAX_TOKENS` apply when a request does not specify its own; per-request values always win. The defaults (1.0 / 0.95 / 20) are Qwen's recommendations for thinking mode, so check what your model's authors suggest if you switch — a mismatched temperature is a common cause of output that looks broken.
