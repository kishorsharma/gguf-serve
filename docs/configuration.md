# Configuration

Everything tunable lives in [`qwenk/config.py`](../qwenk/config.py). A few settings can also be overridden per run with a command-line flag, which is the better choice when you are still experimenting.

| Setting | Flag | Default |
| --- | --- | --- |
| `MODEL_DIR` | `--model-dir` | `/tmp/qwen-k/models` |
| `CTX_SIZE` | `--ctx` | `16384` |
| `SERVER_PORT` | `--port` | `7860` |
| `SHARE` | `--no-share` | on |

`MODEL_DIR` also reads the `QWENK_MODEL_DIR` environment variable, which is handy in notebooks where editing a file is awkward.

## Keeping the model across restarts

By default the GGUF goes to `/tmp`, which a Kaggle or Colab restart wipes — costing you the 19.4 GiB download again. Kaggle's persistent `/kaggle/working` is capped at 20 GB and counts against your output quota, so it is not a workable home for a 19.4 GiB file.

The reliable fix is to store the model outside the notebook:

**Kaggle** — download the GGUF once, upload it as a Kaggle Dataset, attach the dataset to your notebook, then point Qwen-K at it:

```
python launch.py --model-dir /kaggle/input/qwen38-27b-q5-k-xl
```

Attached datasets are read-only, which is fine: the file is already there, so Qwen-K validates it and skips the download.

**Colab** — mount Drive and use a folder on it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```
python launch.py --model-dir /content/drive/MyDrive/qwen-k/models
```

Drive is slow to read from, so loading takes noticeably longer than from local disk. It still beats re-downloading.

## Using a different quantization

`MODEL_FILE` in `qwenk/config.py` selects the file to pull from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Change three values together:

```python
MODEL_FILE = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
MODEL_ID = "qwen3.8-27b-q4-k-xl"   # the name the API advertises
MIN_MODEL_GIB = 16.0               # just under the real file size
```

`MIN_MODEL_GIB` is how a truncated download is caught, so it has to track the file you chose. Set it slightly below the real size.

Useful options, with total VRAM needed for weights plus a 16K KV cache:

| File | Size | VRAM | Notes |
| --- | --- | --- | --- |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.2 GiB | ~12.5 GiB | Fits one 16 GB GPU; noticeable quality loss |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 12.2 GiB | ~15.5 GiB | Tight on one 16 GB GPU |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.4 GiB | ~19.5 GiB | Faster than the default, leaves room for a bigger context |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 19.4 GiB | ~23 GiB | **Default.** Verified on 2 x T4 |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 23.6 GiB | ~27 GiB | Near-lossless; only just fits 2 x 16 GB |

Going below the VRAM figure does not fail — llama.cpp moves layers to system RAM and generation gets dramatically slower.

## Single GPU

`TENSOR_SPLIT` defaults to `[1.0, 1.0]`, an even split across two GPUs. For one GPU:

```python
TENSOR_SPLIT = None
```

You will also need a quantization that fits in a single card — `Q3_K_XL` or smaller for a 16 GB GPU. If the entry count does not match the number of visible GPUs, Qwen-K warns and splits evenly instead of letting llama.cpp address a device that is not there.

For two GPUs of different sizes, weight the split by VRAM, e.g. `[1.0, 2.0]` for a 16 GB paired with a 32 GB.

## Context size

`CTX_SIZE` is 16384. The KV cache grows linearly with it, so raising it costs VRAM you may not have; if loading fails after a bump, this is the first thing to bring back down. `--ctx 8192` is a quick way to free a couple of GiB.

## Sampling defaults

`TEMPERATURE`, `TOP_P`, `TOP_K` and `MAX_TOKENS` are the values used when a request does not specify its own. The defaults (1.0 / 0.95 / 20) are Qwen's recommendations for thinking mode. Per-request values in the API always win.
