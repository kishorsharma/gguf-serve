# Troubleshooting

## "CUDA GPU offload is unavailable"

`launch.py` stops here rather than continuing, because the alternative is a model that loads onto the CPU and generates a token every few seconds.

The cause is almost always that pip installed the PyPI build of `llama-cpp-python` instead of a CUDA wheel. The PyPI build has no GPU support and compiles from source for 20+ minutes on the way to being useless.

Reinstall from the wheel index, forcing a binary:

```
pip uninstall -y llama-cpp-python
pip install --upgrade --only-binary=:all: llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

`--only-binary=:all:` is what stops pip falling back to a source build.

If your CUDA is older than 12.5, change the index to match — `cu124`, `cu123`, `cu122`, or `cu121` — and update `LLAMA_CPP_WHEEL_INDEX` in `ggufserve/config.py`. Check your version with `nvcc --version` or `nvidia-smi`. The cu125 wheels run fine on newer drivers, so CUDA 12.8 (what Kaggle ships) needs no change.

Also confirm the notebook actually has a GPU attached. On Kaggle that is *Settings → Accelerator → GPU T4 x2*; if `nvidia-smi` prints nothing there is no GPU, and no wheel will fix that.

## Out of memory while loading

Budget roughly the model file size plus 3 GiB for a 16K KV cache. The default needs about 23 GiB, which fits 2 × T4 (29.8 GiB) with room to spare — but not if something else is already resident.

In order of what to try:

1. **Shrink the context.** `python launch.py --ctx 8192` frees a couple of GiB.
2. **Check nothing else holds VRAM.** `nvidia-smi` will show it. In a notebook, a previously loaded model stays in memory until the kernel restarts; restarting the runtime is the only reliable way to reclaim it.
3. **Use a smaller model.** `--model-file Qwen3.8-27B-UD-Q4_K_XL.gguf` needs ~19.5 GiB. See [configuration.md](configuration.md).
4. **Control the spill yourself.** `--gpu-layers 40` keeps the remainder on the CPU deliberately, which is more predictable than letting the allocator fail.

A model that loads but generates extremely slowly usually means llama.cpp put some layers in system RAM. The GPU summary printed after loading shows how much VRAM was actually claimed.

## Download is slow or keeps failing

Downloads resume. Re-run `python launch.py` and curl picks up where it stopped, using the `.part` file next to the target.

The partial file is only renamed into place once the transfer completes, so an interrupted download can never be mistaken for a finished model. A file that fails verification is deleted and re-fetched.

To fetch a model without loading it — useful for prepping a Kaggle dataset:

```
python launch.py --download-only
```

## "Downloaded model failed validation"

The expected size comes from a `HEAD` request against the download URL, and the finished file is compared against it exactly. A mismatch means the transfer was truncated, so the file is rejected rather than handed to llama.cpp.

Just re-run — the download resumes. If it fails repeatedly, the URL may be wrong: check that `--model-file` names a file that actually exists in `--model-repo`, since Hugging Face returns a small HTML error page for a bad path and that is what gets measured.

If you see *"could not reach the server, so size was not verified"*, gguf-serve is offline and fell back to checking only the GGUF magic bytes. A model it downloaded itself is still trustworthy; one you copied in manually is not verified beyond its header.

## "Not enough disk space"

Checked before downloading, against the real file size plus 5%. On Kaggle, `/kaggle/working` has only 20 GB, which is why the default target is `/tmp`. Point somewhere larger with `--model-dir` and check free space with `df -h`.

## Gradio import fails after installing

Symptom: an `ImportError` from `huggingface_hub` internals right after the Gradio install step.

Installing Gradio also upgrades `huggingface_hub`. If an older version was already imported in the same process, the two disagree about their own internals. gguf-serve evicts the cached modules and retries automatically, and running `launch.py` as a fresh process avoids it entirely.

If you hit it in a long-lived notebook kernel, restart the kernel and re-run. The model file on disk is unaffected, so only the load is repeated.

## Port already in use

gguf-serve moves to the next free port automatically and prints which one it chose. To pick one yourself:

```
python launch.py --port 7861
```

In a notebook, a server from an earlier cell run keeps its port until the kernel restarts. The process holding the port *is* the kernel, so do not kill it — restart the runtime instead.

## The public URL stopped working

Share URLs are temporary. They die when:

- the process stops, including stopping the notebook cell — the server lives inside it,
- the Kaggle or Colab runtime hits its session limit,
- 72 hours pass.

Re-running `launch.py` creates a new URL at a different address. If the model is still on disk the download is skipped, so a restart is quick.

For something permanent you need a host you control; a share tunnel is not it.

## Model loads but answers are empty

The smoke test catches this before the server starts. The usual causes:

- **A corrupted download** that still passed verification. Delete the file and re-fetch.
- **A chat template mismatch.** llama.cpp reads the template from the GGUF; a model packaged without one produces empty or malformed output. Try a GGUF from a different uploader.

```
rm /tmp/gguf-serve/models/*.gguf
python launch.py
```

## Output is garbled or repetitive

Usually sampling settings that do not suit the model. The defaults (temperature 1.0, top\_p 0.95, top\_k 20) are Qwen's recommendations for thinking mode and are too loose for some models. Check what the model's authors suggest and set `TEMPERATURE`, `TOP_P` and `TOP_K` in `ggufserve/config.py`.

Also confirm `--ctx` is not above the model's own trained context length, which degrades output rather than extending usable memory.

## Responses contain `</think>` and reasoning text

Expected on streaming requests: the server streams raw output and the client separates the sections. See [api.md](api.md) for why, and for how to opt into server-side stripping.

If a non-reasoning model emits `</think>` as genuine content and gguf-serve swallows part of the answer, disable the parser with `--no-reasoning`.
