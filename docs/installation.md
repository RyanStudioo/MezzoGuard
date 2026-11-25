---
layout: page
title: Installation
---

This page covers everything you need to install Mezzo Guard: the SDK itself,
PyTorch (which is deliberately *not* a dependency), and the optional extras that
unlock the ONNX backend, the HTTP server, and the remote client.

## Requirements

- Python **3.11 or newer**
- [PyTorch](https://pytorch.org/get-started/locally/) — installed separately (see below)

## Install the SDK

```bash
pip install mezzo-guard
```

The package itself only pulls in `transformers` and `huggingface_hub`, so
installing it is fast and light.

## Install PyTorch

Mezzo Guard does not pin or install PyTorch, so you are free to use the build
that matches your platform (CUDA, CPU, ROCm, and so on):

```bash
pip install torch
```

On Linux this default wheel already bundles CUDA support. On Windows it is
CPU-only, and on any platform you may want to pin a specific CUDA version —
that is what the dedicated wheel indexes below are for.

### Installing PyTorch with CUDA

PyTorch publishes one wheel index per CUDA version. Pick the one that matches
your NVIDIA driver on
[pytorch.org](https://pytorch.org/get-started/locally/) — at the time of
writing the selector offers CUDA 11.8, 12.6, and 12.8:

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 (older drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

You only need an NVIDIA driver — the wheels bundle the CUDA runtime, so the
full CUDA toolkit does not have to be installed.

Verify that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

When that prints `True`, Mezzo Guard will run on the GPU as soon as you pass
`device="cuda"` (or `device="auto"`), and `dtype="auto"` selects the best
float precision your card supports. The full device and precision reference
lives in [Performance]({{ '/performance/' | relative_url }}).

PyTorch is only needed when you load a model. Importing `mezzoguard` works
without it, and if you try to load a model without PyTorch installed you get a
`RuntimeError` with exactly this hint:

```
Mezzo Guard requires PyTorch to load models, but it is not installed.
Install it with `pip install torch` (use the build matching your platform).
```

## Optional extras

| Extra | Command | What it adds |
|-------|---------|--------------|
| `onnx` | `pip install "mezzo-guard[onnx]"` | The ONNX Runtime backend (`Guard(name, use_onnx=True)`). Exports the model once, then reuses the cached artifacts. |
| `server` | `pip install "mezzo-guard[server]"` | FastAPI + uvicorn, for `mezzo-guard serve`. |
| `client` | `pip install "mezzo-guard[client]"` | httpx, for the remote client (`APIGuard`). |

You can install several at once:

```bash
pip install "mezzo-guard[server,client]"
```

## Verify your installation

```bash
python -c "import mezzoguard; print(mezzoguard.__version__)"
```

The version is single-sourced from `mezzoguard.__version__`.

## Local development

To work on Mezzo Guard itself, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/RyanStudioo/MezzoGuard.git
cd MezzoGuard
uv sync                  # installs runtime + dev dependencies
```

Run the fast offline test suite (no PyTorch or network required):

```bash
uv run pytest tests/test_*.py
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

## See also

- [Quickstart]({{ '/quickstart/' | relative_url }}) — load a guard and scan your first text.
- [Models]({{ '/models/' | relative_url }}) — pick a model to load.
- [Performance]({{ '/performance/' | relative_url }}) — precision, backends, and devices.
