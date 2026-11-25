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
