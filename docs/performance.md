---
layout: page
title: Performance
---

Mezzo Guard offers three inference backends and several knobs for precision,
device placement, caching, and memory. This page explains each one and when to
use it.

All of these are constructor options on `Guard`:

```python
from mezzoguard import Guard

guard = Guard(
    "RyanStudio/Mezzo-Content-Guard-Small",
    dtype="auto",         # precision
    torch_compile=False,  # torch.compile backend
    compile_mode="default",
    use_onnx=False,       # ONNX Runtime backend
    device=None,          # device placement
)
```

## Precision: `dtype`

| Value | Meaning |
|-------|---------|
| `"auto"` (default) | `bfloat16` on GPUs that support it, otherwise `float16` (CPU included). |
| `"float32"` | Full precision — most accurate, slowest, most memory. |
| `"float16"` | Half precision on GPU. |
| `"bfloat16"` | Brain float — same memory as fp16 with a wider exponent range. |

```python
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small", dtype="float16")
```

On the default backend the model id is passed to `transformers.pipeline` as a
string, so transformers can retry the load in fp32 if the requested precision
fails on your device. (The `torch_compile` path loads the model itself and
does not get that fallback.)

## `torch.compile`

```python
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small", torch_compile=True)
guard = Guard("...", torch_compile=True, compile_mode="reduce-overhead")
```

`compile_mode` accepts the usual torch.compile modes (`"default"`,
`"reduce-overhead"`, `"max-autotune"`). Compilation happens once at load, with
dynamic shapes enabled, and pays off on repeated batches.

`torch_compile` and `use_onnx` are mutually exclusive — combining them raises
`ValueError`.

## ONNX Runtime backend

```bash
pip install "mezzo-guard[onnx]"
```

```python
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small", use_onnx=True)
```

The first load exports the model to ONNX and caches the artifacts under
`~/.cache/mezzoguard/onnx/<model-name>/`; every later load reuses them without
re-exporting. Local model directories are hashed into
`~/.cache/mezzoguard/onnx/local/<basename>-<path-hash>/` so your model folder
is never written to.

The export is atomic (it writes to a temporary sibling directory, then renames
it into place), so an interrupted export is simply redone next time — a
finished one is never re-exported.

## Choosing a device

| `device` | Meaning |
|----------|---------|
| `None` (default) | Let the backend decide (current behavior). |
| `"auto"` | CUDA when available, otherwise CPU. |
| `"cpu"` | Force CPU. |
| `"cuda"`, `"cuda:1"`, ... | A specific GPU. |
| `"mps"` | Apple Silicon. |

```python
guard = Guard("...", device="cuda")        # first GPU
guard = Guard("...", use_onnx=True, device="cpu")
```

The ONNX backend supports only `"cpu"` and `"cuda"` (with optional
`:N` index); anything else raises `ValueError`.

## The prediction cache

Repeated and duplicate chunks skip the model entirely — results are cached by
exact chunk text in a per-guard LRU cache:

```python
print(guard.cache_info())
# {"hits": 12, "misses": 3, "size": 3, "maxsize": 1024}

guard.cache_clear()
```

Toggle it at runtime (checked on every call):

```python
import mezzoguard

mezzoguard.config.cache_enabled = False
```

The capacity comes from `mezzoguard.config.cache_maxsize` (default 1024) and is
bound when the module is imported — change it before importing, or just leave
it alone.

## Freeing memory

Models are big. Drop one when you are done:

```python
guard.eject_model()   # deletes the pipeline and empties the CUDA cache
guard.load_model()    # load it back when needed
```

Or scope the whole lifetime:

```python
with Guard("RyanStudio/Mezzo-Content-Guard-Small") as guard:
    results = guard.batch_scan(texts)
# ejected on exit
```

This pairs well with `dtype` and `device`: a small model in fp16 on one GPU
often outperforms a big model in fp32, at a fraction of the memory.

## Which backend should I use?

| Situation | Suggestion |
|-----------|------------|
| Getting started, prototyping | Defaults (`dtype="auto"`, torch backend). |
| CPU-only serving, high throughput | `use_onnx=True` — usually the fastest CPU inference. |
| Long-running GPU service with steady traffic | `torch_compile=True` and/or fp16/bf16. |
| Occasional scans from a script | Defaults + the prediction cache; eject when done. |

## Benchmarking yourself

The repository ships two manual benchmark scripts (run them with
`uv run python tests/...`; they download real models and need torch, plus onnx
for the ONNX one):

- `tests/compile_test.py` — torch.compile timing.
- `tests/onnx_benchmark.py` — ONNX backend timing.

## See also

- [Scanning]({{ '/scanning/' | relative_url }}) — the prediction cache in context.
- [HTTP Server]({{ '/server/' | relative_url }}) — the same flags on `mezzo-guard serve`.
- [Guard API]({{ '/api/guard/' | relative_url }}) — constructor reference.
