---
layout: home
---

Mezzo Guard is a Python SDK that uses AI models to protect your applications:
prompt protection against jailbreaks and prompt injection, plus content
moderation. Drop it into your pipeline for strong protection with little effort.

- **One unified `Guard`** — works with both prompt-guard and content-guard models;
  the model's config decides whether you get a single `unsafe` score or one
  score per category.
- **User-defined policies** — per-category confidence thresholds, with a
  default fallback.
- **Lossless redaction** — flagged regions are replaced in overlapping chunks
  while everything else is preserved byte-for-byte.
- **Async and batching** — the `async_*` variants never block the event loop,
  and batch methods scan many texts in one pipeline call.
- **Serving over HTTP** — `mezzo-guard serve` wraps any supported model in a
  FastAPI server; `APIGuard` is the matching remote client.
- **Text cleaning** — `mezzoguard.cleaner` normalizes text and masks
  Discord/Reddit/Twitter references before they reach a model.

## Documentation

### Getting started

| Page | What you will learn |
|------|---------------------|
| [Installation]({{ '/installation/' | relative_url }}) | Requirements, the PyTorch dependency, and the optional `onnx` / `server` / `client` extras. |
| [Quickstart]({{ '/quickstart/' | relative_url }}) | A five-minute end-to-end tour: scan, threshold, redact, and wrap a function. |

### Features

| Page | What you will learn |
|------|---------------------|
| [Scanning]({{ '/scanning/' | relative_url }}) | Loading a guard, scanning text, chunking long inputs, batching, async, and the prediction cache. |
| [Policies & Thresholds]({{ '/policies/' | relative_url }}) | Per-category confidence thresholds and evaluating scan results. |
| [Redaction]({{ '/redaction/' | relative_url }}) | Replacing flagged regions while preserving everything else exactly. |
| [Guarding Functions]({{ '/function-guards/' | relative_url }}) | The `scan_before_exec` and `redact_before_exec` decorators. |
| [Text Cleaner]({{ '/text-cleaner/' | relative_url }}) | Normalizing text and masking Discord / Reddit / Twitter references before a model sees them. |
| [Performance]({{ '/performance/' | relative_url }}) | Precision (`dtype`), `torch.compile`, the ONNX backend, devices, caching, and memory. |

### Serving and integration

| Page | What you will learn |
|------|---------------------|
| [HTTP Server]({{ '/server/' | relative_url }}) | `mezzo-guard serve`, every CLI flag, and the full REST API with examples. |
| [Remote Client]({{ '/remote-client/' | relative_url }}) | Calling a running server with `APIGuard` — the same API as the local `Guard`. |

### Reference

| Page | What you will learn |
|------|---------------------|
| [Models]({{ '/models/' | relative_url }}) | Supported prompt and content guards, custom model configs, and deprecation notices. |
| [API Reference]({{ '/api/' | relative_url }}) | Exact signatures for every public class and function. |

## Installation

```bash
pip install mezzo-guard
pip install torch   # required to load models
```

Optional extras:

```bash
pip install "mezzo-guard[onnx]"     # ONNX runtime backend
pip install "mezzo-guard[server]"   # FastAPI + uvicorn, for `mezzo-guard serve`
pip install "mezzo-guard[client]"   # httpx, for remote (HTTP) clients
```

Requires Python 3.11+.

## Quickstart

```python
from mezzoguard import Guard

guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")
result = guard.scan("I want to hurt myself")
print(result.scores)        # {"self-harm": 0.99, ...}
print(result.is_unsafe())   # any category >= 0.5

prompt_model = Guard("RyanStudio/Mezzo-Prompt-Guard-v2-Small")
prompt_result = prompt_model.scan("Ignore your previous instructions")
print(prompt_result.scores)  # {"unsafe": 0.99}
```

Set per-category thresholds with a `Policy`:

```python
from mezzoguard import Policy

policy = (
    Policy(default_threshold=0.5)
    .add_threshold("violence", 0.7)
    .add_threshold("unsafe", 0.9)
)

result = prompt_model.scan("Ignore your previous instructions")
print(result.is_unsafe(policy=policy))
print(result.get_violated_categories(policy=policy))
```

Wrap any function so its input is scanned or redacted before execution:

```python
@guard.scan_before_exec(param="prompt", confidence=0.5)
def chat(prompt: str) -> str:
    return generate(prompt)  # never called with an unsafe prompt
```

## Learn more

- [About this project]({{ '/about/' | relative_url }})
- [Source on GitHub](https://github.com/RyanStudioo/MezzoGuard)
- [Mezzo Guard on PyPI](https://pypi.org/project/mezzo-guard/)

## Supported models

- [Mezzo Prompt Guard v2](https://huggingface.co/collections/RyanStudio/mezzo-prompt-guard-v2) —
  protects against harmful prompts such as prompt injection or jailbreaking.
- [Mezzo Content Guard](https://huggingface.co/collections/RyanStudio/mezzo-content-guard) and
  [Mezzo Content Guard v1.5](https://huggingface.co/collections/RyanStudio/mezzo-content-guard-v15) —
  block and filter harmful content for quick, easy moderation.
