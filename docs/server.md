---
layout: page
title: HTTP Server
---

Mezzo Guard ships a FastAPI server that wraps any supported model in a small
REST API, so non-Python services can scan and redact text too. The
[`APIGuard`]({{ '/remote-client/' | relative_url }}) client matches it
perfectly, and the interactive OpenAPI docs are generated for you.

## Install

```bash
pip install "mezzo-guard[server]"
```

## Start a server

```bash
mezzo-guard serve RyanStudio/Mezzo-Content-Guard-Small
# Serving RyanStudio/Mezzo-Content-Guard-Small on http://127.0.0.1:8000
```

The model is the unified `Guard` — a prompt model serves `{"unsafe": score}`,
a content model serves per-category scores.

### CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--host` | `127.0.0.1` | Bind host. Use `0.0.0.0` to expose the server. |
| `--port` | `8000` | Bind port. |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16`. |
| `--onnx` | off | Use the ONNX Runtime backend. |
| `--torch-compile` | off | `torch.compile` the model. Mutually exclusive with `--onnx`. |
| `--compile-mode` | `default` | torch.compile mode (`reduce-overhead`, `max-autotune`, ...). |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`. |
| `--max-batch-texts` | `64` | Maximum texts per request. |
| `--max-text-chars` | `50000` | Maximum characters per text. |
| `--log-level` | `info` | uvicorn log level. |

Examples:

```bash
# fp16 on the first GPU, exposed on the network
mezzo-guard serve RyanStudio/Mezzo-Content-Guard-Small \
  --host 0.0.0.0 --dtype float16 --device cuda

# ONNX backend on CPU with tighter request limits
mezzo-guard serve RyanStudio/Mezzo-Prompt-Guard-v2-Small \
  --onnx --device cpu --max-batch-texts 32 --max-text-chars 10000
```

The performance flags match the SDK — see
[Performance]({{ '/performance/' | relative_url }}) for what each one does.
They are load-time settings only; requests cannot override `dtype`, `device`,
or the backend.

Interactive OpenAPI docs are served at **`/docs`** once the server is running.

## Endpoints

### `GET /healthz`

Liveness probe.

```bash
curl -s http://127.0.0.1:8000/healthz
```

```json
{
  "status": "ok",
  "ready": true,
  "model": "RyanStudio/Mezzo-Content-Guard-Small"
}
```

`ready` is `true` once the model pipeline is loaded.

### `GET /v1/info`

Server metadata: the loaded model, the categories it scores, and its defaults.
For a server started with `--dtype float16 --device cuda`:

```bash
curl -s http://127.0.0.1:8000/v1/info
```

```json
{
  "model": "RyanStudio/Mezzo-Content-Guard-Small",
  "categories": ["hate-speech", "self-harm", "sexual", "toxic", "violence"],
  "default_max_seq_length": 64,
  "default_overlap": 16,
  "default_replace": "[REDACTED]",
  "backend": "torch",
  "dtype": "float16",
  "device": "cuda"
}
```

### `POST /v1/scan`

Scan a batch of texts. Request fields:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `texts` | list of strings | *required* | The texts to scan (at least one). |
| `confidence` | float | `0.5` | Threshold for `violated`/`unsafe` when no `policy` is given. |
| `policy` | object | `null` | `{category: threshold}` overrides. |
| `default_threshold` | float | `null` | Fallback for categories missing from `policy`; falls back to `confidence`. |
| `max_seq_length` | int | `64` | Chunk size in tokens. |
| `overlap` | int | `16` | Chunk overlap in tokens. |
| `include_chunks` | bool | `false` | Also return raw per-chunk predictions. |

```bash
curl -s http://127.0.0.1:8000/v1/scan \
  -H 'content-type: application/json' \
  -d '{"texts": ["I want to hurt myself", "have a nice day"]}'
```

```json
{
  "results": [
    {
      "scores": {"hate-speech": 0.05, "self-harm": 0.99, "sexual": 0.05, "toxic": 0.05, "violence": 0.05},
      "violated": ["self-harm"],
      "unsafe": true,
      "chunks": null
    },
    {
      "scores": {"hate-speech": 0.01, "self-harm": 0.01, "sexual": 0.01, "toxic": 0.01, "violence": 0.01},
      "violated": [],
      "unsafe": false,
      "chunks": null
    }
  ]
}
```

Each result carries `scores` (every configured category), `violated` (the
categories that met their threshold), and `unsafe` (true when `violated` is
non-empty). With `include_chunks: true`, `chunks` holds one entry per chunk,
each a list of `{"label": ..., "score": ...}` dicts.

With a policy on the wire:

```bash
curl -s http://127.0.0.1:8000/v1/scan \
  -H 'content-type: application/json' \
  -d '{"texts": ["I will kill you"], "policy": {"violence": 0.99}, "default_threshold": 0.7}'
```

`violence` is judged against `0.99`; every other category against `0.7`.

### `POST /v1/redact`

Redact a batch of texts. Request fields:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `texts` | list of strings | *required* | The texts to redact (at least one). |
| `confidence` | float | `0.5` | Fallback threshold when no `policy` is given. |
| `policy` | object | `null` | `{category: threshold}` overrides. |
| `default_threshold` | float | `null` | Fallback for categories missing from `policy`. |
| `replace` | string | `"[REDACTED]"` | Replacement marker. |
| `replacements` | object | `null` | Per-category `{category: replacement}` — wins over `replace`. |
| `max_seq_length` | int | `64` | Chunk size in tokens. |
| `overlap` | int | `16` | Chunk overlap in tokens. |
| `return_spans` | bool | `false` | Also return the positions of every replaced region. |

```bash
curl -s http://127.0.0.1:8000/v1/redact \
  -H 'content-type: application/json' \
  -d '{"texts": ["I will kill you"], "return_spans": true}'
```

```json
{
  "results": [
    {
      "text": "[REDACTED]",
      "spans": [
        {"start": 0, "end": 15, "category": "violence", "replacement": "[REDACTED]"}
      ]
    }
  ]
}
```

Without `return_spans`, `spans` is `null`. Span offsets index into the
**original** text — see [Redaction]({{ '/redaction/' | relative_url }}).

Per-category replacements:

```bash
curl -s http://127.0.0.1:8000/v1/redact \
  -H 'content-type: application/json' \
  -d '{"texts": ["I will kill you"], "replacements": {"violence": "[VIOLENCE]"}}'
# {"results": [{"text": "[VIOLENCE]", "spans": null}]}
```

## Request limits

The server validates before doing any work:

| Condition | Response |
|-----------|----------|
| More texts than `--max-batch-texts` | `413` with a message naming the limit. |
| A text longer than `--max-text-chars` | `413` with a message naming the limit. |
| Malformed request (e.g. empty `texts` list, `confidence` out of 0–1) | `422` with pydantic validation details. |

Set both limits from the CLI (`--max-batch-texts`, `--max-text-chars`) or via
`create_app` when embedding.

## Embedding the app

The CLI is a thin wrapper. For custom deployments (auth middleware, mounted
under a path, custom startup) build the app yourself:

```python
from mezzoguard import Guard
from mezzoguard.server import create_app

guard = Guard("RyanStudio/Mezzo-Content-Guard-Small", dtype="float16")
app = create_app(
    guard,
    max_batch_texts=64,      # 413 above this many texts per request
    max_text_chars=50_000,   # 413 above this many characters per text
)

# serve with any ASGI server, e.g.: uvicorn mymodule:app
```

`create_app` takes an already-loaded guard. On shutdown the app ejects the
model, so the process exits cleanly.

## See also

- [Remote Client]({{ '/remote-client/' | relative_url }}) — call this server from Python with the full `Guard` API.
- [Performance]({{ '/performance/' | relative_url }}) — what `--dtype`, `--onnx`, and friends do.
- [Models]({{ '/models/' | relative_url }}) — what you can serve.
