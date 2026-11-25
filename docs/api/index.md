---
layout: page
title: API Reference
---

Exact signatures for every public class and function in Mezzo Guard. For
task-oriented guides with runnable examples, start from the
[home page]({{ '/' | relative_url }}).

## Importing

Everything important lives at the top level:

```python
import mezzoguard

mezzoguard.Guard            # the unified local guard
mezzoguard.APIGuard         # the remote (HTTP) guard
mezzoguard.Policy           # per-category thresholds
mezzoguard.Result           # what scan() returns
mezzoguard.PolicyResult     # what a policy evaluation returns
mezzoguard.Span             # a redacted region
mezzoguard.GuardConfig      # custom category configuration
mezzoguard.UnsafeInputError
mezzoguard.MezzoGuardServerError
mezzoguard.__version__
```

## Module map

| Import | Provides | Reference |
|--------|----------|-----------|
| `mezzoguard` | Root exports shown above. | — |
| `mezzoguard.guard` | `Guard`, `GuardConfig`, `Policy`, `Result`, `PolicyResult`, `MODELS_CONFIG`. | [Guard]({{ '/api/guard/' | relative_url }}), [Policy & Results]({{ '/api/policy-result/' | relative_url }}) |
| `mezzoguard.cleaner` | `TextCleaner`, platform cleaners, convenience functions, tokens. | [Cleaner]({{ '/api/cleaner/' | relative_url }}) |
| `mezzoguard.remote` | `APIGuard`. | [APIGuard]({{ '/api/apiguard/' | relative_url }}) |
| `mezzoguard.server` | `create_app` (FastAPI). | [HTTP Server]({{ '/server/' | relative_url }}) |
| `mezzoguard.errors` | `UnsafeInputError`, `MezzoGuardServerError`. | [Errors]({{ '/api/errors/' | relative_url }}) |
| `mezzoguard.config` | `cache_enabled`, `cache_maxsize` runtime settings. | [Performance]({{ '/performance/' | relative_url }}) |
| `mezzoguard.cli` | `mezzo-guard serve` entry point. | [HTTP Server]({{ '/server/' | relative_url }}) |

## Defaults at a glance

| Constant | Value | Used by |
|----------|-------|--------|
| `DEFAULT_MAX_SEQ_LENGTH` | `64` | `max_seq_length` everywhere (tokens). |
| `DEFAULT_OVERLAP` | `16` | `overlap` everywhere (tokens). |
| `DEFAULT_CONFIDENCE` | `0.5` | `confidence` / threshold fallback. |
| `DEFAULT_REDACTED_LABEL` | `"[REDACTED]"` | Redaction replacement. |
| `MAX_TEXT_LENGTH` | `4000` | Cleaner input cap (code points). |
| `cache_maxsize` | `1024` | Prediction cache capacity. |

## Shared guard interface

Both `Guard` and `APIGuard` implement the `GuardBase` abstract base class, so
these signatures are identical for local and remote guards:

```python
scan(text, max_seq_length=64, overlap=16) -> Result
batch_scan(texts, max_seq_length=64, overlap=16) -> list[Result]
redact(text, max_seq_length=64, overlap=16, replace="[REDACTED]",
       policy=None, confidence=0.5, return_spans=False) -> str | tuple[str, list[Span]]
batch_redact(texts, ...) -> list[str] | list[tuple[str, list[Span]]]

async_scan(...) -> Result                     # plus async_batch_scan,
async_redact(...) -> str | tuple[...]         # async_batch_redact

scan_before_exec(param, ...) -> decorator
redact_before_exec(param, ...) -> decorator
```

Batch methods are the primitives; single-text methods and the decorators are
adapters over them.

## Reference pages

- [Guard]({{ '/api/guard/' | relative_url }}) — the unified local guard.
- [Policy & Results]({{ '/api/policy-result/' | relative_url }}) — `Policy`, `Result`, `PolicyResult`, `Span`.
- [Cleaner]({{ '/api/cleaner/' | relative_url }}) — text cleaning and masking.
- [APIGuard]({{ '/api/apiguard/' | relative_url }}) — the remote client.
- [Errors]({{ '/api/errors/' | relative_url }}) — exceptions raised by the SDK.
