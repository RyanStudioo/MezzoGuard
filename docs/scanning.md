---
layout: page
title: Scanning
---

Scanning is the core of Mezzo Guard: you hand text to a `Guard` and get back a
`Result` with one confidence score per category. This page covers loading a
guard, reading results, how long text is chunked, batching, async, and the
built-in prediction cache.

## One guard, two kinds of model

`Guard` is a single unified class. The model's category configuration decides
what you get back:

| Model family | Scores | Example |
|--------------|--------|---------|
| Prompt guards | one `unsafe` score | `{"unsafe": 0.99}` |
| Content guards | one score per category | `{"violence": 0.98, "toxic": 0.8, ...}` |

```python
from mezzoguard import Guard

prompt_guard = Guard("RyanStudio/Mezzo-Prompt-Guard-v2-Small")
content_guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")

print(prompt_guard.categories)  # ["unsafe"]
print(content_guard.categories)  # ["hate-speech", "self-harm", "sexual", "toxic", "violence"]
```

See [Models]({{ '/models/' | relative_url }}) for the full list of supported
models and how categories are resolved — including how to bring your own model.

## Loading a guard

```python
guard = Guard(
    "RyanStudio/Mezzo-Content-Guard-Small",  # HF model id or local directory
    dtype="auto",        # precision: "auto", "float32", "float16", "bfloat16"
    device=None,         # None, "auto", "cpu", "cuda", "cuda:N", "mps"
    torch_compile=False, # torch.compile the model
    use_onnx=False,      # ONNX Runtime backend
)
```

Constructing a guard loads the model right away. Loading is idempotent —
calling `guard.load_model()` again is a no-op — and you can free the memory
later with `guard.eject_model()` or by using the guard as a context manager:

```python
with Guard("RyanStudio/Mezzo-Content-Guard-Small") as guard:
    result = guard.scan("some text")
# the model is ejected when the block exits
```

The constructor also accepts a `config` argument for custom category
configurations — see [Models]({{ '/models/' | relative_url }}).

## Scanning a single text

```python
result = guard.scan("I want to hurt myself")
```

The return value is a `Result`:

| Member | Meaning |
|--------|---------|
| `result.scores` | `dict` mapping every configured category to its confidence (0.0–1.0). Scores are inclusive per category and take the **maximum across chunks**. |
| `result.chunks` | Raw per-chunk predictions: one entry per chunk, each a list of `{"label": ..., "score": ...}` dicts. |
| `result.is_unsafe(...)` | `True` when any score meets the threshold (default 0.5) or a supplied `Policy`. |
| `result.get_violated_categories(...)` | The list of categories that met the threshold. |
| `result.evaluate(policy)` | Full evaluation against a `Policy`, returning a `PolicyResult`. |

```python
result = guard.scan("I want to hurt myself")

print(result.scores)          # {"self-harm": 0.99, ...} — every category is present
print(result.is_unsafe())     # any score >= 0.5
print(result.get_violated_categories())  # ["self-harm"]
```

Every configured category always appears in `scores`, initialized to `0.0`. A
prompt guard therefore returns `{"unsafe": 0.0}` for completely safe text —
the `"safe"` label itself is never reported.

## Long text and chunking

Models have a limited context window, so Mezzo Guard splits long text into
overlapping chunks with the tokenizer's own offset mapping — chunk boundaries
are character-accurate, and the text is reconstructed losslessly.

```python
result = guard.scan(
    long_text,
    max_seq_length=64,  # tokens per chunk (default 64)
    overlap=16,          # tokens shared between consecutive chunks (default 16)
)
```

- `max_seq_length` — chunk size in **tokens**. Smaller chunks mean
  finer-grained results (useful for [redaction]({{ '/redaction/' | relative_url }})).
- `overlap` — consecutive chunks share this many tokens, so a harmful phrase
  split across a boundary is still caught.
- `overlap` must be smaller than `max_seq_length`, otherwise a `ValueError` is
  raised.

Each text's score for a category is the **maximum across its chunks**, so
scanning is conservative: one bad chunk flags the text.

Empty or whitespace-only text produces no chunks, so every category scores
`0.0`:

```python
guard.scan("")        # Result with {"unsafe": 0.0} (prompt guard)
guard.scan("   ")     # same — no chunks are produced
```

## Batching

Batch methods are the primitives: they flatten the chunks of every input text
into a **single pipeline call**, which is much faster than scanning one text
at a time.

```python
results = guard.batch_scan(["text one", "text two", "text three"])
# results[0] is the Result for "text one", and so on — order is preserved
```

The single-text `scan()` is just an adapter that delegates to `batch_scan`
with a one-element batch, so both produce identical scores.

## Async

The `async_*` variants never block the event loop — the pipeline runs in a
thread executor:

```python
result = await guard.async_scan("some text")
results = await guard.async_batch_scan(["text one", "text two"])
```

Use them inside async web frameworks (FastAPI, discord bots, and so on) so a
slow model call does not stall other requests.

## The prediction cache

Repeated text is cheap: `_predict` de-duplicates chunks within a batch and
caches results by exact chunk text in an LRU cache (default capacity 1024
entries, one cache per guard instance).

```python
guard.scan("the same message posted ten times")  # first call runs the model

print(guard.cache_info())
# {"hits": 12, "misses": 3, "size": 3, "maxsize": 1024}

guard.cache_clear()  # reset the cache and its statistics
```

You can disable or re-enable caching at runtime:

```python
import mezzoguard

mezzoguard.config.cache_enabled = False  # every call hits the model
mezzoguard.config.cache_enabled = True   # back to caching (default)
```

## Token utilities

Guards expose the tokenizer for quick checks:

```python
tokens = guard.tokenize("some text to inspect")
print(len(tokens))               # or: guard.get_token_length("some text to inspect")
```

Both load the model first if it was ejected.

## Memory management

```python
guard.eject_model()   # drop the pipeline and empty the CUDA cache
guard.load_model()    # load it again (idempotent if already loaded)
```

Or use the context manager shown above. More precision/backend options are on
the [Performance]({{ '/performance/' | relative_url }}) page.

## See also

- [Policies & Thresholds]({{ '/policies/' | relative_url }}) — decide *what counts* as unsafe.
- [Redaction]({{ '/redaction/' | relative_url }}) — replace flagged regions instead of just flagging them.
- [Models]({{ '/models/' | relative_url }}) — every supported model and custom configurations.
- [Guard API]({{ '/api/guard/' | relative_url }}) — exact signatures.
