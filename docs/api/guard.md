---
layout: page
title: Guard API
---

Reference for `mezzoguard.Guard` — the unified local guard — and its category
configuration, `GuardConfig`. For a guided tour see
[Scanning]({{ '/scanning/' | relative_url }}).

```python
from mezzoguard import Guard, GuardConfig
```

## Constructor

```python
Guard(
    name: str,
    config: GuardConfig | dict | None = None,
    **kwargs,              # forwarded to the Model base class
)
```

| Parameter | Type / default | Meaning |
|-----------|----------------|---------|
| `name` | `str` | Hugging Face model id or local model directory. |
| `config` | `GuardConfig` or `dict`, default `None` | Explicit category configuration. A `dict` is converted via `GuardConfig(**config)`. |
| `dtype` | `"auto"` | `"auto"`, `"float32"`, `"float16"`, `"bfloat16"`, or a `torch.dtype`. |
| `torch_compile` | `False` | Compile with `torch.compile` (mutually exclusive with `use_onnx`). |
| `compile_mode` | `"default"` | torch.compile mode. |
| `use_onnx` | `False` | ONNX Runtime backend (requires the `onnx` extra). |
| `device` | `None` | `None`, `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`. |

Constructing a guard loads the model immediately. Category resolution
(preset → `.mezzoguard` file → warning) and deprecation notices are described
on the [Models]({{ '/models/' | relative_url }}) page.

### Attributes

| Attribute | Meaning |
|-----------|---------|
| `name`, `task`, `dtype`, `device`, `torch_compile`, `compile_mode`, `use_onnx` | Constructor options. |
| `pipeline` | The loaded transformers pipeline (or `None` after `eject_model()`). |
| `config` | The resolved `GuardConfig`, or `None` if none was found. |
| `categories` | `list[str]` — the scored categories (empty when `config` is `None`). |

## Scanning

```python
scan(text: str, max_seq_length: int = 64, overlap: int = 16) -> Result
batch_scan(texts: list[str], max_seq_length: int = 64, overlap: int = 16) -> list[Result]
async_scan(text: str, ...) -> Result
async_batch_scan(texts: list[str], ...) -> list[Result]
```

- Texts are split into overlapping token chunks (`max_seq_length` / `overlap`);
  `overlap` must be smaller than `max_seq_length` or `ValueError` is raised.
- `batch_scan` flattens the chunks of every text into a single pipeline call.
- Per category, the reported score is the maximum across the text's chunks.
- Async variants run the pipeline in a thread executor and never block the
  event loop.

Returns a [`Result`]({{ '/api/policy-result/' | relative_url }}).

## Redaction

```python
redact(
    text: str,
    max_seq_length: int = 64,
    overlap: int = 16,
    replace: str | dict[str, str] = "[REDACTED]",
    policy: Policy | None = None,
    confidence: float = 0.5,
    return_spans: bool = False,
) -> str | tuple[str, list[Span]]

batch_redact(texts: list[str], ...) -> list[str] | list[tuple[str, list[Span]]]
async_redact(text: str, ...) -> str | tuple[str, list[Span]]
async_batch_redact(texts: list[str], ...) -> list[str] | list[tuple[str, list[Span]]]
```

- When `policy` is `None`, a `Policy(default_threshold=confidence)` is used.
- `replace` is either one string for every category or a `{category: replacement}`
  dict; categories missing from the dict fall back to `"[REDACTED]"`.
- With `return_spans=True` each output is a `(redacted_text, spans)` tuple;
  span offsets index into the original text.

Semantics and examples: [Redaction]({{ '/redaction/' | relative_url }}).

## Function decorators

```python
scan_before_exec(
    param: str,
    max_seq_length: int = 64,
    overlap: int = 16,
    confidence: float = 0.5,
) -> Callable

redact_before_exec(
    param: str,
    max_seq_length: int = 64,
    overlap: int = 16,
    replace: str = "[REDACTED]",
    policy: Policy | None = None,
    confidence: float = 0.5,
) -> Callable
```

Both detect async functions automatically. `scan_before_exec` raises
`UnsafeInputError` when the parameter's scan violates `confidence`;
`redact_before_exec` rewrites the parameter in place. Guide:
[Guarding Functions]({{ '/function-guards/' | relative_url }}).

## Model utilities

Inherited from the `Model` base class:

| Method | Meaning |
|--------|---------|
| `load_model()` | Load the pipeline (idempotent — no-op if already loaded). Raises `RuntimeError` with an install hint if PyTorch is missing. |
| `eject_model()` | Delete the pipeline and empty the CUDA cache. |
| `tokenize(text)` | Tokenize with the model's tokenizer. |
| `get_token_length(text)` | Number of tokens. |
| `cache_clear()` | Empty the prediction LRU cache and reset its statistics. |
| `cache_info()` | `dict` with `hits`, `misses`, `size`, `maxsize`. |
| `__enter__` / `__exit__` | Context manager — the model is ejected on exit. |

## GuardConfig

```python
from mezzoguard import GuardConfig

config = GuardConfig(categories=["hate-speech", "self-harm", "sexual", "toxic", "violence"])
```

| Member | Meaning |
|--------|---------|
| `GuardConfig(categories)` | Build a config. Any category that normalizes to `"safe"` is dropped. |
| `.categories` | The normalized category list. |
| `.get_category_for_label(label)` | Map a raw model label to a category (normalizing case, `_`, and spaces); raises `ValueError` for unknown labels. |
| `.to_dict()` | `{"categories": [...]}` for serialization. |

## See also

- [Policy & Results]({{ '/api/policy-result/' | relative_url }}) — what scanning returns.
- [Models]({{ '/models/' | relative_url }}) — supported models and configuration resolution.
- [APIGuard]({{ '/api/apiguard/' | relative_url }}) — the same interface over HTTP.
