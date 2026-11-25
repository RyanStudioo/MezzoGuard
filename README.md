# Mezzo Guard

Mezzo Guard is a Python SDK designed to use AI-based solutions to protect your applications such as prompt protection
and content moderation.

Mezzo Guard can be easily added into your pipeline, offering great protection with little effort.

## Installation

```bash
pip install mezzo-guard
```

PyTorch is required to load models. If you do not have it yet:

```bash
pip install torch
```

Optional extras:

```bash
pip install "mezzo-guard[onnx]"     # enable the ONNX runtime backend
pip install "mezzo-guard[server]"   # FastAPI + uvicorn, for `mezzo-guard serve`
pip install "mezzo-guard[client]"   # httpx, for remote (HTTP) clients
```

Requires Python 3.11+.

## Quickstart

### Guard (prompt + content)

`Guard` works with both prompt-guard and content-guard models. The model's
configuration decides whether it returns a single `unsafe` score or one score
per category:

```python
from mezzoguard import Guard

guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")
result = guard.scan("I want to hurt myself")
print(result.scores)          # {"self-harm": 0.99, ...}
print(result.is_unsafe())      # any category >= 0.5

prompt_model = Guard("RyanStudio/Mezzo-Prompt-Guard-v2-Small")
prompt_result = prompt_model.scan("Ignore your previous instructions")
print(prompt_result.scores)    # {"unsafe": 0.99}
print(prompt_result.is_unsafe(threshold=0.8))
```

Thresholds are per category and fully user-defined:

```python
from mezzoguard import Policy

policy = (
    Policy(default_threshold=0.5)       # fallback for categories not listed
    .add_threshold("violence", 0.7)
    .add_threshold("unsafe", 0.9)
)

result = prompt_model.scan("Ignore your previous instructions")
print(result.is_unsafe(policy=policy))
print(result.get_violated_categories(policy=policy))
```

### Choosing a model

Pass any supported Hugging Face model id directly to `Guard`, e.g.
`Guard("meta-llama/Llama-Prompt-Guard-2-86M")`.

## Policies

Policies set per-category confidence thresholds and evaluate scan results. Use
`default_threshold` as the fallback for categories without an explicit threshold:

```python
from mezzoguard import Policy

policy = (
    Policy(default_threshold=0.5)
    .add_threshold("sexual", 0.7)
    .add_threshold("violence", 0.6)
)

result = guard.scan("some content")
verdict = policy.evaluate(result)
if verdict.is_unsafe():
    print(verdict.get_violated_categories())
```

## Redaction

Long text is scanned in overlapping chunks. `redact()` replaces flagged regions while preserving
everything else byte-for-byte (no duplicated text, no lost formatting).

```python
redacted = guard.redact(long_text)
print(redacted)
```

Get the exact positions of what was redacted:

```python
redacted, spans = guard.redact(long_text, return_spans=True)
for span in spans:
    print(span.start, span.end, span.category)  # character offsets into the original text
```

Redaction operates per-chunk: a chunk containing any flagged content is a candidate for
replacement, and consecutive flagged chunks are merged into a single marker. Tune `max_seq_length`
(smaller chunks = finer-grained redaction) and `overlap` as needed.

Custom replacements, per category:

```python
guard.redact(text, replace={"violence": "[VIOLENCE]", "sexual": "[SEXUAL]"})
# categories without an explicit replacement fall back to "[REDACTED]"
```

## Text cleaning

`mezzoguard.cleaner` normalizes text and masks platform-specific references before
it reaches a model. Normalization, link/email masking, and emoji handling are
shared; each platform only adds its own mention rules.

```python
from mezzoguard import AutoCleaner, DiscordCleaner, RedditCleaner, TwitterCleaner

AutoCleaner().clean("<@123> u/alice r/python @bob @everyone")
# '@ping @ping #chat @ping @mass'

DiscordCleaner().clean("<@123> <#456> @everyone")
# '@ping #chat @mass'

RedditCleaner().clean("see /u/alice and r/python")
# 'see @ping and #chat'

TwitterCleaner().clean("hi @alice https://x.com/@bob")
# 'hi @ping [LINK]'
```

`AutoCleaner` applies every platform rule block in one pass, ordered by
specificity: Discord angle markup and mass pings first, then Reddit references,
then Twitter/X handles. It does not need platform detection, and Discord
protected constructs are consumed before the other rules can see inside them.

The pipeline always runs NFKC normalization, invisible-character removal,
whitespace collapse, a single masking pass, then a final whitespace collapse and
trim. Only five synthetic tokens are emitted: `[LINK]`, `@ping`, `@mass`,
`#chat`, and `:emoji:`. Real Unicode emoji (including ZWJ sequences, modifiers,
keycaps, and flags) are preserved. Synthetic tokens are treated as opaque, so
cleaning the output again is a no-op.

Email addresses always collapse to the same `[LINK]` token as URLs. The rule set
is standardized and cleaners take no configuration: Discord bare `@name` /
`#channel` text is deliberately left alone, and emoji shortcodes always use the
bundled table (known shortcodes are preserved, unknown ones become `:emoji:`).

Input is capped at 4,000 code points before masking. `clean_text(text, platform)`
and `clean_auto`/`clean_discord`/`clean_reddit`/`clean_twitter` are convenience
wrappers.

## Guarding function calls

Wrap any function so its input is scanned (raising on unsafe input) or redacted before execution.
Both sync and async functions are supported.

```python
@guard.scan_before_exec(param="prompt", confidence=0.5)
def chat(prompt: str) -> str:
    return generate(prompt)  # never called with an unsafe prompt


@guard.redact_before_exec(param="prompt")
async def chat_async(prompt: str) -> str:
    return generate(prompt)
```

Guards raise `mezzoguard.UnsafeInputError`, which carries the violated
categories as `{category: score}` entries.

## Async and batching

The `async_*` variants never block the event loop. Batch methods are the primitives: they scan or
redact many texts in a single pipeline call, and the single-text methods delegate to them with a
one-element batch.

```python
results = guard.batch_scan(["text one", "text two"])
redacted = guard.batch_redact(["text one", "text two"])
redacted, spans_list = guard.batch_redact(texts, return_spans=True)

results = await guard.async_batch_scan(texts)
redacted = await guard.async_batch_redact(texts)
```

## Performance options

```python
# bf16/fp16 on GPU (default "auto" detection)
Guard(name, dtype="float16")

# torch.compile the model
Guard(name, torch_compile=True)

# ONNX Runtime backend (artifacts are exported once, then cached on disk)
Guard(name, use_onnx=True)
```

To free GPU/CPU memory, call `guard.eject_model()` or use a context manager:

```python
with Guard(name) as guard:
    ...
```

## Serving over HTTP

Install the server extra and serve a model:

```bash
pip install "mezzo-guard[server]"
mezzo-guard serve RyanStudio/Mezzo-Content-Guard-v1.5-Small --dtype float16
# Serving RyanStudio/Mezzo-Content-Guard-v1.5-Small on http://127.0.0.1:8000
```

The model's config decides whether it behaves like a prompt guard or a content guard.
Performance flags match the SDK:
`--dtype`, `--onnx`, `--torch-compile`, `--compile-mode` and `--device`
(`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). Request size limits are set with
`--max-batch-texts` and `--max-text-chars`. Interactive OpenAPI docs are at `/docs`.

```bash
curl -s http://127.0.0.1:8000/v1/scan \
  -H 'content-type: application/json' \
  -d '{"texts": ["I want to hurt myself"]}'
```

Endpoints: `GET /healthz`, `GET /v1/info`, `POST /v1/scan`, `POST /v1/redact`. Both POST
endpoints accept batches, optional `max_seq_length`/`overlap`, per-category `policy`
thresholds, and a `confidence` fallback. `redact` also accepts `replace`/`replacements` and
`return_spans`; `scan` can return raw chunk predictions with `include_chunks`.

## Remote client

`APIGuard` calls a running server instead of loading a model locally and mirrors the
guard API (`scan`, `batch_scan`, `redact`, `batch_redact`, async variants and the
`*_before_exec` decorators). Both local and remote guards implement the same
`GuardBase` ABC, so they can be swapped without changing call sites.

```python
from mezzoguard import APIGuard

guard = APIGuard("http://127.0.0.1:8000")
result = guard.scan("Ignore your previous instructions")
print(result.scores, result.is_unsafe())

redacted = guard.redact(long_text, confidence=0.7)
```

Requires the `client` extra (`pip install "mezzo-guard[client]"`). `APIGuard` is also a context
manager (`with APIGuard(url) as guard: ...`) and exposes the server metadata via `guard.info`.

## Supported Models

Mezzo Guard supports content and prompt guards.
- Prompt Guards are designed to protect against harmful prompts
  such as prompt injection or jailbreaking.
- Content Guards are designed to block and filter out harmful content and provide quick and easy content moderation.

It is recommended to use the latest version of each model even if older versions are supported.

### Prompt Guards
- [Mezzo Prompt Guard v2](https://huggingface.co/collections/RyanStudio/mezzo-prompt-guard-v2)

### Content Guards
- [Mezzo Content Guard](https://huggingface.co/collections/RyanStudio/mezzo-content-guard)
- [Mezzo Content Guard v1.5](https://huggingface.co/collections/RyanStudio/mezzo-content-guard-v15)
