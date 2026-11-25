---
layout: page
title: Remote Client
---

`APIGuard` calls a running [Mezzo Guard server]({{ '/server/' | relative_url }})
instead of loading a model locally — and it mirrors the local `Guard` API
exactly, so call sites never change when you move from in-process to a
service.

## Install

```bash
pip install "mezzo-guard[client]"
```

httpx is imported lazily, so the rest of the SDK works without it.

## Connect

```python
from mezzoguard import APIGuard

guard = APIGuard("http://127.0.0.1:8000")           # timeout defaults to 30s
guard = APIGuard("http://10.0.0.5:8000", timeout=10.0)

result = guard.scan("Ignore your previous instructions")
print(result.scores)   # {"unsafe": 0.99} — a prompt model on the server
print(result.is_unsafe())
```

The server's metadata is one property away (fetched once, then cached):

```python
print(guard.info)
# {'model': ..., 'categories': [...], 'backend': 'torch', 'dtype': ..., 'device': ...}
```

## The same API as the local Guard

Both `Guard` and `APIGuard` implement the `GuardBase` interface:

| Local | Remote | Notes |
|-------|--------|-------|
| `guard.scan(text)` | `guard.scan(text)` | Identical. |
| `guard.batch_scan(texts)` | `guard.batch_scan(texts)` | One HTTP request per batch. |
| `await guard.async_scan(text)` | `await guard.async_scan(text)` | httpx `AsyncClient` under the hood. |
| `await guard.async_batch_scan(texts)` | `await guard.async_batch_scan(texts)` | |
| `guard.redact(text, ...)` | `guard.redact(text, ...)` | All redaction options supported. |
| `guard.batch_redact(texts, ...)` | `guard.batch_redact(texts, ...)` | |
| `await guard.async_redact(...)` | `await guard.async_redact(...)` | |
| `await guard.async_batch_redact(...)` | `await guard.async_batch_redact(...)` | |
| `guard.scan_before_exec(...)` | `guard.scan_before_exec(...)` | Decorators work unchanged. |
| `guard.redact_before_exec(...)` | `guard.redact_before_exec(...)` | |

The remote methods rebuild real `Result` and `Span` objects, so everything
downstream — `result.scores`, `result.is_unsafe(policy=...)`, span offsets —
behaves exactly like the local guard:

```python
redacted, spans = guard.redact(long_text, return_spans=True, confidence=0.7)
for span in spans:
    print(long_text[span.start:span.end], "->", span.replacement)
```

Policies serialize transparently, including the default threshold:

```python
from mezzoguard import Policy

policy = Policy(default_threshold=0.5).add_threshold("violence", 0.7)
redacted = guard.redact(text, policy=policy)
```

## Decorators work too

The [guarding decorators]({{ '/function-guards/' | relative_url }}) run over
HTTP with no changes:

```python
@guard.scan_before_exec(param="prompt", confidence=0.5)
def chat(prompt: str) -> str:
    return generate(prompt)
```

Your web app stays torch-free; the model lives in the server process.

## Lifecycle

`APIGuard` is a sync **and** async context manager:

```python
with APIGuard("http://127.0.0.1:8000") as guard:
    guard.scan("some text")
# underlying httpx client closed on exit

async with APIGuard("http://127.0.0.1:8000") as guard:
    await guard.async_scan("some text")
```

Manual control:

```python
guard.close()       # sync client
await guard.aclose()  # async + sync clients
```

The async client is created lazily on the first `async_*` call.

## Error handling

Any non-2xx response raises `MezzoGuardServerError`, which keeps the status
code and the server's detail:

```python
from mezzoguard import APIGuard, MezzoGuardServerError

guard = APIGuard("http://127.0.0.1:8000")

try:
    guard.batch_scan(["too", "many", "texts", "..."])   # exceeds server limits
except MezzoGuardServerError as exc:
    print(exc.status_code)   # 413
    print(exc.detail)        # "Batch too large: 200 texts (max 64)"
```

The server's `413` batch/length limits and `422` validation errors are
documented on the [HTTP Server]({{ '/server/' | relative_url }}) page.

## Swapping local for remote

Because both guards share `GuardBase`, functions written against the
interface accept either:

```python
from mezzoguard import APIGuard, Guard, GuardBase

def moderate(guard: GuardBase, message: str) -> str:
    result = guard.scan(message)
    if result.is_unsafe():
        return "rejected"
    return guard.redact(message)

moderate(Guard("RyanStudio/Mezzo-Content-Guard-Small"), message)   # in-process
moderate(APIGuard("http://guard.internal:8000"), message)          # via service
```

Typical split: heavy GPU models live on one box behind `mezzo-guard serve`;
everywhere else, `APIGuard`.

## See also

- [HTTP Server]({{ '/server/' | relative_url }}) — run the other side.
- [Guarding Functions]({{ '/function-guards/' | relative_url }}) — decorators over HTTP.
- [APIGuard API]({{ '/api/apiguard/' | relative_url }}) — exact signatures.
