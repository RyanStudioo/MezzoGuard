---
layout: page
title: APIGuard API
---

Reference for `mezzoguard.APIGuard` — the remote client. Guide:
[Remote Client]({{ '/remote-client/' | relative_url }}).

```python
from mezzoguard import APIGuard
```

## Constructor

```python
APIGuard(base_url: str, timeout: float = 30.0)
```

| Parameter | Meaning |
|--------|-----------|
| `base_url` | Root URL of a running Mezzo Guard server, e.g. `"http://127.0.0.1:8000"`. A trailing slash is stripped. |
| `timeout` | Per-request timeout in seconds. |

Requires the `client` extra (`pip install "mezzo-guard[client]"`); httpx is
imported lazily and a missing install raises `RuntimeError` with that hint.

## Attributes

| Attribute | Meaning |
|-----------|---------|
| `base_url`, `timeout` | Constructor options. |
| `info` | `dict` from the server's `GET /v1/info` — model, categories, backend, dtype, device. Fetched lazily, then cached. |

## Scanning

```python
scan(text: str, max_seq_length: int = 64, overlap: int = 16) -> Result
batch_scan(texts: list[str], max_seq_length: int = 64, overlap: int = 16) -> list[Result]
async_scan(text: str, ...) -> Result
async_batch_scan(texts: list[str], ...) -> list[Result]
```

Each batch is one `POST /v1/scan` request. The server also returns raw chunk
predictions, so `result.chunks` is populated remotely just like locally.

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

A `replace` dict is sent as per-category `replacements`; a `Policy` is sent as
`policy` plus its `default_threshold`. Results are rebuilt as real `Span`
objects with offsets into the original text.

## Decorators

`scan_before_exec` and `redact_before_exec` are inherited from the shared
`GuardBase` interface and work exactly like the local guard's:

```python
@guard.scan_before_exec(param="prompt", confidence=0.5)
def chat(prompt: str) -> str: ...
```

## Lifecycle

```python
__enter__ / __exit__                 # sync context manager
__aenter__ / __aexit__               # async context manager

close() -> None                      # close the sync client
aclose() -> coroutine                # close the async + sync clients
```

The async httpx client is created lazily on the first `async_*` call.

## Errors

Every non-2xx response raises
[`MezzoGuardServerError`]({{ '/api/errors/' | relative_url }}) with the
server's status code and detail.

## See also

- [Remote Client]({{ '/remote-client/' | relative_url }}) — the guide.
- [HTTP Server]({{ '/server/' | relative_url }}) — the server being called.
- [Guard API]({{ '/api/guard/' | relative_url }}) — the local counterpart.
