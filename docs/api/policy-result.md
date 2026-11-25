---
layout: page
title: Policy & Results API
---

Reference for the evaluation types: `Policy`, `Result`, `PolicyResult`, and
`Span`. Guide: [Policies & Thresholds]({{ '/policies/' | relative_url }}).

```python
from mezzoguard import Policy, PolicyResult, Result, Span
```

## Policy

Per-category thresholds with a default fallback.

```python
Policy(default_threshold: float = 0.0)
```

| Member | Signature | Meaning |
|--------|-----------|---------|
| `default_threshold` | attribute | Fallback for categories without an explicit threshold. **Defaults to 0.0** — pass `0.5` for partial policies. |
| `add_threshold` | `(category: str, threshold: float) -> Policy` | Set a category's threshold; returns `self` for chaining. |
| `get_threshold` | `(category: str) -> float` | The category's threshold, or the default. |
| `to_dict` | `() -> dict[str, float]` | Explicit thresholds only — the wire format. |
| `evaluate` | `(result: Result) -> PolicyResult` | Evaluate every category of `result`. |

Comparison is inclusive: a category is violated when `score >= threshold`.

```python
policy = (
    Policy(default_threshold=0.5)
    .add_threshold("violence", 0.7)
    .add_threshold("unsafe", 0.9)
)

policy.get_threshold("violence")   # 0.7
policy.get_threshold("toxic")      # 0.5 (default)
policy.to_dict()                   # {"violence": 0.7, "unsafe": 0.9}
```

## Result

What `scan()` returns.

```python
Result(chunks: list[list[dict]] | list[dict], scores: dict[str, float])
```

| Member | Signature | Meaning |
|--------|-----------|---------|
| `scores` | attribute | `dict` with every configured category, `0.0`–`1.0`; per category the max across chunks. |
| `chunks` | property | Raw per-chunk predictions: one entry per chunk, each a list of `{"label": ..., "score": ...}` dicts. |
| `is_unsafe` | `(threshold: float = 0.5, policy: Policy | None = None) -> bool` | Any score meeting the threshold — or the policy's verdict. |
| `get_violated_categories` | `(policy: Policy | None = None, threshold: float = 0.5) -> list[str]` | Categories meeting the threshold (or policy). |
| `violated` | same as `get_violated_categories` | Alias. |
| `evaluate` | `(policy: Policy) -> PolicyResult` | Full policy evaluation. |

A supplied `policy` always takes precedence over the plain `threshold`.

```python
result = guard.scan(text)
result.is_unsafe()                          # any score >= 0.5
result.is_unsafe(threshold=0.8)             # any score >= 0.8
result.is_unsafe(policy=strict_policy)      # policy verdicts
result.get_violated_categories(policy=policy)
```

## PolicyResult

What a policy evaluation returns.

```python
PolicyResult(scores: dict[str, float], violated: dict[str, bool], categories: list[str])
```

| Member | Signature | Meaning |
|--------|-----------|---------|
| `scores` | attribute | The scores that were evaluated. |
| `violated` | attribute | `dict` of every category mapped to its violated flag. |
| `categories` | attribute | The evaluated category names. |
| `is_safe` | `() -> bool` | No category is violated. |
| `is_unsafe` | `() -> bool` | Any category is violated. |
| `get_violated_categories` | `() -> list[str]` | Just the violated names. |
| `__bool__` | — | Same as `is_unsafe()` — `if verdict:` works. |
| `__repr__` | — | `PolicyResult(safe=[...], violated=[...])`. |

```python
verdict = policy.evaluate(result)
if verdict:
    print(verdict.get_violated_categories())
```

## Span

A flagged region produced by redaction with `return_spans=True`.

```python
Span(start: int, end: int, category: str | None = None, replacement: str = "")
```

| Field | Meaning |
|-------|---------|
| `start` | Character offset where the flagged run begins — indexes into the **original** text. |
| `end` | Character offset where it ends (exclusive). |
| `category` | The highest-scoring violated category for the run. |
| `replacement` | The marker that was inserted. |

`original[span.start:span.end]` is always the exact text that was replaced.

```python
redacted, spans = guard.redact(text, return_spans=True)
for span in spans:
    print(span.category, text[span.start:span.end], "->", span.replacement)
```

## See also

- [Policies & Thresholds]({{ '/policies/' | relative_url }}) — the guide.
- [Redaction]({{ '/redaction/' | relative_url }}) — where spans come from.
- [Guard API]({{ '/api/guard/' | relative_url }}) — the methods that produce these types.
