---
layout: page
title: Redaction
---

Sometimes you want to remove the harmful parts instead of rejecting the whole
input. `redact()` replaces flagged regions with a marker while preserving
everything else exactly — no duplicated text, no lost formatting.

## How redaction works

Redaction reuses the [chunking]({{ '/scanning/' | relative_url }}) machinery:

1. The text is split into overlapping chunks (`max_seq_length`, `overlap`).
2. Each chunk is evaluated against your threshold or `Policy`.
3. A chunk containing *any* violation is a redaction candidate, tagged with its
   highest-scoring violated category.
4. **Consecutive flagged chunks are merged** into a single replacement, so one
   bad region becomes one marker instead of many.
5. Overlap regions are emitted only once, so unflagged text survives
   byte-for-byte.

## Basic usage

```python
redacted = guard.redact(long_text)
print(redacted)
```

Flagged runs become `[REDACTED]`; everything else is untouched.

## Getting the exact positions

Pass `return_spans=True` to also get the character offsets of every replaced
region:

```python
redacted, spans = guard.redact(long_text, return_spans=True)

for span in spans:
    print(span.start, span.end, span.category, span.replacement)
    print(long_text[span.start:span.end])   # the original flagged text
```

Each span is a `Span`:

| Field | Meaning |
|-------|---------|
| `start` | Character offset where the flagged run begins (in the **original** text). |
| `end` | Character offset where it ends (exclusive). |
| `category` | The highest-scoring violated category for the run. |
| `replacement` | The marker that was inserted. |

Because `start`/`end` index into the original text, you can show users exactly
what was removed, log the removed content separately, or highlight it in a UI.

## Custom replacements

A single string for everything:

```python
guard.redact(text, replace="[BLOCKED]")
```

Or a per-category mapping — categories without an entry fall back to
`[REDACTED]`:

```python
guard.redact(
    text,
    replace={
        "violence": "[VIOLENCE]",
        "sexual": "[SEXUAL]",
    },
)
```

## Choosing what gets flagged

Redaction decides per chunk with the same threshold logic as scanning. When
you pass no `policy`, one is built from `confidence` (default 0.5):

```python
guard.redact(text, confidence=0.7)                       # flag chunks at 0.7
guard.redact(text, policy=Policy(default_threshold=0.5)
                     .add_threshold("violence", 0.7))   # per-category control
```

## Tuning chunk size

Because redaction works per chunk, `max_seq_length` controls how *surgical* it
is: smaller chunks replace less text around a violation.

```python
guard.redact(text, max_seq_length=32, overlap=8)   # finer-grained redaction
guard.redact(text, max_seq_length=128, overlap=32) # fewer, larger chunks
```

The constraint is the same as for scanning: `overlap` must be smaller than
`max_seq_length`.

## Batch and async

```python
redacted_list = guard.batch_redact(["text one", "text two"])

redacted_list, spans_lists = guard.batch_redact(
    ["text one", "text two"],
    return_spans=True,
)
# spans_lists[i] holds the spans for "text one", etc.

redacted_list = await guard.async_batch_redact(texts)
```

`batch_redact` matches per-text `redact` calls exactly, and the `async_*`
variants never block the event loop.

## Putting it together

A realistic moderation flow — clean platform markup, then redact what the
model flags, keeping the spans for the audit log:

```python
from mezzoguard import AutoCleaner, Guard

cleaner = AutoCleaner()
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")

clean_text = cleaner.clean(raw_user_message)
redacted, spans = guard.redact(clean_text, return_spans=True)

if spans:
    log.warning("redacted %d region(s): %s", len(spans), [(s.start, s.end) for s in spans])
```

See the [Text Cleaner]({{ '/text-cleaner/' | relative_url }}) page for what
`AutoCleaner` does to the message first.

## See also

- [Scanning]({{ '/scanning/' | relative_url }}) — chunking and scoring details.
- [Policies & Thresholds]({{ '/policies/' | relative_url }}) — threshold semantics.
- [Guarding Functions]({{ '/function-guards/' | relative_url }}) — redact parameters automatically.
- [Guard API]({{ '/api/guard/' | relative_url }}) — exact signatures.
