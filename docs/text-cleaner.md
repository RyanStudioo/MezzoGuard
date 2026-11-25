---
layout: page
title: Text Cleaner
---

Before text reaches a model, `mezzoguard.cleaner` normalizes it and masks
platform-specific references. Two reasons this matters:

1. **Normalization defeats tricks.** Homoglyphs (`＠everyone`), invisible
   characters, and zero-width joiners smuggled into text are folded away, so a
   guard sees the message as it reads.
2. **Masking protects people and focuses the model.** Usernames, pings, and
   links become short synthetic tokens, so the model scores *content* rather
   than getting distracted by (or learning) handles.

The cleaner is independent of guards — it is plain regex/Unicode processing
with no model involved.

## The five-stage pipeline

Every cleaner runs the same fixed pipeline; only the platform rule block
differs:

1. **Cap** the input at 4,000 code points.
2. **NFKC normalization** — fullwidth and compatibility forms fold to their
   plain equivalents.
3. **Invisible-character removal** — format characters are stripped, except
   emoji ZWJ sequences and tag sequences (so 👨‍👩‍👧‍👦 survives intact).
4. **Whitespace collapse** — every whitespace run becomes a single space.
5. **One masking pass** over a single combined alternation, then a final
   collapse and trim.

The single-pass design is what makes cleaners **idempotent**: emitted tokens
are never re-scanned by later rules, so cleaning an already-clean output is a
no-op.

## Output tokens

Only five synthetic tokens are ever emitted:

| Token | Replaces | Example |
|-------|----------|---------|
| `[LINK]` | URLs, autolinks, Markdown link destinations, email addresses | `https://example.com` |
| `@ping` | User mentions (all platforms) | `<@123>`, `u/alice`, `@bob` |
| `@mass` | Mass pings | `@everyone`, `@here` |
| `#chat` | Channels and subreddits | `<#456>`, `r/python` |
| `:emoji:` | Custom server emoji and unknown shortcodes | `<:name:123>` |

Real Unicode emoji — including ZWJ sequences, skin-tone modifiers, keycaps,
and flags — are preserved as-is. Known shortcodes (`:smile:`) are preserved
using the bundled gemoji table; unknown ones (`:notarealemoji:`) become
`:emoji:`.

## Choosing a cleaner

```python
from mezzoguard import AutoCleaner, DiscordCleaner, RedditCleaner, TextCleaner, TwitterCleaner
```

| Cleaner | Platform rules it adds |
|---------|------------------------|
| `TextCleaner` | None — links, emails, and shortcodes only. |
| `DiscordCleaner` | Mass pings, role/user/channel mentions, custom emoji. |
| `RedditCleaner` | `u/` and `r/` references (with or without a leading slash). |
| `TwitterCleaner` | `@handle` references. |
| `AutoCleaner` | All of the above, in specificity order. |

### Discord

```python
from mezzoguard import DiscordCleaner

DiscordCleaner().clean("<@123> <#456> @everyone <:blob:789>")
# '@ping #chat @mass :emoji:'
```

| Input | Output | Rule |
|-------|--------|------|
| `<@123>`, `<@!123>`, `<@&456>` (role) | `@ping` | User/role mentions |
| `<#456>` | `#chat` | Channel references |
| `@everyone`, `@here` | `@mass` | Mass pings |
| `<:name:789>`, `<a:name:789>` | `:emoji:` | Custom server emoji |
| `<t:1740000000:R>`, `</command:name>`, `<id:foo>` | unchanged | Protected markup — consumed so inner rules leave it alone |

Bare `@name` and `#channel` text is deliberately **left alone**: it is
indistinguishable from prose, and Discord's real markup is always
angle-bracketed.

### Reddit

```python
from mezzoguard import RedditCleaner

RedditCleaner().clean("see /u/alice and r/python")
# 'see @ping and #chat'
```

| Input | Output | Rule |
|-------|--------|------|
| `u/alice`, `/u/alice` | `@ping` | User references |
| `r/python`, `/r/python` | `#chat` | Subreddit references |

`@name` is not a native Reddit mention, so it is untouched; ordinary Markdown
syntax is preserved.

### Twitter / X

```python
from mezzoguard import TwitterCleaner

TwitterCleaner().clean("hi @alice https://x.com/@bob")
# 'hi @ping [LINK]'
```

| Input | Output | Rule |
|-------|--------|------|
| `@alice` (handles up to 15 chars) | `@ping` | Handles |
| `#hashtag`, `$ticker` | unchanged | Deliberately preserved |

Note how `https://x.com/@bob` becomes `[LINK]` and never a handle — links and
emails are always matched *before* platform mentions.

### Auto

`AutoCleaner` applies every platform rule block in one pass, ordered by
specificity: Discord angle markup and mass pings first, then Reddit references,
then Twitter handles. It does not need platform detection:

```python
from mezzoguard import AutoCleaner

AutoCleaner().clean("<@123> u/alice r/python @bob @everyone")
# '@ping @ping #chat @ping @mass'
```

The ordering is why `<@123>` is a Discord mention and `@everyone` a mass ping
rather than a Twitter handle — the most specific rule wins.

## Links and emails

Link masking is shared by every cleaner:

| Input | Output |
|-------|--------|
| `see [docs](https://example.com/x) now` | `see [docs]([LINK]) now` |
| `![alt text](https://example.com/a.png)` | `![alt text]([LINK])` |
| `<https://example.com>` | `[LINK]` |
| `go to https://example.com.` | `go to [LINK].` — trailing punctuation is preserved |
| `mail foo@example.com now` | `mail [LINK] now` |
| `www.example.com/path` | `[LINK]` |

Email addresses always collapse to the same `[LINK]` token as URLs. Mentions
inside Markdown link text are still masked — the link *destination* is
replaced while the text is cleaned with the non-Markdown rules.

## Convenience functions

```python
from mezzoguard import clean_auto, clean_discord, clean_reddit, clean_text, clean_twitter

clean_text(raw, platform="discord")  # "generic", "auto"/"all", "discord", "reddit", "twitter"/"x"
clean_auto(raw)
clean_discord(raw)
clean_reddit(raw)
clean_twitter(raw)
```

`clean_text(text, platform)` accepts the same names as `get_cleaner`, which
builds a cleaner instance from a name:

```python
from mezzoguard.cleaner import get_cleaner

cleaner = get_cleaner("reddit")   # RedditCleaner()
cleaner = get_cleaner("x")        # TwitterCleaner()
get_cleaner("nope")               # ValueError listing supported platforms
```

Cleaners are cheap to build (rule layouts are compiled once and cached), so
the convenience functions building a fresh cleaner per call are fine. For hot
paths, keep one instance:

```python
cleaner = AutoCleaner()
for message in messages:
    process(cleaner.clean(message))
```

## Combining with a guard

Clean, then scan or redact:

```python
from mezzoguard import AutoCleaner, Guard

cleaner = AutoCleaner()
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")

result = guard.scan(cleaner.clean(raw_message))
redacted = guard.redact(cleaner.clean(raw_message))
```

## Good to know

- Input is capped at 4,000 code points **before** masking (the `MAX_TEXT_LENGTH`
  constant).
- Non-string input raises `TypeError`.
- Cleaning is idempotent — tokens are opaque to the rules.
- Cleaners take no configuration: emails always become `[LINK]`, shortcodes
  always use the bundled table, and Discord bare `@name` text is always left
  alone.

## See also

- [Scanning]({{ '/scanning/' | relative_url }}) and [Redaction]({{ '/redaction/' | relative_url }}) — what to do after cleaning.
- [Cleaner API]({{ '/api/cleaner/' | relative_url }}) — exact signatures.
