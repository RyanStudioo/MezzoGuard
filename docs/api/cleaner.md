---
layout: page
title: Cleaner API
---

Reference for `mezzoguard.cleaner` — text normalization and platform masking.
Guide: [Text Cleaner]({{ '/text-cleaner/' | relative_url }}).

```python
from mezzoguard import AutoCleaner, DiscordCleaner, RedditCleaner, TextCleaner, TwitterCleaner
from mezzoguard import clean_auto, clean_discord, clean_reddit, clean_text, clean_twitter
```

## Classes

Every cleaner exposes exactly one method, `clean`, plus a `platform` class
attribute.

### TextCleaner

```python
TextCleaner().clean(text: str) -> str
```

The platform-independent base: NFKC normalization, invisible-character
removal, whitespace collapse, and masking of links, emails, autolinks,
Markdown link destinations, and unknown emoji shortcodes. No mentions are
masked.

`platform` is `"generic"`. Raises `TypeError` for non-string input.

### DiscordCleaner / RedditCleaner / TwitterCleaner

Subclasses that add one platform rule block each:

| Class | `platform` | Extra rules |
|-------|-----------|-------------|
| `DiscordCleaner` | `"discord"` | `@everyone`/`@here` → `@mass`; `<@id>`, `<@!id>`, `<@&id>` → `@ping`; `<#id>` → `#chat`; custom emoji `<:name:id>` → `:emoji:`; timestamps, slash commands, and `<id:...>` markup preserved. |
| `RedditCleaner` | `"reddit"` | `u/name`, `/u/name` → `@ping`; `r/name`, `/r/name` → `#chat`. |
| `TwitterCleaner` | `"twitter"` | `@handle` (1–15 chars) → `@ping`; hashtags and cashtags preserved. |

### AutoCleaner

```python
AutoCleaner().clean(text: str) -> str
```

`platform` is `"auto"`. Every platform rule block in one pass, ordered by
specificity: Discord → Reddit → Twitter. No platform detection needed.

## Convenience functions

```python
clean_text(text: str, platform: str = "generic") -> str   # build + run in one call
clean_auto(text: str) -> str
clean_discord(text: str) -> str
clean_reddit(text: str) -> str
clean_twitter(text: str) -> str

get_cleaner(platform: str = "generic") -> TextCleaner      # mezzoguard.cleaner import
```

`get_cleaner` accepts: `"generic"`, `"auto"`, `"all"`, `"discord"`,
`"reddit"`, `"twitter"`, `"x"` (case-insensitive). Unknown names raise
`ValueError` listing the supported platforms.

## Tokens

The five synthetic output tokens, importable from the top level:

| Name | Value |
|------|-------|
| `LINK_TOKEN` | `"[LINK]"` |
| `USER_MENTION_TOKEN` | `"@ping"` |
| `MASS_PING_TOKEN` | `"@mass"` |
| `CHANNEL_TOKEN` | `"#chat"` |
| `EMOJI_TOKEN` | `":emoji:"` |

## Constants

| Name | Value | Meaning |
|------|-------|---------|
| `MAX_TEXT_LENGTH` | `4000` | Input cap in code points, applied before masking. Import from `mezzoguard.cleaner`. |
| `KNOWN_SHORTCODES` | `frozenset[str]` | The bundled gemoji shortcode table; known shortcodes are preserved, unknown ones become `:emoji:`. Import from `mezzoguard.cleaner`. |

## Rule

```python
Rule(name: str, pattern: str, replacement: str | Callable)  # mezzoguard.cleaner import
```

One named alternative in the combined masking alternation. The alternation is
compiled once per rule layout and cached, so building cleaners repeatedly is
cheap.

## Behavior notes

- The pipeline is fixed and order-dependent: cap → NFKC → strip invisibles →
  collapse whitespace → **one** masking pass → final collapse and trim.
- Cleaning is idempotent: emitted tokens are opaque to the rules.
- Emails always collapse to `[LINK]`, exactly like URLs.
- Real Unicode emoji (ZWJ sequences, modifiers, keycaps, flags) are preserved.
- Discord bare `@name` / `#channel` text is deliberately left alone.

## See also

- [Text Cleaner]({{ '/text-cleaner/' | relative_url }}) — the guide with before/after examples.
- [Scanning]({{ '/scanning/' | relative_url }}) — what to do with the cleaned text.
