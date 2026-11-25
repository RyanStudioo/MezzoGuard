"""Platform-independent text cleaning and masking pipeline.

The pipeline is deliberately fixed and order-dependent:

1. NFKC Unicode normalization.
2. Removal of invisible format characters (keeping the emoji ZWJ where it joins
   two pictographs).
3. Whitespace collapse.
4. One single left-to-right masking pass over one combined alternation.
5. Final whitespace collapse and trim.

Masking happens in one left-to-right pass over one combined alternation, so
emitted tokens are never re-scanned by a later rule. Link text is masked with
the same non-Markdown rules inside the Markdown rule's replacement, which keeps
mentions inside link text masked without a second pass over the message. Links
and emails are matched before platform mentions, which is what prevents
``https://x.com/@user`` and ``foo@bar.com`` from producing user-mention tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from .shortcodes import KNOWN_SHORTCODES

LINK_TOKEN = "[LINK]"
USER_MENTION_TOKEN = "@ping"
MASS_PING_TOKEN = "@mass"
CHANNEL_TOKEN = "#chat"
EMOJI_TOKEN = ":emoji:"

MAX_TEXT_LENGTH = 4000

Replacement = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class Rule:
    """One named alternative in the global masking alternation."""

    name: str
    pattern: str
    replacement: Replacement  # literal (group-expanded) or callable


# ---------------------------------------------------------------------------
# Stage 1-3 and 5 helpers: normalization and whitespace
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_ZWJ = "\u200d"
_EXTRA_INVISIBLE = frozenset({"\u034f"})  # combining grapheme joiner (Mn, not Cf)
_EMOJI_TAG_RANGE = (0xE0020, 0xE007F)
_BLACK_FLAG = 0x1F3F4


def _collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and trim the result."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_pictographic(char: str) -> bool:
    """Best-effort Extended_Pictographic check without an extra dependency."""
    codepoint = ord(char)
    if (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x1F1E6 <= codepoint <= 0x1F1FF
        or 0x2300 <= codepoint <= 0x23FF
        or 0x2B00 <= codepoint <= 0x2BFF
        or 0x2190 <= codepoint <= 0x21FF
        or codepoint in {0x3030, 0x303D, 0x3297, 0x3299}
    ):
        return True
    return unicodedata.category(char) == "So"


def _is_emoji_extension(char: str) -> bool:
    """Characters that may sit between an emoji base and a following ZWJ."""
    codepoint = ord(char)
    return (
        codepoint in {0xFE0E, 0xFE0F}
        or 0xE0100 <= codepoint <= 0xE01EF  # variation selectors supplement
        or 0x1F3FB <= codepoint <= 0x1F3FF  # skin-tone modifiers
        or unicodedata.combining(char) != 0
        or unicodedata.category(char) in {"Mn", "Me", "Sk"}
    )


def _has_pictograph_before(chars: list[str]) -> bool:
    index = len(chars) - 1
    while index >= 0 and _is_emoji_extension(chars[index]):
        index -= 1
    return index >= 0 and _is_pictographic(chars[index])


def _strip_invisible(text: str) -> str:
    """Remove invisible format characters, preserving emoji ZWJ sequences.

    Every ``Cf`` character is removed except:

    * a ZWJ that joins two pictographs, so family/role emoji survive; and
    * tag characters inside an emoji tag sequence (subdivision flags), which are
      part of the grapheme even though they are technically format characters.
    """
    output: list[str] = []
    in_emoji_tag_sequence = False

    for index, char in enumerate(text):
        codepoint = ord(char)

        if char == _ZWJ:
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char and _has_pictograph_before(output) and _is_pictographic(next_char):
                output.append(char)
            continue

        if _EMOJI_TAG_RANGE[0] <= codepoint <= _EMOJI_TAG_RANGE[1]:
            if in_emoji_tag_sequence:
                output.append(char)
                if codepoint == _EMOJI_TAG_RANGE[1]:
                    in_emoji_tag_sequence = False
            continue

        if unicodedata.category(char) == "Cf" or char in _EXTRA_INVISIBLE:
            continue

        if codepoint == _BLACK_FLAG:
            in_emoji_tag_sequence = True
        elif codepoint not in {0xFE0E, 0xFE0F} and not 0xE0100 <= codepoint <= 0xE01EF:
            in_emoji_tag_sequence = False

        output.append(char)

    return "".join(output)


# ---------------------------------------------------------------------------
# Stage 4 rules: links, emails, and the platform-independent emoji rule
# ---------------------------------------------------------------------------

# Markdown inline links/images are matched as a whole so that a prose "](...)"
# cannot be mistaken for a link. The callable re-emits the link text after
# masking it with the non-Markdown rules, so mentions inside the text are still
# masked, and replaces the destination/title with the link token.
_MARKDOWN_LINK_PATTERN = (
    r"(?!!?\[LINK\]\()"
    r"(?P<md_prefix>!?\[)"
    r"(?P<md_text>[^\[\]\n]*)"
    r"\]\("
    # Prefer a properly balanced destination, then fall back to "anything up to
    # the next paren" so a malformed destination is still consumed on this pass.
    r"(?:<[^<>\n]*>|[^()\s]+(?:\([^()\s]*\)[^()\s]*)*|[^)\s]+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?"
    r"\s*\)"
)

# Autolinks use the same well-known schemes as bare URLs. Allowing an inner
# "<" lets a first pass consume markup nested inside the angle construct and
# prevents a later pass from turning it into a new autolink. The email form uses
# the same ASCII shape (and token guard) as the global email rule.
_AUTOLINK_PATTERN = (
    r"<(?:https?://[^\s>]+|ftp://[^\s>]+|www\.[^\s>]+"
    r"|[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*@(?!(?:ping|mass))(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,63})>"
)

# URLs are matched before emails: an address-like userinfo component inside a
# URL (or a URL containing "@") must be consumed as one link, not as an email.
_BARE_URL_PATTERN = r"(?<![\w@])(?:(?:https?|ftp)://|www\.)[^\s<>]+"

_EMAIL_PATTERN = (
    r"(?<!mailto:)(?<![\w@#])"
    r"[A-Za-z0-9_%+\-]+(?:\.[A-Za-z0-9_%+\-]+)*"
    # Never absorb an emitted user/mass token as the address separator.
    r"@(?!(?:ping|mass))"
    r"(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,63}"
    r"(?![@\w\-/#])"
)

# Requires at least one letter in the name, so clock times ("12:30:45") and
# numeric ratios cannot match. The trailing lookahead prevents a shortcode from
# consuming the leading colon of an emitted ":emoji:" token.
_SHORTCODE_PATTERN = r":(?=[A-Za-z0-9_+\-]*[A-Za-z])[A-Za-z0-9_+\-]+:(?!emoji:)"

_TRAILING_URL_PUNCTUATION = frozenset(".,;:!?\"'“”‘’…»")
_CLOSING_BRACKETS = {")": "(", "]": "[", "}": "{"}


def _trim_url_suffix(url: str) -> tuple[str, str]:
    """Split punctuation/unbalanced brackets off the end of a bare URL."""
    end = len(url)
    while True:
        previous_end = end
        while end and url[end - 1] in _TRAILING_URL_PUNCTUATION:
            end -= 1
        while end:
            closing = url[end - 1]
            opening = _CLOSING_BRACKETS.get(closing)
            if opening is None:
                break
            candidate = url[:end]
            if candidate.count(closing) <= candidate.count(opening):
                break
            end -= 1
        if end == previous_end:
            break
    return url[:end], url[end:]


def _replace_bare_url(match: re.Match[str]) -> str:
    url = match.group("bare_url")
    trimmed, suffix = _trim_url_suffix(url)
    # Re-emit the trimmed punctuation so only the URL itself is replaced.
    return LINK_TOKEN + suffix


def _replace_shortcode(match: re.Match[str]) -> str:
    name = match.group("shortcode")[1:-1].lower()
    if name in KNOWN_SHORTCODES:
        return match.group(0)
    return EMOJI_TOKEN


@lru_cache(maxsize=128)
def _compile_alternation(rule_specs: tuple[tuple[str, str], ...]) -> re.Pattern[str]:
    """Compile one alternation with one named group per rule.

    Cached per rule layout, which is what makes repeated cleaner construction
    (including one cleaner per supported platform) cheap.
    """
    alternation = "|".join(f"(?P<{name}>{pattern})" for name, pattern in rule_specs)
    return re.compile(alternation, re.IGNORECASE | re.UNICODE)


class TextCleaner:
    """Clean and mask text with the fixed platform-independent rules.

    The rule set and its order are standardized; there are no per-cleaner
    options. Use :class:`~mezzoguard.cleaner.platforms.DiscordCleaner`,
    :class:`~mezzoguard.cleaner.platforms.RedditCleaner`, or
    :class:`~mezzoguard.cleaner.platforms.TwitterCleaner` to add a platform
    rule block. The base class itself is usable and masks links, emails, and
    unknown emoji shortcodes but no mentions.
    """

    platform = "generic"

    def __init__(self) -> None:
        rules = self._rules()
        self._replacements: dict[str, Replacement] = {rule.name: rule.replacement for rule in rules}
        self._pattern = _compile_alternation(tuple((rule.name, rule.pattern) for rule in rules))

        text_rules = self._text_rules()
        self._text_replacements: dict[str, Replacement] = {rule.name: rule.replacement for rule in text_rules}
        self._text_pattern = _compile_alternation(tuple((rule.name, rule.pattern) for rule in text_rules))

    def _base_rules(self, *, include_markdown: bool = True) -> list[Rule]:
        rules = []
        if include_markdown:
            rules.append(Rule("markdown_link", _MARKDOWN_LINK_PATTERN, self._replace_markdown))
        rules.extend(
            [
                Rule("autolink", _AUTOLINK_PATTERN, LINK_TOKEN),
                Rule("bare_url", _BARE_URL_PATTERN, _replace_bare_url),
                # Email addresses always collapse to the same link token as URLs.
                Rule("email", _EMAIL_PATTERN, LINK_TOKEN),
            ]
        )
        return rules

    def _platform_rules(self) -> list[Rule]:
        return []

    def _platform_and_shortcode_rules(self) -> list[Rule]:
        rules = self._platform_rules()
        # The shortcode rule is last: the Discord custom-emoji rule must see
        # "<:name:id>" before ":name:" can match inside it.
        rules.append(Rule("shortcode", _SHORTCODE_PATTERN, _replace_shortcode))
        return rules

    def _rules(self) -> list[Rule]:
        return self._base_rules() + self._platform_and_shortcode_rules()

    def _text_rules(self) -> list[Rule]:
        return self._base_rules(include_markdown=False) + self._platform_and_shortcode_rules()

    def _replace_markdown(self, match: re.Match[str]) -> str:
        prefix = match.group("md_prefix")
        text = self._text_pattern.sub(
            lambda inner: self._dispatch(inner, self._text_replacements), match.group("md_text")
        )
        return f"{prefix}{text}]({LINK_TOKEN})"

    def _dispatch(self, match: re.Match[str], replacements: dict[str, Replacement]) -> str:
        name = match.lastgroup
        if name not in replacements:
            # A nested group (for example the Markdown text) was the last match.
            for candidate in replacements:
                if match.group(candidate) is not None:
                    name = candidate
                    break
        if name is None:  # pragma: no cover - every alternative is named
            return match.group(0)
        replacement = replacements[name]
        if callable(replacement):
            return replacement(match)
        return match.expand(replacement)

    def clean(self, text: str) -> str:
        """Run the fixed five-stage pipeline over ``text``."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        cap = MAX_TEXT_LENGTH
        # Cap before masking, and again after NFKC in case normalization expanded
        # ligatures/circled characters past the limit.
        text = text[:cap]
        text = unicodedata.normalize("NFKC", text)
        text = text[:cap]
        text = _strip_invisible(text)
        text = _collapse_whitespace(text)
        text = self._mask(text)
        return _collapse_whitespace(text)

    def _mask(self, text: str) -> str:
        return self._pattern.sub(lambda match: self._dispatch(match, self._replacements), text)


__all__ = [
    "CHANNEL_TOKEN",
    "EMOJI_TOKEN",
    "LINK_TOKEN",
    "MASS_PING_TOKEN",
    "MAX_TEXT_LENGTH",
    "USER_MENTION_TOKEN",
    "Rule",
    "TextCleaner",
]
