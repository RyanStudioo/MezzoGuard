"""Platform-specific cleaners built on the shared :class:`TextCleaner` pipeline.

Each subclass only contributes its ordered platform rule block; normalization,
link/email masking, and the emoji rule are shared and therefore identical across
platforms. Rules are listed most-specific first because the combined alternation
resolves overlaps in list order.
"""

from __future__ import annotations

from .base import (
    CHANNEL_TOKEN,
    EMOJI_TOKEN,
    MASS_PING_TOKEN,
    USER_MENTION_TOKEN,
    Rule,
    TextCleaner,
)

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

# Mass pings are plain text, unlike other Discord markup. The lookarounds keep
# "@everyone123", "x@everyone", and "@@everyone" untouched.
_DISCORD_MASS_PING = r"(?<![\w@])@(?:here|everyone)(?!\w)"
_DISCORD_ROLE = r"<@&\d+>"
_DISCORD_USER = r"<@!?\d+>"
_DISCORD_CHANNEL = r"<#\d+>"
# The ":emoji:" token is reserved: an angle construct produced by a generated
# token must not be re-interpreted as a custom server emoji on a later pass.
_DISCORD_CUSTOM_EMOJI = r"<(?!a?:emoji:)[a]?:[A-Za-z0-9_~]+:\d+>"

# Timestamps, slash commands, and the generic "<id:...>" construct are not
# modified. This runs after the known markup rules and keeps the shortcode rule
# from reaching inside leftover angle-bracket constructs such as "<id:foo:bar>".
_DISCORD_PROTECTED_MARKUP = r"<(?:t:\d+(?::[tTdDfFR])?|/[^<>\s]+|[A-Za-z][A-Za-z0-9_]*:[^<>\s]+)>"


def _discord_rules() -> list[Rule]:
    return [
        Rule("discord_mass_ping", _DISCORD_MASS_PING, MASS_PING_TOKEN),
        Rule("discord_role", _DISCORD_ROLE, USER_MENTION_TOKEN),
        Rule("discord_user", _DISCORD_USER, USER_MENTION_TOKEN),
        Rule("discord_channel", _DISCORD_CHANNEL, CHANNEL_TOKEN),
        Rule("discord_custom_emoji", _DISCORD_CUSTOM_EMOJI, EMOJI_TOKEN),
        # Group expansion reproduces the protected construct verbatim.
        Rule("discord_protected_markup", _DISCORD_PROTECTED_MARKUP, r"\g<discord_protected_markup>"),
    ]


class DiscordCleaner(TextCleaner):
    """Discord-aware cleaner.

    Handles mass pings, role/user mentions, channels, and custom server emoji.
    Timestamps, slash commands, generic ``<id:...>`` markup, and spoilers are
    left alone. Plain ``@name``/``#channel`` text is deliberately not masked
    because it is indistinguishable from prose and colour codes.
    """

    platform = "discord"

    def _platform_rules(self) -> list[Rule]:
        return _discord_rules()


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

# The optional leading slash is inside the rule so "/u/name" collapses as a
# whole instead of leaving a stray slash behind. Both boundaries reject word
# characters and slashes, keeping names inside longer paths intact.
_REDDIT_USER = r"(?<![\w/])/?u/[A-Za-z0-9][A-Za-z0-9_-]{0,31}(?![\w/])"
_REDDIT_SUBREDDIT = r"(?<![\w/])/?r/[A-Za-z0-9][A-Za-z0-9_-]{0,31}(?![\w/])"


def _reddit_rules() -> list[Rule]:
    return [
        Rule("reddit_user", _REDDIT_USER, USER_MENTION_TOKEN),
        Rule("reddit_subreddit", _REDDIT_SUBREDDIT, CHANNEL_TOKEN),
    ]


class RedditCleaner(TextCleaner):
    """Reddit-aware cleaner for ``u/``, ``/u/``, ``r/``, and ``/r/`` references.

    ``@name`` is not a native Reddit mention and is left untouched. Blockquote
    markers, media links, and all ordinary Markdown syntax are preserved (media
    URLs are still masked by the shared link rules).
    """

    platform = "reddit"

    def _platform_rules(self) -> list[Rule]:
        return _reddit_rules()


# ---------------------------------------------------------------------------
# Twitter / X
# ---------------------------------------------------------------------------

# Email addresses and URLs are consumed by earlier shared rules. The lookbehind
# additionally protects "@@name" and any handle directly attached to a word.
# Handles beginning with an emitted token are reserved so token text is never
# extended.
_TWITTER_HANDLE = r"(?<![\w@])@(?!(?:ping|mass))[A-Za-z0-9_]{1,15}(?!\w)"


def _twitter_rules() -> list[Rule]:
    return [Rule("twitter_handle", _TWITTER_HANDLE, USER_MENTION_TOKEN)]


class TwitterCleaner(TextCleaner):
    """Twitter/X-aware cleaner for ``@handle`` references.

    Hashtags (#), cashtags ($), and shortened links are left to the shared
    link/emoji rules; hashtags and cashtags are deliberately preserved.
    """

    platform = "twitter"

    def _platform_rules(self) -> list[Rule]:
        return _twitter_rules()


# ---------------------------------------------------------------------------
# Auto (all platforms)
# ---------------------------------------------------------------------------


class AutoCleaner(TextCleaner):
    """Clean text with every platform rule block in one automatically ordered pass.

    Rule blocks are sorted by specificity: Discord angle markup and mass pings
    first, then Reddit ``u/``/``r/`` references, then Twitter/X bare ``@handle``
    references. This means overlapping text follows the most specific rule, e.g.
    ``<@123>`` becomes a Discord mention and ``@here`` becomes a mass ping rather
    than a Twitter handle.

    The cleaner does not need to detect a platform. Discord exclusions such as
    ``<id:...>``, timestamps, and slash commands are consumed by their protective
    rule before the Reddit or Twitter rules can see their contents.
    """

    platform = "auto"

    def _platform_rules(self) -> list[Rule]:
        return [*_discord_rules(), *_reddit_rules(), *_twitter_rules()]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

_PLATFORMS: dict[str, type[TextCleaner]] = {
    "generic": TextCleaner,
    "auto": AutoCleaner,
    "all": AutoCleaner,
    "discord": DiscordCleaner,
    "reddit": RedditCleaner,
    "twitter": TwitterCleaner,
    "x": TwitterCleaner,
}


def get_cleaner(platform: str = "generic") -> TextCleaner:
    """Build a cleaner for a platform name (including ``auto`` for all platforms)."""
    try:
        cleaner_class = _PLATFORMS[platform.lower()]
    except KeyError:
        supported = ", ".join(sorted(_PLATFORMS))
        raise ValueError(f"Unknown cleaner platform {platform!r}. Supported platforms: {supported}") from None
    return cleaner_class()


def clean_text(text: str, platform: str = "generic") -> str:
    """Clean ``text`` with a freshly built cleaner for ``platform``."""
    return get_cleaner(platform).clean(text)


def clean_auto(text: str) -> str:
    """Clean ``text`` with all platform rule blocks in automatic order."""
    return AutoCleaner().clean(text)


def clean_discord(text: str) -> str:
    """Clean ``text`` with the Discord rule block."""
    return DiscordCleaner().clean(text)


def clean_reddit(text: str) -> str:
    """Clean ``text`` with the Reddit rule block."""
    return RedditCleaner().clean(text)


def clean_twitter(text: str) -> str:
    """Clean ``text`` with the Twitter/X rule block."""
    return TwitterCleaner().clean(text)


__all__ = [
    "AutoCleaner",
    "DiscordCleaner",
    "RedditCleaner",
    "TwitterCleaner",
    "clean_auto",
    "clean_discord",
    "clean_reddit",
    "clean_text",
    "clean_twitter",
    "get_cleaner",
]
