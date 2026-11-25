import re

import pytest

from mezzoguard.cleaner import (
    CHANNEL_TOKEN,
    EMOJI_TOKEN,
    LINK_TOKEN,
    MASS_PING_TOKEN,
    MAX_TEXT_LENGTH,
    USER_MENTION_TOKEN,
    AutoCleaner,
    DiscordCleaner,
    RedditCleaner,
    TextCleaner,
    TwitterCleaner,
    clean_auto,
    clean_discord,
    clean_reddit,
    clean_text,
    clean_twitter,
    get_cleaner,
)


@pytest.fixture(scope="module")
def discord():
    return DiscordCleaner()


@pytest.fixture(scope="module")
def reddit():
    return RedditCleaner()


@pytest.fixture(scope="module")
def twitter():
    return TwitterCleaner()


@pytest.fixture(scope="module")
def auto():
    return AutoCleaner()


@pytest.fixture(scope="module")
def generic():
    return TextCleaner()


# ---------------------------------------------------------------------------
# Pipeline: normalization, invisibles, whitespace
# ---------------------------------------------------------------------------


def test_nfkc_normalization_folds_fullwidth_characters(discord):
    assert discord.clean("＠everyone") == MASS_PING_TOKEN
    assert discord.clean("＠ＨＥＲＥ") == MASS_PING_TOKEN
    assert TextCleaner().clean("Ｈ＃") == "H#"


def test_zero_width_characters_are_removed_from_words(generic):
    assert generic.clean("ig\u200bnore\u200cme\ufeff") == "ignoreme"


def test_zwj_survives_only_between_pictographs(generic):
    family = "👨\u200d👩\u200d👧\u200d👦"
    assert generic.clean(family) == family
    assert generic.clean("a\u200db") == "ab"


def test_whitespace_runs_collapse(generic):
    assert generic.clean("  a\u00a0\u00a0\tb\n\n c  ") == "a b c"


# ---------------------------------------------------------------------------
# Links, autolinks, emails, Markdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see [docs](https://example.com/x) now", "see [docs]([LINK]) now"),
        ("![alt text](https://example.com/a.png)", "![alt text]([LINK])"),
        ('[docs](https://example.com "Title")', "[docs]([LINK])"),
        ("<https://example.com>", "[LINK]"),
        ("go to https://example.com.", "go to [LINK]."),
        ("(https://example.com)", "([LINK])"),
        ("https://en.wikipedia.org/wiki/Foo_(bar).", "[LINK]."),
        ("https://example.com/a_(b)).", "[LINK])."),
        ("ftp://example.com/file.txt", "[LINK]"),
        ("www.example.com/path", "[LINK]"),
        ("HTTP://EXAMPLE.COM", "[LINK]"),
        ("mail foo@example.com now", "mail [LINK] now"),
        ("https://user@example.com/path", "[LINK]"),
    ],
)
def test_link_and_email_masking(text, expected):
    assert TextCleaner().clean(text) == expected


@pytest.mark.parametrize("punctuation", [".", ",", ";", ":", "!", "?", '"', "'", "...", "»"])
def test_trailing_url_punctuation_is_preserved(punctuation):
    assert TextCleaner().clean(f"x https://example.com{punctuation}") == f"x [LINK]{punctuation}"


def test_email_masking_is_standardized():
    assert TextCleaner().clean("foo@example.com") == "[LINK]"
    assert DiscordCleaner().clean("foo@example.com") == "[LINK]"
    assert RedditCleaner().clean("foo@example.com") == "[LINK]"
    assert TwitterCleaner().clean("foo@example.com") == "[LINK]"


@pytest.mark.parametrize(
    "text",
    [
        "[docs][ref]",
        "![alt][ref]",
        "mailto:foo@example.com",
        "tel:+15551234",
        "**bold** _italic_ # heading > quote",
    ],
)
def test_excluded_markdown_and_schemes_are_preserved(text):
    assert TextCleaner().clean(text) == text


def test_mentions_inside_code_fences_are_still_masked(discord):
    assert discord.clean("```\n@here\n```") == "``` @mass ```"


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<@123456789012345678>", USER_MENTION_TOKEN),
        ("<@!123456789012345678>", USER_MENTION_TOKEN),
        ("<@&987654321098765432>", USER_MENTION_TOKEN),
        ("<#555555555555555555>", CHANNEL_TOKEN),
        ("@here", MASS_PING_TOKEN),
        ("@everyone", MASS_PING_TOKEN),
        ("@HERE", MASS_PING_TOKEN),
        ("@EveryOne", MASS_PING_TOKEN),
        ("＠everyone", MASS_PING_TOKEN),
        ("@everyone123", "@everyone123"),
        ("<:smile:123456789012345678>", EMOJI_TOKEN),
        ("<a:partyparrot:123456789012345678>", EMOJI_TOKEN),
        ("https://discord.gg/abcd", LINK_TOKEN),
        ("https://discord.com/channels/1/2", LINK_TOKEN),
    ],
)
def test_discord_masking(discord, text, expected):
    assert discord.clean(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<t:1700000000:R>",
        "</ban:123456789012345678>",
        "<id:123456789012345678>",
        "||spoiler||",
    ],
)
def test_discord_excluded_markup_is_preserved(discord, text):
    assert discord.clean(text) == text


def test_discord_bare_mentions_are_left_alone(discord):
    assert discord.clean("@user #general") == "@user #general"


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("u/Some_User", USER_MENTION_TOKEN),
        ("/u/Some_User", USER_MENTION_TOKEN),
        ("U/Some_User", USER_MENTION_TOKEN),
        ("r/python", CHANNEL_TOKEN),
        ("/r/PythonTricks", CHANNEL_TOKEN),
        ("R/Python", CHANNEL_TOKEN),
        ("https://www.reddit.com/u/foo", LINK_TOKEN),
        ("https://www.reddit.com/r/python/comments/abc", LINK_TOKEN),
        ("https://example.com/u/foo", LINK_TOKEN),
        ("notu/name", "notu/name"),
        ("u/name/extra", "u/name/extra"),
        ("example.com/u/name", "example.com/u/name"),
        ("@name", "@name"),
        ("> quoted text", "> quoted text"),
    ],
)
def test_reddit_masking(reddit, text, expected):
    assert reddit.clean(text) == expected


# ---------------------------------------------------------------------------
# Twitter / X
# ---------------------------------------------------------------------------


def test_twitter_mentions_are_case_insensitive(twitter):
    assert twitter.clean("@Alice and @alice") == "@ping and @ping"


def test_twitter_hashtags_and_cashtags_survive(twitter):
    assert twitter.clean("#Python $AAPL") == "#Python $AAPL"


def test_twitter_email_becomes_a_link(twitter):
    assert twitter.clean("contact foo@example.com") == "contact [LINK]"


def test_twitter_doubled_at_sign_is_untouched(twitter):
    assert twitter.clean("@@user") == "@@user"


def test_twitter_urls_are_masked_before_mentions(twitter):
    assert twitter.clean("https://x.com/@user") == "[LINK]"
    assert twitter.clean("https://t.co/abcd") == "[LINK]"


# ---------------------------------------------------------------------------
# Auto cleaner
# ---------------------------------------------------------------------------


def test_auto_cleans_all_platforms_in_one_pass(auto):
    text = "<@123> @everyone u/foo r/bar @user foo@bar.com https://x.com/@bob :nope: <:party:123>"
    assert auto.clean(text) == "@ping @mass @ping #chat @ping [LINK] [LINK] :emoji: :emoji:"


def test_auto_uses_most_specific_rule_order(auto):
    assert auto.clean("<@123> @here @user") == "@ping @mass @ping"


def test_auto_preserves_discord_protected_markup(auto):
    text = "<id:foo:@user> <t:1700000000:R> </cmd:u/foo>"
    assert auto.clean(text) == text


def test_auto_is_idempotent(auto):
    text = "<@123> @here u/foo r/bar @user foo@bar.com https://x.com/@bob :nope: <:party:123> <id:foo:@user>"
    once = auto.clean(text)
    assert auto.clean(once) == once


def test_clean_auto_and_factory():
    assert clean_auto("@here") == MASS_PING_TOKEN
    assert isinstance(get_cleaner("auto"), AutoCleaner)
    assert isinstance(get_cleaner("all"), AutoCleaner)


# ---------------------------------------------------------------------------
# Cross-platform behavior
# ---------------------------------------------------------------------------


def test_mention_inside_link_text_and_alt_text_is_masked(twitter):
    assert twitter.clean("[hi @user](https://example.com)") == "[hi @ping]([LINK])"
    assert twitter.clean("![alt @user](https://example.com/x.png)") == "![alt @ping]([LINK])"


def test_discord_mention_inside_link_text_is_masked(discord):
    assert discord.clean("[hi <@123>](https://example.com)") == "[hi @ping]([LINK])"
    assert discord.clean("![alt <@&456>](https://example.com/x.png)") == "![alt @ping]([LINK])"


def test_clock_times_and_numeric_shortcodes_are_not_mangled(generic):
    assert generic.clean("12:30 12:30:45 :100:") == "12:30 12:30:45 :100:"


def test_known_and_unknown_shortcodes(generic):
    assert generic.clean(":smile:") == ":smile:"
    assert generic.clean(":blobcat_wave:") == EMOJI_TOKEN
    assert generic.clean(":not-a-real-emoji:") == EMOJI_TOKEN


def test_shortcode_table_is_always_applied():
    cleaner = DiscordCleaner()
    assert cleaner.clean(":smile: :whatever: <:custom:123>") == ":smile: :emoji: :emoji:"


@pytest.mark.parametrize(
    "emoji",
    [
        "👨\u200d👩\u200d👧\u200d👦",
        "👨🏽\u200d👩🏽\u200d👧🏽",
        "👩🏽\u200d🚀",
        "🇺🇸",
        "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
        "👍🏽",
        "1\ufe0f\u20e3",
        "❤\ufe0f",
    ],
)
def test_unicode_emoji_survive_byte_identical(generic, emoji):
    assert generic.clean(emoji) == emoji


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_no_platform_markup_or_url_leakage(discord):
    text = (
        "https://example.com/@user www.example.com "
        "<@123> <@!123> <@&123> <#123> <:smile:123> <a:party:123> @everyone @here"
    )
    output = discord.clean(text)
    for leaked in ("http", "www.", "<@", "<#", "<:", "<a:", "@everyone", "@here"):
        assert leaked not in output


def test_no_reddit_reference_leakage(reddit):
    output = reddit.clean("u/foo /u/bar r/baz /r/qux")
    assert "u/" not in output
    assert "r/" not in output


def test_only_permitted_tokens_are_emitted(discord):
    text = "https://example.com <@123> <@&123> <#123> <:smile:123> @here :blobcat: foo@example.com"
    output = discord.clean(text)
    found = set(re.findall(r"\[LINK\]|@ping|@mass|#chat|:emoji:", output))
    assert found <= {LINK_TOKEN, USER_MENTION_TOKEN, MASS_PING_TOKEN, CHANNEL_TOKEN, EMOJI_TOKEN}


def test_link_token_is_never_nested(generic):
    output = generic.clean("https://a.example/https://b.example")
    assert output == LINK_TOKEN
    assert output.count(LINK_TOKEN) == 1

    output = generic.clean("[a](https://example.com)")
    assert output == "[a]([LINK])"
    assert output.count(LINK_TOKEN) == 1


@pytest.mark.parametrize(
    "cleaner",
    [
        TextCleaner(),
        AutoCleaner(),
        DiscordCleaner(),
        RedditCleaner(),
        TwitterCleaner(),
    ],
)
def test_masking_is_idempotent(cleaner):
    text = (
        "  <@123> <@!456> <@&789> <#555> @here @everyone @user #general "
        "u/foo /u/bar r/baz /r/qux [docs](https://example.com) "
        "![alt](https://example.com/a.png) <https://x> foo@example.com "
        ":smile: :blobcat: <:custom:123> <a:party:456> "
        "👨\u200d👩\u200d👧\u200d👦 🇺🇸 👍🏽 12:30:45  "
    )
    once = cleaner.clean(text)
    assert cleaner.clean(once) == once


def test_masking_is_deterministic():
    text = "a\u200db https://example.com [x](https://y) <@1> @here :blobcat:"
    cleaner = DiscordCleaner()
    first = cleaner.clean(text)
    assert cleaner.clean(text) == first
    assert DiscordCleaner().clean(text) == first


def test_normalization_and_links_are_platform_independent():
    text = "  Foo\u200bBar  https://example.com. foo@example.com :blobcat:  "
    expected = TextCleaner().clean(text)
    assert DiscordCleaner().clean(text) == expected
    assert RedditCleaner().clean(text) == expected
    assert TwitterCleaner().clean(text) == expected


def test_input_is_capped_before_masking(discord):
    assert len(discord.clean("x" * (MAX_TEXT_LENGTH + 100))) == MAX_TEXT_LENGTH

    # A mention before the cap is masked; content after the cap is dropped.
    text = "a" * (MAX_TEXT_LENGTH - 10) + "<@123>" + "b" * 100
    assert discord.clean(text) == "a" * (MAX_TEXT_LENGTH - 10) + "@ping" + "b" * 4

    # Normalization can expand the text, so the cap is applied again before masking.
    expanded = TextCleaner().clean("ﬁ" * MAX_TEXT_LENGTH)
    assert len(expanded) == MAX_TEXT_LENGTH
    assert "ﬁ" not in expanded


def test_input_type_is_checked():
    with pytest.raises(TypeError):
        TextCleaner().clean(None)  # type: ignore[arg-type]


def test_factory_and_convenience_functions():
    assert clean_discord("@here") == MASS_PING_TOKEN
    assert clean_reddit("r/python") == CHANNEL_TOKEN
    assert clean_twitter("@user") == USER_MENTION_TOKEN
    assert clean_text("<@123>", "discord") == USER_MENTION_TOKEN
    assert isinstance(get_cleaner("x"), TwitterCleaner)
    with pytest.raises(ValueError):
        get_cleaner("myspace")
