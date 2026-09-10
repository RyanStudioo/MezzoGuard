import pytest

from mezzoguard._types import Span
from mezzoguard.prompt_guard import Category as PromptCategory


def test_redact_safe_text_unchanged_with_overlap(prompt_guard):
    text = "hello world foo bar"
    assert prompt_guard.redact(text, max_seq_length=2, overlap=1) == text


def test_redact_removes_unsafe_chunks(prompt_guard):
    text = "a b c kill e f g h i j k kill m n"
    out = prompt_guard.redact(text, max_seq_length=4, overlap=1)
    assert out == "[REDACTED] h i j[REDACTED] n"


def test_redact_merges_adjacent_unsafe_chunks_into_one_marker(prompt_guard):
    text = "a b c kill e f g h i"
    out = prompt_guard.redact(text, max_seq_length=4, overlap=1)
    assert out == "[REDACTED] h i"
    assert out.count("[REDACTED]") == 1


def test_redact_no_overlap_duplication(prompt_guard):
    text = "kill kill kill a a a"
    out = prompt_guard.redact(text, max_seq_length=2, overlap=0)
    assert out == "[REDACTED] a a"


def test_redact_empty_text(prompt_guard):
    assert prompt_guard.redact("") == ""
    assert prompt_guard.redact("", return_spans=True) == ("", [])


def test_redact_return_spans(prompt_guard):
    text = "a b c kill e f"
    out, spans = prompt_guard.redact(text, max_seq_length=4, overlap=0, return_spans=True)
    assert out == "[REDACTED] e f"
    assert len(spans) == 1
    span = spans[0]
    assert isinstance(span, Span)
    assert text[span.start : span.end] == "a b c kill"
    assert span.category == PromptCategory.UNSAFE
    assert span.replacement == "[REDACTED]"


def test_redact_merged_spans_cover_full_unsafe_region(prompt_guard):
    text = "x kill y kill z"
    out, spans = prompt_guard.redact(text, max_seq_length=2, overlap=0, return_spans=True)
    assert text[spans[0].start : spans[0].end] == "x kill y kill"
    assert out == "[REDACTED] z"


def test_redact_custom_replacement(prompt_guard):
    text = "a kill b"
    out = prompt_guard.redact(text, max_seq_length=1, overlap=0, replace="[BLOCKED]")
    assert out == "a[BLOCKED] b"


def test_content_redact_per_category_replacement(content_guard):
    from mezzoguard.content_guard import Category as ContentCategory

    text = "a kill b"
    out = content_guard.redact(
        text,
        max_seq_length=1,
        overlap=0,
        replace={ContentCategory.VIOLENCE: "[V]", ContentCategory.TOXIC: "[TOXIC]"},
    )
    assert out == "a[V] b"

    text = "a stupid b"
    out = content_guard.redact(
        text,
        max_seq_length=1,
        overlap=0,
        replace={ContentCategory.HATE_SPEECH: "[HATE]", ContentCategory.TOXIC: "[TOXIC]"},
    )
    assert out == "a[HATE] b"


def test_content_redact_default_replacement_for_unknown_category(content_guard):
    from mezzoguard.content_guard import Category as ContentCategory

    text = "a kill b"
    out = content_guard.redact(
        text,
        max_seq_length=1,
        overlap=0,
        replace={ContentCategory.SEXUAL: "[SEXUAL]"},
    )
    assert out == "a[REDACTED] b"


def test_batch_redact_matches_single(prompt_guard):
    texts = ["hello world foo bar", "a b c kill e f", "all good here"]
    batch = prompt_guard.batch_redact(texts, max_seq_length=4, overlap=1)
    assert batch == [prompt_guard.redact(t, max_seq_length=4, overlap=1) for t in texts]


def test_batch_redact_empty(prompt_guard):
    assert prompt_guard.batch_redact([]) == []
    assert prompt_guard.batch_redact([""]) == [""]


def test_batch_redact_return_spans(prompt_guard):
    texts = ["a b c kill e f", "hello world"]
    outputs = prompt_guard.batch_redact(texts, max_seq_length=4, overlap=0, return_spans=True)
    assert len(outputs) == 2

    redacted, spans = outputs[0]
    assert redacted == "[REDACTED] e f"
    assert len(spans) == 1
    assert isinstance(spans[0], Span)
    assert texts[0][spans[0].start : spans[0].end] == "a b c kill"
    assert spans[0].category == PromptCategory.UNSAFE

    assert outputs[1] == ("hello world", [])


def test_content_batch_redact_matches_single(content_guard):
    texts = ["a kill b stupid c", "the weather is nice"]
    batch = content_guard.batch_redact(texts, max_seq_length=1, overlap=0)
    assert batch == [content_guard.redact(t, max_seq_length=1, overlap=0) for t in texts]


@pytest.mark.asyncio
async def test_async_batch_redact_matches_sync(prompt_guard):
    texts = ["a b c kill e f", "hello world"]
    async_results = await prompt_guard.async_batch_redact(texts, max_seq_length=4, overlap=1)
    sync_results = prompt_guard.batch_redact(texts, max_seq_length=4, overlap=1)
    assert async_results == sync_results
