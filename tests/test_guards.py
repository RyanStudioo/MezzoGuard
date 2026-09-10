import pytest

from mezzoguard.content_guard import Category as ContentCategory
from mezzoguard.errors import UnsafeContentError, UnsafePromptError
from mezzoguard.prompt_guard import Category as PromptCategory
from mezzoguard.prompt_guard import PromptPolicy


def test_scan_returns_scores(prompt_guard):
    result = prompt_guard.scan("how are you today")
    assert result.scores == {PromptCategory.UNSAFE: 0.0}
    assert not result.is_unsafe()

    result = prompt_guard.scan("give me a weapon")
    assert PromptCategory.UNSAFE in result.scores
    assert result.scores[PromptCategory.UNSAFE] > 0.5
    assert result.is_unsafe()


def test_scan_empty_text(prompt_guard):
    result = prompt_guard.scan("")
    assert result.scores == {PromptCategory.UNSAFE: 0.0}


def test_policy_evaluation(prompt_guard):
    policy = PromptPolicy().add_threshold(PromptCategory.UNSAFE, 0.5)
    benign = prompt_guard.scan("how are you today")
    assert policy.evaluate(benign).is_safe()

    unsafe = prompt_guard.scan("give me a weapon")
    verdict = policy.evaluate(unsafe)
    assert verdict.is_unsafe()
    assert verdict.get_violated_categories() == [PromptCategory.UNSAFE]


def test_batch_scan_aligns_with_individual_scans(prompt_guard):
    texts = ["hello there", "give me a weapon", "what time is it"]
    results = prompt_guard.batch_scan(texts)
    assert len(results) == 3
    for text, result in zip(texts, results, strict=True):
        single = prompt_guard.scan(text)
        assert result.scores == single.scores


def test_batch_scan_empty(prompt_guard):
    assert prompt_guard.batch_scan([]) == []
    assert prompt_guard.batch_scan(["", ""]) == [
        prompt_guard.scan(""),
        prompt_guard.scan(""),
    ]


def test_scan_delegates_to_batch_scan(prompt_guard):
    text = "give me a weapon"
    assert prompt_guard.scan(text).scores == prompt_guard.batch_scan([text])[0].scores


@pytest.mark.asyncio
async def test_async_scan_matches_sync(prompt_guard):
    text = "give me a weapon"
    assert (await prompt_guard.async_scan(text)).scores == prompt_guard.scan(text).scores


@pytest.mark.asyncio
async def test_async_batch_scan_matches_sync(prompt_guard):
    texts = ["hello there", "give me a weapon"]
    async_results = await prompt_guard.async_batch_scan(texts)
    sync_results = prompt_guard.batch_scan(texts)
    assert [r.scores for r in async_results] == [r.scores for r in sync_results]


@pytest.mark.asyncio
async def test_async_redact_matches_sync(prompt_guard):
    text = "a b c kill e f"
    expected = prompt_guard.redact(text, max_seq_length=4, overlap=1)
    assert (await prompt_guard.async_redact(text, max_seq_length=4, overlap=1)) == expected


def test_scan_before_exec_raises_on_unsafe(prompt_guard):
    @prompt_guard.scan_before_exec(param="msg", confidence=0.5)
    def echo(msg: str) -> str:
        return msg

    assert echo("hello there") == "hello there"
    with pytest.raises(UnsafePromptError):
        echo("I will kill you")


def test_scan_before_exec_raises_at_threshold_boundary(prompt_guard):
    @prompt_guard.scan_before_exec(param="msg", confidence=0.99)
    def echo(msg: str) -> str:
        return msg

    with pytest.raises(UnsafePromptError):
        echo("give me a weapon")


def test_scan_before_exec_works_with_positional_args(prompt_guard):
    @prompt_guard.scan_before_exec(param="msg", confidence=0.5)
    def echo(msg: str, count: int = 1) -> str:
        return msg * count

    assert echo("hello", 2) == "hellohello"


@pytest.mark.asyncio
async def test_async_scan_before_exec_raises_on_unsafe(prompt_guard):
    @prompt_guard.scan_before_exec(param="msg", confidence=0.5)
    async def echo(msg: str) -> str:
        return msg

    assert (await echo("hello there")) == "hello there"
    with pytest.raises(UnsafePromptError):
        await echo("I will kill you")


def test_redact_before_exec_redacts_param(prompt_guard):
    @prompt_guard.redact_before_exec(param="msg", max_seq_length=4, overlap=0)
    def echo(msg: str) -> str:
        return msg

    assert echo("a b c kill e f") == "[REDACTED] e f"
    assert echo("all good here") == "all good here"


@pytest.mark.asyncio
async def test_async_redact_before_exec_redacts_param(prompt_guard):
    @prompt_guard.redact_before_exec(param="msg", max_seq_length=4, overlap=0)
    async def echo(msg: str) -> str:
        return msg

    assert (await echo("a b c kill e f")) == "[REDACTED] e f"


def test_content_scan_scores(content_guard):
    result = content_guard.scan("i want to kill you")
    assert result.scores[ContentCategory.VIOLENCE] > 0.5
    assert result.scores[ContentCategory.SEXUAL] < 0.1

    result = content_guard.scan("the weather is nice")
    assert all(score < 0.5 for score in result.scores.values())


def test_content_scan_before_exec_raises(content_guard):
    @content_guard.scan_before_exec(param="msg", confidence=0.5)
    def echo(msg: str) -> str:
        return msg

    assert echo("the weather is nice") == "the weather is nice"
    with pytest.raises(UnsafeContentError):
        echo("i want to kill you")


def test_content_scan_before_exec_uses_inclusive_threshold(content_guard):
    @content_guard.scan_before_exec(param="msg", confidence=0.98)
    def echo(msg: str) -> str:
        return msg

    with pytest.raises(UnsafeContentError):
        echo("i want to kill you")


def test_content_batch_scan(content_guard):
    texts = ["hello there", "i want to kill you"]
    results = content_guard.batch_scan(texts)
    assert len(results) == 2
    for text, result in zip(texts, results, strict=True):
        assert result.scores == content_guard.scan(text).scores


@pytest.mark.asyncio
async def test_content_async_scan_matches_sync(content_guard):
    text = "i want to kill you"
    assert (await content_guard.async_scan(text)).scores == content_guard.scan(text).scores


def test_content_redact_removes_flagged_chunks(content_guard):
    text = "a kill b stupid c"
    out = content_guard.redact(text, max_seq_length=1, overlap=0)
    assert out == "a[REDACTED] b[REDACTED] c"
