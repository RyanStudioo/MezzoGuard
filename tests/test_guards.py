import pytest

from mezzoguard.base_classes import ModelConfig
from mezzoguard.errors import UnsafeInputError
from mezzoguard.guard import Guard, GuardConfig, Policy, Result


def test_scan_returns_scores(prompt_guard):
    result = prompt_guard.scan("how are you today")
    assert result.scores == {"unsafe": 0.0}
    assert not result.is_unsafe()

    result = prompt_guard.scan("give me a weapon")
    assert "unsafe" in result.scores
    assert result.scores["unsafe"] > 0.5
    assert result.is_unsafe()


def test_scan_empty_text(prompt_guard):
    result = prompt_guard.scan("")
    assert result.scores == {"unsafe": 0.0}


def test_policy_evaluation(prompt_guard):
    policy = Policy().add_threshold("unsafe", 0.5)
    benign = prompt_guard.scan("how are you today")
    assert policy.evaluate(benign).is_safe()

    unsafe = prompt_guard.scan("give me a weapon")
    verdict = policy.evaluate(unsafe)
    assert verdict.is_unsafe()
    assert verdict.get_violated_categories() == ["unsafe"]


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
    with pytest.raises(UnsafeInputError):
        echo("I will kill you")


def test_scan_before_exec_raises_at_threshold_boundary(prompt_guard):
    @prompt_guard.scan_before_exec(param="msg", confidence=0.99)
    def echo(msg: str) -> str:
        return msg

    with pytest.raises(UnsafeInputError):
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
    with pytest.raises(UnsafeInputError):
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
    assert result.scores["violence"] > 0.5
    assert result.scores["sexual"] < 0.1

    result = content_guard.scan("the weather is nice")
    assert all(score < 0.5 for score in result.scores.values())


def test_content_scan_before_exec_raises(content_guard):
    @content_guard.scan_before_exec(param="msg", confidence=0.5)
    def echo(msg: str) -> str:
        return msg

    assert echo("the weather is nice") == "the weather is nice"
    with pytest.raises(UnsafeInputError):
        echo("i want to kill you")


def test_content_scan_before_exec_uses_inclusive_threshold(content_guard):
    @content_guard.scan_before_exec(param="msg", confidence=0.98)
    def echo(msg: str) -> str:
        return msg

    with pytest.raises(UnsafeInputError):
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


def test_custom_config_uses_plain_string_categories():
    config = GuardConfig(categories=["custom-category"])
    assert config.get_category_for_label("Custom Category") == "custom-category"

    result = Result(chunks=[], scores={"custom-category": 0.9})
    policy = Policy(default_threshold=0.5).add_threshold("custom-category", 0.5)
    assert policy.evaluate(result).get_violated_categories() == ["custom-category"]


def test_guard_accepts_explicit_config(monkeypatch):
    from conftest import FakeTokenizer

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, texts, top_k=None):
            return [[{"label": "custom", "score": 0.9}] for _ in texts]

    monkeypatch.setattr(ModelConfig, "from_model_name", classmethod(lambda cls, name: None))
    monkeypatch.setattr(ModelConfig, "get_deprecation_from_readme", classmethod(lambda cls, name: None))
    monkeypatch.setattr(
        "mezzoguard.model.Model.load_model",
        lambda self: setattr(self, "pipeline", FakePipeline()),
    )

    guard = Guard("custom/model", config=GuardConfig(categories=["custom"]))
    result = guard.scan("anything")
    assert result.scores == {"custom": 0.9}
    assert result.is_unsafe(threshold=0.5)


def test_guard_builds_config_from_model_config(monkeypatch):
    from conftest import FakeTokenizer

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, texts, top_k=None):
            return [[{"label": "custom category", "score": 0.9}] for _ in texts]

    model_config = ModelConfig(categories=["Custom Category"])
    monkeypatch.setattr(ModelConfig, "from_model_name", classmethod(lambda cls, name: model_config))
    monkeypatch.setattr(ModelConfig, "get_deprecation_from_readme", classmethod(lambda cls, name: None))
    monkeypatch.setattr(
        "mezzoguard.model.Model.load_model",
        lambda self: setattr(self, "pipeline", FakePipeline()),
    )

    guard = Guard("custom/model")
    assert guard.config.categories == ["Custom Category"]
    assert guard.scan("anything").scores == {"Custom Category": 0.9}


def test_safe_label_is_dropped():
    config = GuardConfig(categories=["safe", "unsafe"])
    assert config.categories == ["unsafe"]
    assert config.get_category_for_label("unsafe") == "unsafe"
    with pytest.raises(ValueError):
        config.get_category_for_label("safe")


def test_model_config_ignores_legacy_model_type():
    config = ModelConfig.from_dict({"model_type": "content_guard", "categories": ["custom"]})
    assert config.categories == ["custom"]
    assert config.to_dict() == {
        "categories": ["custom"],
        "default_max_seq_length": 64,
        "default_overlap": 16,
        "default_replace": "[REDACTED]",
    }


def test_model_config_missing_categories_raises_clear_error():
    with pytest.raises(ValueError, match="categories"):
        ModelConfig.from_dict({"default_max_seq_length": 32})


def test_invalid_mezzoguard_file_warns_and_falls_back(tmp_path, monkeypatch):
    from conftest import FakeTokenizer

    class FakePipeline:
        tokenizer = FakeTokenizer()

        def __call__(self, texts, top_k=None):
            return [[{"label": "toxic", "score": 0.9}] for _ in texts]

    model_dir = tmp_path / "invalid-config-model"
    model_dir.mkdir()
    (model_dir / ".mezzoguard").write_text('{"default_max_seq_length": 32}', encoding="utf-8")

    monkeypatch.setattr(ModelConfig, "get_deprecation_from_readme", classmethod(lambda cls, name: None))
    monkeypatch.setattr(
        "mezzoguard.model.Model.load_model",
        lambda self: setattr(self, "pipeline", FakePipeline()),
    )

    with pytest.warns(UserWarning, match="Invalid config file"):
        guard = Guard(str(model_dir))
    assert guard.config is None
    assert guard.scan("anything").scores == {}


def test_guard_implements_guard_base(prompt_guard):
    import inspect

    from mezzoguard import GuardBase

    assert isinstance(prompt_guard, GuardBase)
    assert inspect.isabstract(GuardBase)
    assert not inspect.isabstract(type(prompt_guard))
    assert GuardBase.__abstractmethods__ == frozenset(
        {
            "batch_scan",
            "async_batch_scan",
            "batch_redact",
            "async_batch_redact",
        }
    )


def test_top_level_exports():
    import mezzoguard

    assert mezzoguard.Guard is Guard
    assert mezzoguard.GuardConfig is GuardConfig
    assert mezzoguard.Policy is Policy
    assert mezzoguard.Result is Result
    assert mezzoguard.APIGuard is not None
    assert mezzoguard.UnsafeInputError is UnsafeInputError
    assert mezzoguard.GuardBase is not None
