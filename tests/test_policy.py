from mezzoguard.base_classes import PolicyResult
from mezzoguard.content_guard import ContentPolicy
from mezzoguard.content_guard.result import Result as ContentResult
from mezzoguard.prompt_guard import PromptPolicy
from mezzoguard.prompt_guard.result import Result as PromptResult


def test_content_policy_benign():
    policy = (
        ContentPolicy().add_threshold("sexual", 0.7).add_threshold("violence", 0.6).add_threshold("hate-speech", 0.5)
    )
    result = ContentResult(
        chunks=[],
        scores={
            "sexual": 0.1,
            "violence": 0.2,
            "hate-speech": 0.3,
        },
    )
    pr = policy.evaluate(result)
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)
    assert pr.get_violated_categories() == []


def test_content_policy_single_violation():
    policy = ContentPolicy().add_threshold("sexual", 0.7).add_threshold("violence", 0.6)
    result = ContentResult(
        chunks=[],
        scores={"sexual": 0.95, "violence": 0.2},
    )
    pr = policy.evaluate(result)
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == ["sexual"]


def test_content_policy_multiple_violations():
    policy = ContentPolicy()
    policy.add_threshold("sexual", 0.7)
    policy.add_threshold("violence", 0.6)
    result = ContentResult(
        chunks=[],
        scores={"sexual": 0.95, "violence": 0.85},
    )
    pr = policy.evaluate(result)
    assert set(pr.get_violated_categories()) == {"sexual", "violence"}


def test_content_policy_no_threshold_defaults_to_zero():
    policy = ContentPolicy()
    result = ContentResult(chunks=[], scores={"toxic": 0.01})
    pr = policy.evaluate(result)
    assert pr.is_unsafe()
    assert pr.get_violated_categories() == ["toxic"]


def test_prompt_policy_benign():
    policy = PromptPolicy().add_threshold("unsafe", 0.5)
    result = PromptResult(chunks=[], scores={"unsafe": 0.1})
    pr = policy.evaluate(result)
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)


def test_prompt_policy_unsafe():
    policy = PromptPolicy().add_threshold("unsafe", 0.5)
    result = PromptResult(chunks=[], scores={"unsafe": 0.95})
    pr = policy.evaluate(result)
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == ["unsafe"]


def test_prompt_policy_threshold_boundary_is_violation():
    policy = PromptPolicy().add_threshold("unsafe", 0.5)
    result = PromptResult(chunks=[], scores={"unsafe": 0.5})
    pr = policy.evaluate(result)
    assert pr.is_unsafe()
    assert pr.get_violated_categories() == ["unsafe"]


def test_policy_result_repr():
    pr = PolicyResult(
        scores={"violence": 0.9},
        violated={"violence": True},
        categories=["violence"],
    )
    assert "violated" in repr(pr)
    assert pr.is_unsafe() and bool(pr) and not pr.is_safe()


def test_policy_result_empty():
    pr = PolicyResult(scores={}, violated={}, categories=[])
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)
    assert pr.get_violated_categories() == []
