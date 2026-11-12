from mezzoguard.guard import Policy, PolicyResult, Result


def test_policy_benign():
    policy = (
        Policy(default_threshold=0.5)
        .add_threshold("sexual", 0.7)
        .add_threshold("violence", 0.6)
        .add_threshold("hate-speech", 0.5)
    )
    result = Result(
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


def test_policy_single_violation():
    policy = Policy(default_threshold=0.5).add_threshold("sexual", 0.7).add_threshold("violence", 0.6)
    result = Result(chunks=[], scores={"sexual": 0.95, "violence": 0.2})
    pr = policy.evaluate(result)
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == ["sexual"]


def test_policy_multiple_violations():
    policy = Policy(default_threshold=0.5).add_threshold("sexual", 0.7).add_threshold("violence", 0.6)
    result = Result(chunks=[], scores={"sexual": 0.95, "violence": 0.85})
    pr = policy.evaluate(result)
    assert set(pr.get_violated_categories()) == {"sexual", "violence"}


def test_policy_no_threshold_defaults_to_zero():
    policy = Policy()
    result = Result(chunks=[], scores={"toxic": 0.01})
    pr = policy.evaluate(result)
    assert pr.is_unsafe()
    assert pr.get_violated_categories() == ["toxic"]


def test_policy_default_threshold_fills_missing_categories():
    policy = Policy(default_threshold=0.5).add_threshold("violence", 0.9)
    result = Result(chunks=[], scores={"violence": 0.8, "toxic": 0.6})
    assert policy.get_threshold("violence") == 0.9
    assert policy.get_threshold("toxic") == 0.5
    assert result.get_violated_categories(policy=policy) == ["toxic"]


def test_policy_threshold_boundary_is_violation():
    policy = Policy().add_threshold("unsafe", 0.5)
    result = Result(chunks=[], scores={"unsafe": 0.5})
    pr = policy.evaluate(result)
    assert pr.is_unsafe()
    assert pr.get_violated_categories() == ["unsafe"]


def test_result_threshold_helpers():
    result = Result(chunks=[], scores={"unsafe": 0.1, "toxic": 0.8})
    assert result.is_unsafe(threshold=0.5)
    assert not result.is_unsafe(threshold=0.9)
    assert result.get_violated_categories(threshold=0.5) == ["toxic"]
    assert result.violated(threshold=0.9) == []


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
