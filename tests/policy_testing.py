"""Test script for ContentPolicy and PromptPolicy with PolicyResult.
True = violation, False = benign."""

from mezzoguard.content_guard import ContentPolicy, BaseCategory as ContentCategory
from mezzoguard.content_guard.result import Result as ContentResult
from mezzoguard.prompt_guard import PromptPolicy, BaseCategory as PromptCategory
from mezzoguard.prompt_guard.result import Result as PromptResult


def test_content_policy():
    print("=== ContentPolicy Tests ===")
    policy = (
        ContentPolicy()
        .add_threshold(ContentCategory.SEXUAL, 0.7)
        .add_threshold(ContentCategory.VIOLENCE, 0.6)
        .add_threshold(ContentCategory.HATE_SPEECH, 0.5)
    )

    # Benign example: all scores below thresholds
    benign_scores = {
        ContentCategory.SEXUAL: 0.1,
        ContentCategory.VIOLENCE: 0.2,
        ContentCategory.HATE_SPEECH: 0.3,
    }
    result = ContentResult(chunks=[], scores=benign_scores)
    pr = policy.evaluate(result)
    print(f"Benign query:")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)
    assert pr.get_violated_categories() == []

    # Sexual violation: score exceeds threshold
    sexual_scores = {
        ContentCategory.SEXUAL: 0.95,
        ContentCategory.VIOLENCE: 0.2,
        ContentCategory.HATE_SPEECH: 0.3,
    }
    result = ContentResult(chunks=[], scores=sexual_scores)
    pr = policy.evaluate(result)
    print(f"Sexual violation:")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == [ContentCategory.SEXUAL]

    # Multiple violations
    multi_scores = {
        ContentCategory.SEXUAL: 0.95,
        ContentCategory.VIOLENCE: 0.85,
        ContentCategory.HATE_SPEECH: 0.3,
    }
    result = ContentResult(chunks=[], scores=multi_scores)
    pr = policy.evaluate(result)
    print(f"Multiple violations:")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert set(pr.get_violated_categories()) == {ContentCategory.SEXUAL, ContentCategory.VIOLENCE}

    # No threshold set for a category (defaults to 0.0 -> always violated)
    no_threshold_scores = {
        ContentCategory.TOXIC: 0.01,
    }
    result = ContentResult(chunks=[], scores=no_threshold_scores)
    pr = policy.evaluate(result)
    print(f"No threshold set (default 0.0):")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == [ContentCategory.TOXIC]

    print("All content policy tests passed!\n")


def test_prompt_policy():
    print("=== PromptPolicy Tests ===")
    policy = PromptPolicy().add_threshold(PromptCategory.UNSAFE, 0.5)

    # Benign prompt
    benign_scores = {PromptCategory.UNSAFE: 0.1}
    result = PromptResult(chunks=[], scores=benign_scores)
    pr = policy.evaluate(result)
    print(f"Benign prompt:")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)
    assert pr.get_violated_categories() == []

    # Unsafe prompt
    unsafe_scores = {PromptCategory.UNSAFE: 0.95}
    result = PromptResult(chunks=[], scores=unsafe_scores)
    pr = policy.evaluate(result)
    print(f"Unsafe prompt:")
    print(f"  {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == [PromptCategory.UNSAFE]

    # Exact threshold boundary
    boundary_scores = {PromptCategory.UNSAFE: 0.5}
    result = PromptResult(chunks=[], scores=boundary_scores)
    pr = policy.evaluate(result)
    print(f"On threshold boundary (0.5 >= 0.5):")
    print(f"  {pr}")
    assert not pr.is_safe() and pr.is_unsafe()
    assert pr.get_violated_categories() == [PromptCategory.UNSAFE]

    print("All prompt policy tests passed!\n")


def test_policy_result_manual():
    print("=== PolicyResult Manual Tests ===")
    from mezzoguard.base_classes import PolicyResult

    # Empty categories
    pr = PolicyResult(scores={}, violated={}, categories=[])
    print(f"Empty: {pr}")
    print(f"  is_safe: {pr.is_safe()}, is_unsafe: {pr.is_unsafe()}, bool: {bool(pr)}")

    # All violated
    pr = PolicyResult(
        scores={ContentCategory.VIOLENCE: 0.9},
        violated={ContentCategory.VIOLENCE: True},
        categories=[ContentCategory.VIOLENCE]
    )
    assert pr.is_unsafe() and bool(pr) and not pr.is_safe()
    assert pr.get_violated_categories() == [ContentCategory.VIOLENCE]

    # All benign
    pr = PolicyResult(
        scores={ContentCategory.VIOLENCE: 0.1},
        violated={ContentCategory.VIOLENCE: False},
        categories=[ContentCategory.VIOLENCE]
    )
    assert pr.is_safe() and not pr.is_unsafe() and not bool(pr)
    assert pr.get_violated_categories() == []

    # Mixed
    pr = PolicyResult(
        scores={ContentCategory.VIOLENCE: 0.9, ContentCategory.SEXUAL: 0.1},
        violated={ContentCategory.VIOLENCE: True, ContentCategory.SEXUAL: False},
        categories=[ContentCategory.VIOLENCE, ContentCategory.SEXUAL]
    )
    assert not pr.is_safe() and pr.is_unsafe() and bool(pr)
    assert pr.get_violated_categories() == [ContentCategory.VIOLENCE]

    print("All PolicyResult manual tests passed!\n")


if __name__ == "__main__":
    test_policy_result_manual()
    test_content_policy()
    test_prompt_policy()
    print("All tests passed!")
