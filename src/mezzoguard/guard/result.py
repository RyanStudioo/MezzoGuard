from __future__ import annotations

from typing import TYPE_CHECKING

from .._types import DEFAULT_CONFIDENCE, BaseResult

if TYPE_CHECKING:
    from .policy import Policy


class PolicyResult(BaseResult):
    """Result of evaluating a :class:`Policy` against a :class:`Result`."""

    def __init__(self, scores: dict[str, float], violated: dict[str, bool], categories: list[str]):
        self.scores = scores
        self.violated = violated
        self.categories = categories

    def __bool__(self) -> bool:
        return self.is_unsafe()

    def __repr__(self) -> str:
        violated = [category for category, is_violated in self.violated.items() if is_violated]
        safe = [category for category, is_violated in self.violated.items() if not is_violated]
        return f"PolicyResult(safe={safe}, violated={violated})"

    def is_safe(self) -> bool:
        return not any(self.violated.values())

    def is_unsafe(self) -> bool:
        return any(self.violated.values())

    def get_violated_categories(self) -> list[str]:
        return [category for category, violated in self.violated.items() if violated]


class Result(BaseResult):
    """Result returned by the unified guard.

    ``scores`` is a ``{category: score}`` mapping. Prompt-style models produce
    ``{"unsafe": score}``; content models produce one entry per configured
    category. Thresholds can be evaluated either with a plain float or a
    :class:`~mezzoguard.guard.policy.Policy`.
    """

    def __init__(self, chunks: list[list[dict]] | list[dict], scores: dict[str, float]):
        self._chunks = chunks
        self.scores = scores

    @property
    def chunks(self) -> list[list[dict]] | list[dict]:
        return self._chunks

    def evaluate(self, policy: Policy) -> PolicyResult:
        """Evaluate this result with ``policy``."""
        return policy.evaluate(self)

    def is_unsafe(self, threshold: float = DEFAULT_CONFIDENCE, policy: Policy | None = None) -> bool:
        """Return whether any score meets the threshold (or the supplied policy)."""
        if policy is not None:
            return policy.evaluate(self).is_unsafe()
        return any(score >= threshold for score in self.scores.values())

    def get_violated_categories(self, policy: Policy | None = None, threshold: float = DEFAULT_CONFIDENCE) -> list[str]:
        """Return the categories that meet the threshold (or the supplied policy)."""
        if policy is None:
            from .policy import Policy

            policy = Policy(default_threshold=threshold)
        return policy.evaluate(self).get_violated_categories()

    def violated(self, policy: Policy | None = None, threshold: float = DEFAULT_CONFIDENCE) -> list[str]:
        """Alias for :meth:`get_violated_categories`."""
        return self.get_violated_categories(policy=policy, threshold=threshold)


__all__ = ["PolicyResult", "Result"]
