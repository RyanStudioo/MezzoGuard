from __future__ import annotations

from .._types import BaseResult
from .result import PolicyResult


class Policy:
    """Per-category thresholds for the unified guard.

    ``default_threshold`` is used for categories that have no explicit
    threshold. Pass ``default_threshold=0.5`` to make partial policies fall
    back to a sensible value instead of flagging every category at ``0.0``.
    """

    def __init__(self, default_threshold: float = 0.0):
        self._mapping: dict[str, float] = {}
        self.default_threshold = default_threshold

    def add_threshold(self, category: str, threshold: float) -> Policy:
        """Add a threshold for ``category`` and return ``self`` for chaining."""
        self._mapping[category] = threshold
        return self

    def get_threshold(self, category: str) -> float:
        """Get the threshold for ``category``, falling back to the default."""
        return self._mapping.get(category, self.default_threshold)

    def to_dict(self) -> dict[str, float]:
        """Serialize explicit thresholds to a ``{category: threshold}`` mapping."""
        return dict(self._mapping)

    def evaluate(self, result: BaseResult, **kwargs) -> PolicyResult:
        """Evaluate a guard result against these thresholds."""
        violated: dict[str, bool] = {}
        for category, score in result.scores.items():
            violated[category] = score >= self.get_threshold(category)
        return PolicyResult(scores=result.scores, violated=violated, categories=list(result.scores))


__all__ = ["Policy"]
