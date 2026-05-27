from ..base_classes import BasePolicy, PolicyResult
from .categories import Category
from .result import Result


class ContentPolicy(BasePolicy):
    """A policy that evaluates the safety of content based on the scores of different categories."""

    def add_threshold(self, category: Category, threshold: float) -> "ContentPolicy":
        super().add_threshold(category, threshold)
        return self

    def evaluate(self, result: Result, **kwargs) -> PolicyResult:
        categories: dict[Category, bool] = {}
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            categories[key] = value >= threshold
        return PolicyResult(categories=categories)