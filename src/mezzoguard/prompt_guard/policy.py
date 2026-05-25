from .categories import Category
from ..base_classes import BasePolicy
from .result import Result


class PromptPolicy(BasePolicy):
    """A policy that evaluates the safety of a prompt based on the scores of different categories."""

    def add_threshold(self, category: Category, threshold: float) -> "PromptPolicy":
        super().add_threshold(category, threshold)
        return self

    def evaluate(self, result: Result, **kwargs) -> bool:
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            if threshold is None:
                continue
            if value >= threshold:
                return True
        return False