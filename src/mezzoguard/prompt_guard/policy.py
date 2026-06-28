from ..base_classes import BasePolicy, PolicyResult
from .categories import Category
from .result import Result


class PromptPolicy(BasePolicy):
    """A policy that evaluates the safety of a prompt based on the scores of different categories."""

    def add_threshold(self, category: Category, threshold: float) -> "PromptPolicy":
        super().add_threshold(category, threshold)
        return self
