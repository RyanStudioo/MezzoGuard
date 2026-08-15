from ..base_classes import BasePolicy
from .categories import Category


class ContentPolicy(BasePolicy):
    """A policy that evaluates the safety of content based on the scores of different categories."""

    def add_threshold(self, category: Category, threshold: float) -> "ContentPolicy":
        super().add_threshold(category, threshold)
        return self
