from .categories import Category
from ..base_classes import BasePolicy
from .result import Result


class ContentPolicy(BasePolicy):

    def add_threshold(self, category: Category, threshold: float) -> "ContentPolicy":
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