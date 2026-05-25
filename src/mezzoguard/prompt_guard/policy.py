from .categories import Category
from ..base_classes import BasePolicy
from .result import Result


class PromptPolicy(BasePolicy):

    def add_threshold(self, category: Category, threshold: float) -> "PromptPolicy":
        super().add_threshold(category, threshold)
        return self

    def evaluate(self, result: Result, **kwargs) -> bool:
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            if threshold is None:
                continue
            if value >= self.mapping[key]:
                return True
        return False