from ..base_classes import BasePolicy
from .result import Result


class ContentPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def evaluate(self, result: Result, **kwargs) -> bool:
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            if threshold is None:
                continue
            if value >= self.mapping[key]:
                return True
        return False