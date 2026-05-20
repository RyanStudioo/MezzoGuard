from mezzoguard.base_classes import BasePolicy, BaseResult
from .result import Result


class PromptPolicy(BasePolicy):

    def evaluate(self, result: Result, **kwargs) -> bool:
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            if threshold is None:
                continue
            if value >= self.mapping[key]:
                return True
        return False