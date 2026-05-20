from mezzoguard.base_classes import BasePolicy, BaseResult
from .result import Result


class PromptPolicy(BasePolicy):

    def evaluate(self, result: Result, **kwargs) -> bool:
        pass