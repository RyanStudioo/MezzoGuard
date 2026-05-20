from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self


class Config:
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):\
        self.model_type = model_type


@dataclass
class BaseResult:
    pass


class BasePolicy(ABC):
    def __init__(self):
        self.mapping = {}

    def add_threshold(self, category: str, threshold: float) -> Self:
        self.mapping[category] = threshold
        return self

    def get_threshold(self, category: str):
        if category not in self.mapping:
            return None
        return self.mapping[category]

    @abstractmethod
    def evaluate(self, result: BaseResult, **kwargs) -> bool:
        raise NotImplementedError