from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self


class Config:
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):
        self.model_type = model_type


@dataclass
class BaseResult:
    pass


class BasePolicy(ABC):
    def __init__(self):
        self._mapping = {}

    def add_threshold(self, category: str, threshold: float) -> Self:
        self._mapping[category] = threshold
        return self

    def get_threshold(self, category: str) -> float:
        if category not in self._mapping:
            return None
        return self._mapping[category]

    @abstractmethod
    def evaluate(self, result: BaseResult, **kwargs) -> bool:
        raise NotImplementedError