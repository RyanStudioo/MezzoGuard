from abc import ABC, abstractmethod
from typing import Literal, Self, Any

from ._types import BaseResult, Category


class BaseConfig:
    """Base Config class"""
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):
        self.model_type = model_type


class PolicyResult(BaseResult):
    def __init__(self, scores: dict[Category, float], violated: dict[Category, bool], categories: list[Category]):
        self.scores = scores
        self.violated = violated
        self.categories = categories

    def __bool__(self) -> bool:
        return self.is_unsafe()

    def __repr__(self) -> str:
        violated = [k.name for k, v in self.violated.items() if v]
        safe = [k.name for k, v in self.violated.items() if not v]
        return f"PolicyResult(safe={safe}, violated={violated})"

    def is_safe(self) -> bool:
        return not any(self.violated.values())

    def is_unsafe(self) -> bool:
        return any(self.violated.values())

    def get_violated_categories(self) -> list[Category]:
        return [category for category, violated in self.violated.items() if violated]

class BasePolicy(ABC):
    """Base Policy class"""
    def __init__(self):
        self._mapping = {}

    def add_threshold(self, category: Any, threshold: float) -> Self:
        """Add a threshold to a category"""
        self._mapping[category] = threshold
        return self

    def get_threshold(self, category: Any) -> float:
        """Get the threshold of a category"""
        if category not in self._mapping:
            return 0.0
        return self._mapping[category]

    @abstractmethod
    def evaluate(self, result: BaseResult, **kwargs) -> PolicyResult:
        """Evaluate a result from a guard scan"""
        raise NotImplementedError