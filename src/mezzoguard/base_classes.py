from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Self, Any


class BaseConfig:
    """Base Config class"""
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):
        self.model_type = model_type


@dataclass
class BaseResult:
    """Base Result class"""
    pass

class PolicyResult(BaseResult):
    def __init__(self, categories: dict[Enum, bool]):
        self.categories = categories

    def __bool__(self) -> bool:
        return self.is_unsafe()

    def __repr__(self) -> str:
        safe = [k.name for k, v in self.categories.items() if v]
        violated = [k.name for k, v in self.categories.items() if not v]
        return f"PolicyResult(safe={safe}, violated={violated})"

    def is_safe(self) -> bool:
        return all(self.categories.values())

    def is_unsafe(self) -> bool:
        return not all(self.categories.values())

    def get_violated_categories(self) -> list[Enum]:
        return [category for category, is_safe in self.categories.items() if not is_safe]

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