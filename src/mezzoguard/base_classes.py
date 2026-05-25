from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self, Any


class BaseConfig:
    """Base Config class"""
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):
        self.model_type = model_type


@dataclass
class BaseResult:
    """Base Result class"""
    pass


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
            return None
        return self._mapping[category]

    @abstractmethod
    def evaluate(self, result: BaseResult, **kwargs) -> bool:
        """Evaluate a result from a guard scan"""
        raise NotImplementedError