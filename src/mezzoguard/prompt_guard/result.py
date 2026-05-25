from ..base_classes import BaseResult
from .categories import Category

class Result(BaseResult):
    def __init__(self, chunks: list[dict], scores: dict[Category, float]):
        self._chunks = chunks
        self.scores = scores

    def is_safe(self, threshold: float = 0.5) -> bool:
        unsafe_score = self.scores.get(Category.UNSAFE, 0.0)
        return unsafe_score < threshold


__all__ = ["Result"]