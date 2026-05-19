from .categories import Category
from mezzoguard.base_classes import BaseResult


class Result(BaseResult):
    def __init__(self, chunks: list[list[dict]], scores: dict[Category, float]):
        self._chunks = chunks
        self.scores = scores

__all__ = ["Result"]