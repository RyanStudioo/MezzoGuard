from .._types import BaseResult
from .categories import Category


class Result(BaseResult):
    def __init__(self, chunks: list[list[dict]], scores: dict[Category, float]):
        self._chunks = chunks
        self.scores = scores

    @property
    def chunks(self) -> list[list[dict]]:
        return self._chunks


__all__ = ["Result"]
