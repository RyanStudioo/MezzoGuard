from .._types import BaseResult


class Result(BaseResult):
    def __init__(self, chunks: list[list[dict]], scores: dict[str, float]):
        self._chunks = chunks
        self.scores = scores

    @property
    def chunks(self) -> list[list[dict]]:
        return self._chunks


__all__ = ["Result"]
