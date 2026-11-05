from .._types import BaseResult


class Result(BaseResult):
    def __init__(self, chunks: list[dict], scores: dict[str, float]):
        self._chunks = chunks
        self.scores = scores

    @property
    def chunks(self) -> list[dict]:
        return self._chunks

    def is_unsafe(self, threshold: float = 0.5) -> bool:
        unsafe_score = self.scores.get("unsafe", 0.0)
        return unsafe_score >= threshold


__all__ = ["Result"]
