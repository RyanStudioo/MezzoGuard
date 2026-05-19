from mezzoguard.base_classes import BaseResult
from .categories import Category

class Result(BaseResult):
    accepted_labels = [Category.SAFE, Category.UNSAFE]

    def __init__(self, chunks: list[dict], label: Category, confidence: float):
        self._chunks = chunks
        self.label = label
        self.confidence = confidence

        if self.label not in self.accepted_labels:
            raise ValueError(f"Invalid label: {self.label}. Accepted labels are: {self.accepted_labels}")

    def is_safe(self, threshold: float=0.5) -> bool:
        if self.label == Category.SAFE:
            return True
        elif self.label == Category.UNSAFE:
            if self.confidence < threshold:
                return True
            else:
                return False
        else:
            raise ValueError(f"Invalid label: {self.label}")


__all__ = ["Result"]