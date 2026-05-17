from .categories import Category

from ...base_classes import BaseResult


class Result(BaseResult):
    def __init__(self, chunks: list[list[dict]], violations: dict[Category, bool]):
        self._chunks = chunks
        self.violations = violations

    def is_safe(self) -> bool:
        return not any(self.violations.values())

    def check_category_violation(self, category: Category) -> bool:
        return self.violations[category]

    @property
    def flagged_categories(self) -> list[Category]:
        return [cat for cat, violated in self.violations.items() if violated]

__all__ = ["Result"]