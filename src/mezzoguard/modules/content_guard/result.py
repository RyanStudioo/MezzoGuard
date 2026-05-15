from .categories import ModerationCategory

from ...results import Result


class ContentGuardResult(Result):
    def __init__(self, chunks: list[list[dict]], violations: dict[ModerationCategory, bool]):
        self._chunks = chunks
        self.violations = violations

    def is_safe(self) -> bool:
        return not any(self.violations.values())

    def check_category_violation(self, category: ModerationCategory) -> bool:
        return self.violations[category]

    @property
    def flagged_categories(self) -> list[ModerationCategory]:
        return [cat for cat, violated in self.violations.items() if violated]
