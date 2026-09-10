from __future__ import annotations

from typing import Any

from .._types import Span


def category_value(category: Any) -> str:
    """Serialize a category (Enum or string) to a plain string."""
    return getattr(category, "value", str(category))


def scores_to_dict(scores: dict) -> dict[str, float]:
    """Serialize a ``{BaseCategory: float}`` mapping to ``{str: float}``."""
    return {category_value(category): float(score) for category, score in scores.items()}


def span_to_dict(span: Span) -> dict[str, Any]:
    """Serialize a redaction span for JSON transport."""
    return {
        "start": span.start,
        "end": span.end,
        "category": category_value(span.category) if span.category is not None else None,
        "replacement": span.replacement,
    }
