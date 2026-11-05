from __future__ import annotations

from typing import Any

from .._types import Span


def scores_to_dict(scores: dict[str, float]) -> dict[str, float]:
    """Serialize a ``{category: float}`` mapping to ``{str: float}``."""
    return {str(category): float(score) for category, score in scores.items()}


def span_to_dict(span: Span) -> dict[str, Any]:
    """Serialize a redaction span for JSON transport."""
    return {
        "start": span.start,
        "end": span.end,
        "category": span.category,
        "replacement": span.replacement,
    }
