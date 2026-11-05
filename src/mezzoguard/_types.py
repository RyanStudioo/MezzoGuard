from dataclasses import dataclass

DEFAULT_MAX_SEQ_LENGTH = 64
DEFAULT_OVERLAP = 16
DEFAULT_REDACTED_LABEL = "[REDACTED]"
DEFAULT_CONFIDENCE = 0.5
PIPELINE_TASK = "text-classification"


@dataclass
class BaseResult:
    """Base Result class"""

    pass


@dataclass
class Span:
    """A flagged region of text. `start` and `end` are character offsets into the original text."""

    start: int
    end: int
    category: str | None = None
    replacement: str = ""
