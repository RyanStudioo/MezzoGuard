from dataclasses import dataclass
from enum import Enum

DEFAULT_MAX_SEQ_LENGTH = 64
DEFAULT_OVERLAP = 16
DEFAULT_REDACTED_LABEL = "[REDACTED]"
DEFAULT_CONFIDENCE = 0.5
PIPELINE_TASK = "text-classification"


class BaseCategory(Enum):
    """Base Category enum class for all guard categories."""
    pass


@dataclass
class BaseResult:
    """Base Result class"""
    pass
