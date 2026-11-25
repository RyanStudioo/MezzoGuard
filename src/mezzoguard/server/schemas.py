from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .._types import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
)


class ChunkedTextsRequest(BaseModel):
    """Request fields shared by ``/v1/scan`` and ``/v1/redact``.

    ``overlap`` must stay strictly below ``max_seq_length`` once defaults are
    applied, mirroring the chunker's own ``ValueError``. Validating here keeps
    the invalid combination a ``422`` instead of a server error mid-request.
    """

    texts: list[str] = Field(..., min_length=1)
    max_seq_length: int | None = Field(default=None, ge=1)
    overlap: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _overlap_below_max_seq_length(self) -> ChunkedTextsRequest:
        max_seq_length = DEFAULT_MAX_SEQ_LENGTH if self.max_seq_length is None else self.max_seq_length
        overlap = DEFAULT_OVERLAP if self.overlap is None else self.overlap
        if overlap >= max_seq_length:
            raise ValueError(
                f"Invalid parameters: overlap ({overlap}) must be less than max_seq_length ({max_seq_length})"
            )
        return self


class ScanRequest(ChunkedTextsRequest):
    confidence: float = Field(default=DEFAULT_CONFIDENCE, ge=0.0, le=1.0)
    policy: dict[str, float] | None = None
    default_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    include_chunks: bool = False


class ScanResult(BaseModel):
    scores: dict[str, float]
    violated: list[str] = Field(default_factory=list)
    unsafe: bool = False
    chunks: list | None = None


class ScanResponse(BaseModel):
    results: list[ScanResult]


class RedactRequest(ChunkedTextsRequest):
    replace: str = DEFAULT_REDACTED_LABEL
    replacements: dict[str, str] | None = None
    confidence: float = Field(default=DEFAULT_CONFIDENCE, ge=0.0, le=1.0)
    policy: dict[str, float] | None = None
    default_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    return_spans: bool = False


class RedactedSpan(BaseModel):
    start: int
    end: int
    category: str | None = None
    replacement: str = ""


class RedactResult(BaseModel):
    text: str
    spans: list[RedactedSpan] | None = None


class RedactResponse(BaseModel):
    results: list[RedactResult]


class InfoResponse(BaseModel):
    model: str
    categories: list[str]
    default_max_seq_length: int
    default_overlap: int
    default_replace: str
    backend: str
    dtype: str | None
    device: str | None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model: str
