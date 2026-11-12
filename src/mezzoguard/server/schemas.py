from __future__ import annotations

from pydantic import BaseModel, Field

from .._types import DEFAULT_CONFIDENCE, DEFAULT_REDACTED_LABEL


class ScanRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    max_seq_length: int | None = Field(default=None, ge=1)
    overlap: int | None = Field(default=None, ge=0)
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


class RedactRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    max_seq_length: int | None = Field(default=None, ge=1)
    overlap: int | None = Field(default=None, ge=0)
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
