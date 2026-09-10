from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException

from .. import __version__
from .._types import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, BaseCategory
from ..base_classes import BasePolicy
from ..content_guard import Category as ContentCategory
from ..content_guard import ContentPolicy
from ..model import GuardModel
from ..prompt_guard import Category as PromptCategory
from ..prompt_guard import PromptPolicy
from .schemas import (
    HealthResponse,
    InfoResponse,
    RedactedSpan,
    RedactRequest,
    RedactResponse,
    RedactResult,
    ScanRequest,
    ScanResponse,
    ScanResult,
)
from .serialize import category_value, scores_to_dict, span_to_dict

ModelType = Literal["prompt_guard", "content_guard"]

_CATEGORY_ENUMS: dict[str, type[BaseCategory]] = {
    "prompt_guard": PromptCategory,
    "content_guard": ContentCategory,
}
_POLICY_CLASSES: dict[str, type[BasePolicy]] = {
    "prompt_guard": PromptPolicy,
    "content_guard": ContentPolicy,
}


def _category_for(model_type: str, name: str) -> BaseCategory | None:
    try:
        return _CATEGORY_ENUMS[model_type](name)
    except ValueError:
        return None


def _build_policy(model_type: str, mapping: dict[str, float] | None, confidence: float) -> BasePolicy:
    """Build a guard policy from wire thresholds, or a confidence default."""
    policy = _POLICY_CLASSES[model_type]()
    if mapping is None:
        for category in _CATEGORY_ENUMS[model_type]:
            policy.add_threshold(category, confidence)
        return policy
    for name, threshold in mapping.items():
        category = _category_for(model_type, name)
        if category is not None:
            policy.add_threshold(category, threshold)
    return policy


def _dtype_name(guard: GuardModel) -> str | None:
    if guard.dtype is None:
        return None
    return str(guard.dtype).replace("torch.", "")


def _backend_name(guard: GuardModel) -> str:
    if guard.use_onnx:
        return "onnx"
    if guard.torch_compile:
        return "torch-compile"
    return "torch"


def create_app(
    guard: GuardModel,
    *,
    model_type: ModelType,
    max_batch_texts: int = 64,
    max_text_chars: int = 50_000,
) -> FastAPI:
    """Create a FastAPI app exposing ``guard`` over HTTP.

    ``guard`` is expected to be already loaded; the app ejects it on shutdown.
    """
    if model_type not in _CATEGORY_ENUMS:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'prompt_guard' or 'content_guard'.")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        guard.eject_model()

    app = FastAPI(title="Mezzo Guard", version=__version__, lifespan=lifespan)

    def validate_batch(texts: list[str]) -> None:
        if len(texts) > max_batch_texts:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: {len(texts)} texts (max {max_batch_texts})",
            )
        for text in texts:
            if len(text) > max_text_chars:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text too long: {len(text)} characters (max {max_text_chars})",
                )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=guard.pipeline is not None,
            model=guard.name,
            model_type=model_type,
        )

    @app.get("/v1/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        return InfoResponse(
            model=guard.name,
            model_type=model_type,
            categories=[category.value for category in _CATEGORY_ENUMS[model_type]],
            default_max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
            default_overlap=DEFAULT_OVERLAP,
            default_replace=DEFAULT_REDACTED_LABEL,
            backend=_backend_name(guard),
            dtype=_dtype_name(guard),
            device=guard.device,
        )

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(request: ScanRequest) -> ScanResponse:
        validate_batch(request.texts)
        max_seq_length = request.max_seq_length if request.max_seq_length is not None else DEFAULT_MAX_SEQ_LENGTH
        overlap = request.overlap if request.overlap is not None else DEFAULT_OVERLAP
        results = await guard.async_batch_scan(request.texts, max_seq_length, overlap)

        payload = []
        for result in results:
            if request.policy is not None:
                policy = _build_policy(model_type, request.policy, request.confidence)
                violated = [category_value(c) for c in policy.evaluate(result).get_violated_categories()]
            else:
                violated = [category_value(c) for c, score in result.scores.items() if score >= request.confidence]
            payload.append(
                ScanResult(
                    scores=scores_to_dict(result.scores),
                    violated=violated,
                    unsafe=bool(violated),
                    chunks=result.chunks if request.include_chunks else None,
                )
            )
        return ScanResponse(model_type=model_type, results=payload)

    @app.post("/v1/redact", response_model=RedactResponse)
    async def redact(request: RedactRequest) -> RedactResponse:
        validate_batch(request.texts)
        max_seq_length = request.max_seq_length if request.max_seq_length is not None else DEFAULT_MAX_SEQ_LENGTH
        overlap = request.overlap if request.overlap is not None else DEFAULT_OVERLAP

        if request.replacements:
            replace: str | dict[BaseCategory, str] = {}
            for name, value in request.replacements.items():
                category = _category_for(model_type, name)
                if category is not None:
                    replace[category] = value
        else:
            replace = request.replace

        policy = _build_policy(model_type, request.policy, request.confidence) if request.policy is not None else None
        results = await guard.async_batch_redact(
            request.texts,
            max_seq_length,
            overlap,
            replace,
            policy=policy,
            confidence=request.confidence,
            return_spans=True,
        )
        payload = [
            RedactResult(
                text=redacted,
                spans=[RedactedSpan(**span_to_dict(span)) for span in spans] if request.return_spans else None,
            )
            for redacted, spans in results
        ]
        return RedactResponse(model_type=model_type, results=payload)

    return app
