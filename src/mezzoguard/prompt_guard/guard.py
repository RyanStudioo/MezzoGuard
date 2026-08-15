import asyncio
import functools
import warnings
from collections.abc import Callable

from .._types import DEFAULT_CONFIDENCE, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, Span
from ..base_classes import (
    ModelConfig,
    _init_guard_config,
    _make_redact_before_exec,
    _make_scan_before_exec,
    _redact_text_with_spans,
)
from ..errors import UnsafePromptError
from ..model import GuardModel
from .categories import Category
from .config import MODELS_CONFIG, PromptGuardConfig
from .policy import PromptPolicy
from .result import Result


class Guard(GuardModel):
    """A Prompt Guard Model"""

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, task="text-classification", **kwargs)

        self.model_config: ModelConfig | None = None
        self.model_config, self.config = _init_guard_config(name, Category, PromptGuardConfig, MODELS_CONFIG)

        if self.model_config is not None and self.model_config.deprecated:
            warnings.warn(
                self.model_config.deprecated_message,
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            readme_deprecation = ModelConfig.get_deprecation_from_readme(name)
            if readme_deprecation:
                warnings.warn(
                    readme_deprecation["deprecated_message"],
                    DeprecationWarning,
                    stacklevel=2,
                )

    def _from_prediction(self, chunks: list[dict]) -> Result:
        unsafe_score = 0.0
        safe_category = self.config.safe_category
        for chunk in chunks:
            label = chunk.get("label", "")
            score = chunk.get("score", 0.0)
            category = self.config.get_category_for_label(label)
            if category != safe_category and score > unsafe_score:
                unsafe_score = score
        scores = {
            Category.UNSAFE: unsafe_score,
        }
        return Result(chunks=chunks, scores=scores)

    def _resolve_redaction_policy(self, policy: PromptPolicy | None, confidence: float) -> PromptPolicy:
        if policy is not None:
            return policy
        return PromptPolicy().add_threshold(Category.UNSAFE, confidence)

    def _chunk_matches_policy(self, chunk_result: dict, policy: PromptPolicy) -> bool:
        result = self._from_prediction([chunk_result])
        return bool(policy.evaluate(result))

    def _on_unsafe_prompt(self, result: Result, confidence: float) -> None:
        unsafe_score = result.scores.get(Category.UNSAFE, 0.0)
        if unsafe_score >= confidence:
            raise UnsafePromptError(unsafe_score)

    def _chunk_texts(self, text: str, max_seq_length: int, overlap: int) -> tuple[list[tuple[int, int]], list[str]]:
        spans = self._chunk_text(text, max_seq_length, overlap)
        return spans, [text[start:end] for start, end in spans]

    def scan(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP) -> Result:
        spans, chunk_texts = self._chunk_texts(text, max_seq_length, overlap)
        if not chunk_texts:
            return self._from_prediction([])
        results = self.pipeline(chunk_texts)
        return self._from_prediction(results)

    def scan_many(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]:
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        if not flat:
            return [self._from_prediction([]) for _ in texts]
        results = self.pipeline([chunk_text for _, chunk_text in flat])
        per_text: list[list[dict]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, results, strict=True):
            per_text[idx].append(chunk_result)
        return [self._from_prediction(chunk_results) for chunk_results in per_text]

    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> Result:
        spans, chunk_texts = self._chunk_texts(text, max_seq_length, overlap)
        if not chunk_texts:
            return self._from_prediction([])
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, functools.partial(self.pipeline, chunk_texts))
        return self._from_prediction(results)

    async def async_scan_many(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]:
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        if not flat:
            return [self._from_prediction([]) for _ in texts]
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, functools.partial(self.pipeline, [chunk_text for _, chunk_text in flat])
        )
        per_text: list[list[dict]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, results, strict=True):
            per_text[idx].append(chunk_result)
        return [self._from_prediction(chunk_results) for chunk_results in per_text]

    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[Category, str] = DEFAULT_REDACTED_LABEL,
        policy: PromptPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
    ) -> str | tuple[str, list[Span]]:
        policy = self._resolve_redaction_policy(policy, confidence)
        spans, chunk_texts = self._chunk_texts(text, max_seq_length, overlap)
        if not chunk_texts:
            return ("", []) if return_spans else ""
        results = self.pipeline(chunk_texts)

        flags = []
        for chunk_result in results:
            if self._chunk_matches_policy(chunk_result, policy):
                flags.append((True, Category.UNSAFE))
            else:
                flags.append((False, None))

        redacted, redaction_spans = _redact_text_with_spans(text, spans, flags, replace)
        if return_spans:
            return redacted, redaction_spans
        return redacted

    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[Category, str] = DEFAULT_REDACTED_LABEL,
        policy: PromptPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
    ) -> str | tuple[str, list[Span]]:
        policy = self._resolve_redaction_policy(policy, confidence)
        spans, chunk_texts = self._chunk_texts(text, max_seq_length, overlap)
        if not chunk_texts:
            return ("", []) if return_spans else ""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, functools.partial(self.pipeline, chunk_texts))

        flags = []
        for chunk_result in results:
            if self._chunk_matches_policy(chunk_result, policy):
                flags.append((True, Category.UNSAFE))
            else:
                flags.append((False, None))

        redacted, redaction_spans = _redact_text_with_spans(text, spans, flags, replace)
        if return_spans:
            return redacted, redaction_spans
        return redacted

    def redact_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: PromptPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Callable:
        return _make_redact_before_exec(self.redact, self.async_redact)(
            param, max_seq_length, overlap, replace, policy, confidence
        )

    def scan_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Callable:
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe_prompt)(
            param, max_seq_length, overlap, confidence
        )


__all__ = ["Guard"]
