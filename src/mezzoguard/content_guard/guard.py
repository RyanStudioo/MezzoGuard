import asyncio
import functools
import warnings
from typing import Any

from .._types import DEFAULT_CONFIDENCE, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, Span
from ..base_classes import (
    ModelConfig,
    _init_guard_config,
    _make_redact_before_exec,
    _make_scan_before_exec,
    _redact_text_with_spans,
)
from ..errors import UnsafeContentError
from ..model import GuardModel
from .config import MODELS_CONFIG, ContentGuardConfig
from .policy import ContentPolicy
from .result import Result


class Guard(GuardModel):
    """A Content Guard Model"""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, task="text-classification", **kwargs)

        self.model_config: ModelConfig | None = None
        self.model_config, self.config = _init_guard_config(name, ContentGuardConfig, MODELS_CONFIG)

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

    def _get_category_for_label(self, label: str) -> str | None:
        try:
            return self.config.get_category_for_label(label)
        except ValueError:
            return None

    def _scores_from_chunk(self, chunk_result: list[dict]) -> dict[str, float]:
        max_scores: dict[str, float] = {}
        for pred in chunk_result:
            label = pred.get("label", "")
            category = self._get_category_for_label(label)
            if category is None:
                continue
            score = pred.get("score", 0.0)
            if category not in max_scores or score > max_scores[category]:
                max_scores[category] = score
        return max_scores

    def _from_prediction(self, results: list[list[dict]]) -> Result:
        max_scores: dict[str, float] = {}
        for chunk_result in results:
            chunk_scores = self._scores_from_chunk(chunk_result)
            for category, score in chunk_scores.items():
                if category not in max_scores or score > max_scores[category]:
                    max_scores[category] = score
        return Result(chunks=results, scores=max_scores)

    def _resolve_redaction_policy(self, policy: ContentPolicy | None, confidence: float) -> ContentPolicy:
        if policy is not None:
            return policy
        default_policy = ContentPolicy()
        if self.config is not None:
            for category in self.config.categories:
                default_policy.add_threshold(category, confidence)
        return default_policy

    def _chunk_violation(self, chunk_result: list[dict], policy: ContentPolicy) -> str | None:
        scores = self._scores_from_chunk(chunk_result)
        verdict = policy.evaluate(Result(chunks=[chunk_result], scores=scores))
        if not verdict:
            return None
        violated = verdict.get_violated_categories()
        if not violated:
            return None
        return max(violated, key=lambda category: scores.get(category, 0.0))

    def _on_unsafe_content(self, result: Result, confidence: float) -> None:
        violated = []
        for cat, score in result.scores.items():
            if score >= confidence:
                violated.append({cat: score})
        if violated:
            raise UnsafeContentError(violated)

    def batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]:
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        if not flat:
            return [self._from_prediction([]) for _ in texts]
        chunk_results = self._predict([chunk_text for _, chunk_text in flat], top_k=None)
        per_text: list[list[list[dict]]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, chunk_results, strict=True):
            per_text[idx].append(chunk_result)
        return [self._from_prediction(chunks) for chunks in per_text]

    async def async_batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]:
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        if not flat:
            return [self._from_prediction([]) for _ in texts]
        loop = asyncio.get_running_loop()
        chunk_results = await loop.run_in_executor(
            None,
            functools.partial(self._predict, [chunk_text for _, chunk_text in flat], top_k=None),
        )
        per_text: list[list[list[dict]]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, chunk_results, strict=True):
            per_text[idx].append(chunk_result)
        return [self._from_prediction(chunks) for chunks in per_text]

    def batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: ContentPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]:
        policy = self._resolve_redaction_policy(policy, confidence)
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        chunk_results = self._predict([chunk_text for _, chunk_text in flat], top_k=None) if flat else []

        per_text: list[list[list[dict]]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, chunk_results, strict=True):
            per_text[idx].append(chunk_result)

        outputs = []
        for text, spans, results in zip(texts, all_spans, per_text, strict=True):
            flags = []
            for chunk_result in results:
                category = self._chunk_violation(chunk_result, policy)
                flags.append((category is not None, category))
            redacted, redaction_spans = _redact_text_with_spans(text, spans, flags, replace)
            outputs.append((redacted, redaction_spans) if return_spans else redacted)
        return outputs

    async def async_batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: ContentPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]:
        policy = self._resolve_redaction_policy(policy, confidence)
        all_spans = [self._chunk_text(t, max_seq_length, overlap) for t in texts]
        flat = [(idx, texts[idx][start:end]) for idx, spans in enumerate(all_spans) for start, end in spans]
        loop = asyncio.get_running_loop()
        chunk_results = (
            await loop.run_in_executor(
                None, functools.partial(self._predict, [chunk_text for _, chunk_text in flat], top_k=None)
            )
            if flat
            else []
        )

        per_text: list[list[list[dict]]] = [[] for _ in texts]
        for (idx, _), chunk_result in zip(flat, chunk_results, strict=True):
            per_text[idx].append(chunk_result)

        outputs = []
        for text, spans, results in zip(texts, all_spans, per_text, strict=True):
            flags = []
            for chunk_result in results:
                category = self._chunk_violation(chunk_result, policy)
                flags.append((category is not None, category))
            redacted, redaction_spans = _redact_text_with_spans(text, spans, flags, replace)
            outputs.append((redacted, redaction_spans) if return_spans else redacted)
        return outputs

    def redact_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: ContentPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        **kwargs: Any,
    ):
        return _make_redact_before_exec(self.redact, self.async_redact)(
            param, max_seq_length, overlap, replace, policy, confidence
        )

    def scan_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe_content)(
            param, max_seq_length, overlap, confidence
        )


__all__ = ["Guard"]
