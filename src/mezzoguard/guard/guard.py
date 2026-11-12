from __future__ import annotations

import asyncio
import functools
import warnings
from typing import Any

from .._types import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
    Span,
)
from ..base_classes import GuardBase, ModelConfig, _redact_text_with_spans, _warn_if_deprecated
from ..model import Model
from .config import MODELS_CONFIG, GuardConfig
from .policy import Policy
from .result import Result


class Guard(GuardBase, Model):
    """Unified prompt and content guard.

    The model's category configuration decides how predictions are exposed:

    * prompt models list only ``"unsafe"`` and produce ``{"unsafe": score}``;
    * content models produce one score per configured category.

    Both use the same chunking, caching, redaction and async machinery. Set
    thresholds per category with ``Policy`` or use ``Result.is_unsafe``:

    .. code-block:: python

        result = guard.scan(text)
        result.is_unsafe(threshold=0.5)
        result.is_unsafe(policy=Policy(default_threshold=0.5).add_threshold("violence", 0.7))
    """

    def __init__(self, name: str, config: GuardConfig | dict | None = None, **kwargs: Any):
        super().__init__(name=name, task="text-classification", **kwargs)

        self.model_config = ModelConfig.from_model_name(name)
        if isinstance(config, dict):
            config = GuardConfig(**config)
        if config is None:
            config = MODELS_CONFIG.get(name)
        if config is None and self.model_config is not None:
            config = GuardConfig(categories=self.model_config.categories)
        if config is None:
            warnings.warn(
                f"No preset config found for model {name}. You may need to provide a custom config.",
                stacklevel=2,
            )
        self.config = config

        _warn_if_deprecated(self.model_config, name)

    @property
    def categories(self) -> list[str]:
        return list(self.config.categories) if self.config is not None else []

    def _get_category_for_label(self, label: str) -> str | None:
        if self.config is None:
            return None
        try:
            return self.config.get_category_for_label(label)
        except ValueError:
            return None

    @staticmethod
    def _normalize_chunk(chunk: dict | list[dict]) -> list[dict]:
        """Accept both top-1 (dict) and top-k/all-labels (list) chunk outputs."""
        return [chunk] if isinstance(chunk, dict) else list(chunk)

    def _scores_from_chunks(self, chunks: list[list[dict]], initialize: bool = False) -> dict[str, float]:
        scores = {category: 0.0 for category in self.categories} if initialize else {}
        for chunk in chunks:
            for pred in chunk:
                category = self._get_category_for_label(pred.get("label", ""))
                if category is None:
                    continue
                score = pred.get("score", 0.0)
                if category not in scores or score > scores[category]:
                    scores[category] = score
        return scores

    def _from_prediction(self, chunks: list[dict] | list[list[dict]]) -> Result:
        normalized = [self._normalize_chunk(chunk) for chunk in chunks]
        return Result(chunks=normalized, scores=self._scores_from_chunks(normalized, initialize=True))

    def _resolve_redaction_policy(self, policy: Policy | None, confidence: float) -> Policy:
        if policy is not None:
            return policy
        return Policy(default_threshold=confidence)

    def _chunk_violation(self, chunk_result: list[dict], policy: Policy) -> str | None:
        normalized = self._normalize_chunk(chunk_result)
        scores = self._scores_from_chunks([normalized])
        verdict = policy.evaluate(Result(chunks=[normalized], scores=scores))
        if not verdict:
            return None
        violated = verdict.get_violated_categories()
        if not violated:
            return None
        return max(violated, key=lambda category: scores.get(category, 0.0))

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
        policy: Policy | None = None,
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
        policy: Policy | None = None,
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


__all__ = ["Guard"]
