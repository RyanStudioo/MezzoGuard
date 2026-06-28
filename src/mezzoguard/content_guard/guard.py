import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .categories import Category
from .result import Result
from .config import MODELS_CONFIG, ContentGuardConfig
from .policy import ContentPolicy
from ..errors import UnsafeContentError
from ..model import GuardModel
from ..base_classes import ModelConfig, _init_guard_config, _make_redact_before_exec, _make_scan_before_exec
from .._types import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, DEFAULT_CONFIDENCE


class Guard(GuardModel):
    """A Content Guard Model"""
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, task="text-classification", **kwargs)

        self.model_config: ModelConfig | None = None
        self.model_config, self.config = _init_guard_config(
            name, Category, ContentGuardConfig, MODELS_CONFIG
        )

        readme_deprecation = ModelConfig.get_deprecation_from_readme(name)
        if readme_deprecation:
            warnings.warn(
                readme_deprecation["deprecated_message"],
                DeprecationWarning,
                stacklevel=2,
            )

    def _get_category_for_label(self, label: str) -> Category | None:
        try:
            return self.config.get_category_for_label(label)
        except ValueError:
            return None

    def _scores_from_chunk(self, chunk_result: list[dict]) -> dict[Category, float]:
        max_scores: dict[Category, float] = {}
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
        max_scores: dict[Category, float] = {}
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
        for category in Category:
            default_policy.add_threshold(category, confidence)
        return default_policy

    def _chunk_matches_policy(self, chunk_result: list[dict], policy: ContentPolicy) -> bool:
        scores = self._scores_from_chunk(chunk_result)
        result = policy.evaluate(Result(chunks=[chunk_result], scores=scores))
        return bool(result)

    def _on_unsafe_content(self, result: Result, confidence: float) -> None:
        violated = []
        for cat, score in result.scores.items():
            if score > confidence:
                violated.append({cat.value: score})
        if violated:
            raise UnsafeContentError(violated)

    def scan(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = [future.result() for future in futures]
        return self._from_prediction(chunk_results)

    async def async_scan(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)
        return self._from_prediction(chunk_results)

    def redact(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP,
               replace: str = DEFAULT_REDACTED_LABEL, policy: ContentPolicy | None = None, confidence: float = DEFAULT_CONFIDENCE, **kwargs: Any) -> str:
        policy = self._resolve_redaction_policy(policy, confidence)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]

            previous_flagged = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if self._chunk_matches_policy(result, policy):
                    if previous_flagged:
                        continue
                    redacted_chunks.append(replace)
                    previous_flagged = True
                else:
                    redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                    previous_flagged = False
        return " ".join(redacted_chunks)

    async def async_redact(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP,
                           replace: str = DEFAULT_REDACTED_LABEL, policy: ContentPolicy | None = None, confidence: float = DEFAULT_CONFIDENCE, **kwargs: Any) -> str:
        policy = self._resolve_redaction_policy(policy, confidence)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)

        redacted_chunks = []
        previous_flagged = False
        for chunk, result in zip(chunks, chunk_results):
            if self._chunk_matches_policy(result, policy):
                if previous_flagged:
                    continue
                redacted_chunks.append(replace)
                previous_flagged = True
            else:
                redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                previous_flagged = False
        return " ".join(redacted_chunks)

    def redact_before_exec(self, param: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP, replace: str = DEFAULT_REDACTED_LABEL,
                           policy: ContentPolicy | None = None, confidence: float = DEFAULT_CONFIDENCE, **kwargs: Any):
        return _make_redact_before_exec(self.redact, self.async_redact)(
            param, max_seq_length, overlap, replace, policy, confidence
        )

    def scan_before_exec(self, param: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP,
                         confidence: float = DEFAULT_CONFIDENCE):
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe_content)(
            param, max_seq_length, overlap, confidence
        )

__all__ = ["Guard"]
