import asyncio
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable

from . import Result
from .config import MODELS_CONFIG, PromptGuardConfig
from .categories import Category
from .policy import PromptPolicy
from ..errors import UnsafePromptError
from ..model import GuardModel
from ..base_classes import ModelConfig, _init_guard_config, _make_redact_before_exec, _make_scan_before_exec
from .._types import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, DEFAULT_CONFIDENCE


class Guard(GuardModel):
    """A Prompt Guard Model"""
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, task="text-classification", **kwargs)

        self.model_config: ModelConfig | None = None
        self.model_config, self.config = _init_guard_config(
            name, Category, PromptGuardConfig, MODELS_CONFIG
        )

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

    def _resolve_redaction_policy(
        self, policy: PromptPolicy | None, confidence: float
    ) -> PromptPolicy:
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

    def scan(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        return self._from_prediction(results)

    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, self._predict_tokenized_text, chunk)
                for chunk in chunks
            ]
            results = await asyncio.gather(*tasks)
        return self._from_prediction(results)

    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: PromptPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> str:
        policy = self._resolve_redaction_policy(policy, confidence)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]

            previous_unsafe = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if self._chunk_matches_policy(result, policy):
                    if previous_unsafe:
                        continue
                    redacted_chunks.append(replace)
                    previous_unsafe = True
                else:
                    redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                    previous_unsafe = False
        return " ".join(redacted_chunks)

    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: PromptPolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> str:
        policy = self._resolve_redaction_policy(policy, confidence)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, self._predict_tokenized_text, chunk)
                for chunk in chunks
            ]
            chunk_results = await asyncio.gather(*tasks)

        redacted_chunks = []
        previous_unsafe = False
        for chunk, result in zip(chunks, chunk_results):
            if self._chunk_matches_policy(result, policy):
                if previous_unsafe:
                    continue
                redacted_chunks.append(replace)
                previous_unsafe = True
            else:
                redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                previous_unsafe = False
        return " ".join(redacted_chunks)

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
        self, param: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP, confidence: float = DEFAULT_CONFIDENCE
    ) -> Callable:
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe_prompt)(
            param, max_seq_length, overlap, confidence
        )


__all__ = ["Guard"]
