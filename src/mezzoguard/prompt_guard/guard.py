import asyncio
import functools
import inspect
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Callable
import warnings

from transformers import pipeline

from . import Result
from .config import MODELS_CONFIG, PromptGuardConfig
from .categories import Category
from .policy import PromptPolicy
from ..errors import UnsafePromptError
from ..model import GuardModel


class Guard(GuardModel):
    """A Prompt Guard Model"""
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, task="text-classification", **kwargs)

        self.config: PromptGuardConfig = MODELS_CONFIG.get(name, None)
        if not self.config:
            warnings.warn(
                f"No preset config found for model {self.name}. You may need to provide a custom config."
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
        self, policy: Optional[PromptPolicy], confidence: float
    ) -> PromptPolicy:
        if policy is not None:
            return policy
        return PromptPolicy().add_threshold(Category.UNSAFE, confidence)

    def _chunk_matches_policy(self, chunk_result: dict, policy: PromptPolicy) -> bool:
        result = self._from_prediction([chunk_result])
        return bool(policy.evaluate(result))

    def scan(self, text: str, max_seq_length: int = 64, overlap: int = 16) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        return self._from_prediction(results)

    async def async_scan(
        self, text: str, max_seq_length: int = 64, overlap: int = 16
    ) -> Result:
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
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
        max_seq_length: int = 64,
        overlap: int = 16,
        replace: str = "[REDACTED]",
        policy: Optional[PromptPolicy] = None,
        confidence: float = 0.5,
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
        max_seq_length: int = 64,
        overlap: int = 16,
        replace: str = "[REDACTED]",
        policy: Optional[PromptPolicy] = None,
        confidence: float = 0.5,
    ) -> str:
        policy = self._resolve_redaction_policy(policy, confidence)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
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
        max_seq_length: int = 64,
        overlap: int = 16,
        replace: str = "[REDACTED]",
        policy: Optional[PromptPolicy] = None,
        confidence: float = 0.5,
    ) -> Callable:

        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    value = bound.arguments.get(param)

                    if value is not None:
                        redacted = await self.async_redact(
                            value,
                            max_seq_length,
                            overlap,
                            replace,
                            policy=policy,
                            confidence=confidence,
                        )
                        bound.arguments[param] = redacted

                    return await func(*bound.args, **bound.kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    value = bound.arguments.get(param)

                    if value is not None:
                        redacted = self.redact(
                            value,
                            max_seq_length,
                            overlap,
                            replace,
                            policy=policy,
                            confidence=confidence,
                        )
                        bound.arguments[param] = redacted

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator

    def scan_before_exec(
        self, param: str, max_seq_length: int = 64, overlap: int = 16, confidence: float = 0.5
    ) -> Callable:

        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    value = bound.arguments.get(param)

                    if value is not None:
                        result = await self.async_scan(value, max_seq_length, overlap)
                        unsafe_score = result.scores.get(Category.UNSAFE, 0.0)
                        if unsafe_score >= confidence:
                            raise UnsafePromptError(unsafe_score)

                    return await func(*bound.args, **bound.kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    value = bound.arguments.get(param)

                    if value is not None:
                        result = self.scan(value, max_seq_length, overlap)
                        unsafe_score = result.scores.get(Category.UNSAFE, 0.0)
                        if unsafe_score >= confidence:
                            raise UnsafePromptError(unsafe_score)

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator


__all__ = ["Guard"]