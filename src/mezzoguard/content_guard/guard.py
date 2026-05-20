import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any
import warnings

from .categories import Category
from .result import Result
from .config import MODELS_CONFIG, ContentGuardConfig
from mezzoguard.errors import UnsafePromptError
from mezzoguard.model import GuardModel


class Guard(GuardModel):
    def __init__(self, name: str):
        super().__init__(name, task="text-classification")
        self.config: ContentGuardConfig = MODELS_CONFIG[self.name]
        if not self.config.mappings:
            warnings.warn(f"No preset config found for model {self.name}. You may need to provide a custom config.")

    def _get_category_for_label(self, label: str) -> Category | None:
        try:
            return self.config.get_category_for_label(label)
        except ValueError:
            return None

    def _from_prediction(self, results: list[list[dict]]):
        max_scores: dict[Category, float] = {}
        for chunk_result in results:
            for pred in chunk_result:
                label = pred.get("label", "")
                score = pred.get("score", 0.0)
                category = self._get_category_for_label(label)
                if category is not None:
                    if category not in max_scores or score > max_scores[category]:
                        max_scores[category] = score

        return Result(chunks=results, scores=max_scores)

    def _chunk_has_violation(self, chunk_result: list[dict], confidence: float) -> bool:
        for result in chunk_result:
            label = result.get("label", "")
            category = self._get_category_for_label(label)
            if category is None:
                continue
            score = result.get("score", 0.0)
            if score >= confidence:
                return True
        return False

    def scan(self, text: str, max_seq_length: int = 64, overlap: int = 8) -> Result:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = [future.result() for future in futures]
        return self._from_prediction(chunk_results)

    async def async_scan(self, text: str, max_seq_length: int = 64, overlap: int = 8) -> Result:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)
        return self._from_prediction(chunk_results)

    def redact(self, text: str, max_seq_length: int = 64, overlap: int = 16,
               replace: str = "[REDACTED]", confidence: float = 0.5, **kwargs: Any) -> str:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]

            previous_flagged = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if self._chunk_has_violation(result, confidence):
                    if previous_flagged:
                        continue
                    redacted_chunks.append(replace)
                    previous_flagged = True
                else:
                    redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                    previous_flagged = False
        return " ".join(redacted_chunks)

    async def async_redact(self, text: str, max_seq_length: int = 64, overlap: int = 16,
                           replace: str = "[REDACTED]", confidence: float = 0.5, **kwargs: Any) -> str:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)

        redacted_chunks = []
        previous_flagged = False
        for chunk, result in zip(chunks, chunk_results):
            if self._chunk_has_violation(result, confidence):
                if previous_flagged:
                    continue
                redacted_chunks.append(replace)
                previous_flagged = True
            else:
                redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                previous_flagged = False
        return " ".join(redacted_chunks)

    def redact_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]", **kwargs: Any) -> Callable:
        self.load_model()

        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    value = bound.arguments.get(param)

                    if value is not None:
                        redacted = await self.async_redact(value, max_seq_length, overlap, replace)
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
                        redacted = self.redact(value, max_seq_length, overlap, replace)
                        bound.arguments[param] = redacted

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator

    def scan_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16,
                         confidence: float = 0.5) -> Callable:
        self.load_model()

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
                        if any(score >= confidence for score in result.scores.values()):
                            raise UnsafePromptError(confidence)

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
                        if any(score >= confidence for score in result.scores.values()):
                            raise UnsafePromptError(confidence)

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator

__all__ = ["Guard"]