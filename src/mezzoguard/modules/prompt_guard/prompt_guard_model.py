import asyncio
import functools
import inspect
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Callable
import warnings

from transformers import pipeline

from .config import MODELS_CONFIG, PromptGuardConfig
from .categories import PromptGuardCategory
from ...errors import UnsafePromptError
from ...model import GuardModel
from .result import PromptGuardResult

class PromptGuardModel(GuardModel):
    def __init__(self, name: str):
        super().__init__(name=name, task="text-classification")
        self.pipeline: Optional[pipeline] = None

        self.config: PromptGuardConfig = MODELS_CONFIG[self.name]
        if not self.config.mappings:
            warnings.warn(f"No preset config found for model {self.name}. You may need to provide a custom config.")


    def _from_prediction(self, chunks: list[dict]) -> PromptGuardResult:
        if all(self.config.get_category_for_label(c["label"]) == PromptGuardCategory.SAFE for c in chunks):
            label = PromptGuardCategory.SAFE
            confidence = min(c["score"] for c in chunks)
        else:
            chunks = [c for c in chunks if self.config.get_category_for_label(c["label"]) == PromptGuardCategory.UNSAFE]
            label = PromptGuardCategory.UNSAFE
            confidence = max(c["score"] for c in chunks)
        return PromptGuardResult(
            chunks=chunks,
            label=label,
            confidence=confidence
        )

    def scan(self, text: str, max_seq_length: int=64, overlap: int=16) -> PromptGuardResult:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        return self._from_prediction(results)

    async def async_scan(self, text: str, max_seq_length: int = 64, overlap: int = 16) -> PromptGuardResult:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text, chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)
        return self._from_prediction(results)

    def redact(self, text: str, max_seq_length: int=64, overlap: int=16, replace: str="[REDACTED]", confidence: float=0.5) -> str:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]

            previous_unsafe = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if self.config.get_category_for_label(result["label"]) == PromptGuardCategory.UNSAFE and result["score"] >= confidence:
                    if previous_unsafe:
                        continue
                    redacted_chunks.append(replace)
                    previous_unsafe = True
                else:
                    redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                    previous_unsafe = False
        return " ".join(redacted_chunks)

    async def async_redact(self, text: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]",
                           confidence: float = 0.5) -> str:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)

        redacted_chunks = []
        previous_unsafe = False
        for chunk, result in zip(chunks, chunk_results):
            if self.config.get_category_for_label(result["label"]) == PromptGuardCategory.UNSAFE and result["score"] >= confidence:
                if previous_unsafe:
                    continue
                redacted_chunks.append(replace)
                previous_unsafe = True
            else:
                redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                previous_unsafe = False
        return " ".join(redacted_chunks)

    def redact_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]",
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
                        redacted = await self.async_redact(value, max_seq_length, overlap, replace, confidence)
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
                        redacted = self.redact(value, max_seq_length, overlap, replace, confidence)
                        bound.arguments[param] = redacted

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator

    def scan_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16, confidence: float=0.5) -> Callable:
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
                        if result.label == PromptGuardCategory.UNSAFE and result.confidence >= confidence:
                            raise UnsafePromptError(result.confidence)

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
                        if result.label == PromptGuardCategory.UNSAFE and result.confidence >= confidence:
                            raise UnsafePromptError(result.confidence)

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator


