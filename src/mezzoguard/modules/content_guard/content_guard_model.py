import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union, Any

from .categories import ContentGuardCheck, ModerationCategory
from .result import ContentGuardResult
from ...errors import UnsafePromptError
from ...model import Model, GuardModel


class ContentGuardModel(GuardModel):
    def __init__(self, name: str,
                 categories: list[ContentGuardCheck]=None
                 ):
        super().__init__(name, task="text-classification")
        if categories is None:
            categories = [
                ContentGuardCheck(ModerationCategory.DIVISIVE, 0.5),
                ContentGuardCheck(ModerationCategory.HATE_SPEECH, 0.5),
                ContentGuardCheck(ModerationCategory.SELF_HARM, 0.5),
                ContentGuardCheck(ModerationCategory.SEXUAL, 0.5),
                ContentGuardCheck(ModerationCategory.TOXIC, 0.5),
                ContentGuardCheck(ModerationCategory.VIOLENCE, 0.5),
            ]
        self.categories = categories

    def _get_threshold_from_category(self, category: Union[str, ModerationCategory]) -> float:
        for cat in self.categories:
            if isinstance(category, str):
                category_name = category.upper()
            elif isinstance(category, ModerationCategory):
                category_name = category.value.upper()
            else:
                raise
            if category_name == cat.category.value.upper():
                 return cat.threshold
        return 1.0

    def _check_if_violation(self, category: Union[str, ModerationCategory], confidence: float) -> bool:
        threshold = self._get_threshold_from_category(category)
        return confidence >= threshold

    def _check_results_for_violations(self, results: list[dict]) -> dict:
        violations = {}
        for result in results:
            category = result["label"]
            confidence = result["score"]
            if self._check_if_violation(category, confidence):
                violations[category] = True
            else:
                violations[category] = False
        return violations

    def _is_supported_category(self, category: ModerationCategory) -> bool:
        return any([i for i in self.categories if i.category == category])

    def _from_prediction(self, results: list[list[dict]]):
        label_to_category = {cat.value.lower(): cat for cat in ModerationCategory}
        max_scores: dict[ModerationCategory, float] = {}
        for chunk_result in results:
            for pred in chunk_result:
                label = pred.get("label", "").lower()
                score = pred.get("score", 0.0)
                category = label_to_category.get(label)
                if category is not None:
                    if category not in max_scores or score > max_scores[category]:
                        max_scores[category] = score

        violations = {}
        for category, max_score in max_scores.items():
            if not self._is_supported_category(category):
                continue
            violations[category] = max_score >= self._get_threshold_from_category(category)

        return ContentGuardResult(chunks=results, violations=violations)

    def _chunk_has_violation(self, chunk_result: list[dict]) -> bool:
        for result in chunk_result:
            label = result.get("label", "")
            score = result.get("score", 0.0)
            threshold = self._get_threshold_from_category(label)
            if score >= threshold:
                return True
        return False

    def scan(self, text: str, max_seq_length: int = 64, overlap: int = 8) -> ContentGuardResult:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = [future.result() for future in futures]
        return self._from_prediction(chunk_results)

    async def async_scan(self, text: str, max_seq_length: int = 64, overlap: int = 8) -> ContentGuardResult:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)
        return self._from_prediction(chunk_results)


    def redact(self, text: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]", **kwargs: Any) -> str:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]

            previous_flagged = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if self._chunk_has_violation(result):
                    if previous_flagged:
                        continue
                    redacted_chunks.append(replace)
                    previous_flagged = True
                else:
                    redacted_chunks.append(self._reform_tokenized_chunk(chunk))
                    previous_flagged = False
        return " ".join(redacted_chunks)

    async def async_redact(self, text: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]", **kwargs: Any) -> str:
        await asyncio.to_thread(self.load_model)
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = [loop.run_in_executor(executor, self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)

        redacted_chunks = []
        previous_flagged = False
        for chunk, result in zip(chunks, chunk_results):
            if self._chunk_has_violation(result):
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
                        if not result.is_safe():
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
                        if not result.is_safe():
                            raise UnsafePromptError(confidence)

                    return func(*bound.args, **bound.kwargs)
                return wrapper

        return decorator

