import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Union

from .moderation_categories import ContentGuardCheck, ModerationCategory
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

    @classmethod
    def _from_prediction(cls, results: list[list[dict]]):
        ...

    def _chunk_has_violation(self, chunk_result: list[dict], confidence: float) -> bool:
        for result in chunk_result:
            if result["score"] >= confidence:
                return True
        return False

    def scan(self, text: str, max_seq_length: int=64, overlap: int=8):
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            chunk_results = [future.result() for future in futures]

            for chunk_result in chunk_results:
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
        return self._check_results_for_violations(results)


    def redact(self, text: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]",
               confidence: float = 0.5) -> str:
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

    def redact_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16, replace: str = "[REDACTED]",
                           confidence: float = 0.5) -> Callable:
        self.load_model()

        def decorator(func):
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

    def scan_before_exec(self, param: str, max_seq_length: int = 64, overlap: int = 16,
                         confidence: float = 0.5) -> Callable:
        self.load_model()

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                value = bound.arguments.get(param)

                if value is not None:
                    violations = self.scan(value, max_seq_length, overlap)
                    flagged = [cat for cat, is_violation in violations.items() if is_violation]
                    if flagged:
                        raise UnsafePromptError(confidence)

                return func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator

