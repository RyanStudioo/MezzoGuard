import functools
import inspect
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Callable

from transformers import pipeline

from ...errors import UnsafePromptError
from ...model import Model
from ...modules.prompt_guard.result import PromptGuardResult

class PromptGuardModel(Model):
    def __init__(self, name: str):
        super().__init__(name=name, task="text-classification")
        self.pipeline: Optional[pipeline] = None


    @classmethod
    def _from_prediction(cls, chunks: list[dict]) -> PromptGuardResult:
        if len(chunks) == len([i for i in chunks if i["label"] == "safe"]):
            label = "safe"
            confidence = min([i["score"] for i in chunks])
        else:
            chunks = [i for i in chunks if i["label"] == "unsafe"]
            label = "unsafe"
            confidence = max([i["score"] for i in chunks])
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

    def redact(self, text: str, max_seq_length: int=64, overlap: int=16, replace: str="[REDACTED]", confidence: float=0.5) -> str:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        redacted_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]

            previous_unsafe = False
            for chunk, future in zip(chunks, futures):
                result = future.result()
                if result["label"] == "unsafe" and result["score"] >= confidence:
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
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                value = bound.arguments.get(param)

                if value is not None:
                    result = self.scan(value, max_seq_length, overlap)
                    if result.label == "unsafe" and result.confidence >= confidence:
                        raise UnsafePromptError(result.confidence)

                return func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator


