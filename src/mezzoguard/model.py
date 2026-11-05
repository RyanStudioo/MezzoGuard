from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from . import config as _config
from ._types import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
    BaseResult,
    Span,
)
from .base_classes import BasePolicy
from .cache import TextLRUCache

if TYPE_CHECKING:
    import torch


class Model(ABC):
    """Base Model Class"""

    def __init__(
        self,
        name: str,
        task: Literal["text-classification"],
        dtype: torch.dtype | str = "auto",
        torch_compile: bool = False,
        compile_mode: str = "default",
        use_onnx: bool = False,
        device: str | None = None,
    ):
        if torch_compile and use_onnx:
            raise ValueError("torch_compile and use_onnx cannot be used together")

        self.name = name
        self.task = task
        self.pipeline = None
        self.dtype = dtype
        self.torch_compile = torch_compile
        self.compile_mode = compile_mode
        self.use_onnx = use_onnx
        self.device = device
        self._cache = TextLRUCache()

        self.load_model()

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.eject_model()
        return False

    @classmethod
    @abstractmethod
    def _from_prediction(cls, results: list): ...

    def _chunk_text(self, text: str, max_seq_length: int, overlap: int = 0) -> list[tuple[int, int]]:
        """Split text into character spans, one per model chunk.

        Each span is a ``(start_char, end_char)`` pair into the original text, so
        chunks can be reconstructed losslessly. Consecutive spans overlap by
        ``overlap`` tokens.
        """
        if overlap >= max_seq_length:
            raise ValueError(
                f"Invalid parameters: overlap ({overlap}) must be less than max_seq_length ({max_seq_length})"
            )
        if not text:
            return []

        tokenized = self.pipeline.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenized.offset_mapping
        total = len(offsets)
        if total == 0:
            return []

        step = max_seq_length - overlap
        spans = []
        i = 0
        while i < total:
            end = min(i + max_seq_length, total)
            spans.append((offsets[i][0], offsets[end - 1][1]))
            if end == total:
                break
            i += step
        return spans

    def tokenize(self, text: str) -> list[str]:
        self.load_model()
        return self.pipeline.tokenizer.tokenize(text)

    def get_token_length(self, text: str) -> int:
        self.load_model()
        tokens = self.tokenize(text)
        return len(tokens)

    def load_model(self) -> None:
        if self.pipeline:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "Mezzo Guard requires PyTorch to load models, but it is not installed. "
                "Install it with `pip install torch` (use the build matching your platform)."
            ) from e

        if self.dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        device = self._resolve_device(torch)

        try:
            if self.use_onnx:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer

                export_dir = self._resolve_onnx_dir()
                provider, provider_options = self._resolve_onnx_provider(device)
                ort_kwargs: dict[str, Any] = {}
                if provider is not None:
                    ort_kwargs["provider"] = provider
                if provider_options:
                    ort_kwargs["provider_options"] = provider_options
                if os.path.isdir(export_dir):
                    tokenizer = AutoTokenizer.from_pretrained(export_dir)
                    model = ORTModelForSequenceClassification.from_pretrained(export_dir, export=False, **ort_kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(self.name)
                    model = ORTModelForSequenceClassification.from_pretrained(self.name, export=True, **ort_kwargs)
                    tokenizer.save_pretrained(export_dir)
                    model.save_pretrained(export_dir)
                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, dtype=self.dtype)
            elif self.torch_compile:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.name)
                model = AutoModelForSequenceClassification.from_pretrained(self.name, torch_dtype=self.dtype)
                model.forward = torch.compile(model.forward, mode=self.compile_mode, dynamic=True)
                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, dtype=self.dtype, device=device)
            else:
                self.pipeline = pipeline(self.task, model=self.name, dtype=self.dtype, device=device)
            self._strip_token_type_ids()
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.name}': {e}") from e

    def _resolve_device(self, torch) -> str | None:
        """Resolve the requested device, expanding ``"auto"`` to CUDA when available."""
        if self.device is None:
            return None
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _resolve_onnx_provider(self, device: str | None) -> tuple[str | None, dict]:
        """Map a device string to an ONNX Runtime execution provider."""
        if device is None:
            return None, {}
        if device == "cpu":
            return "CPUExecutionProvider", {}
        if device == "cuda" or device.startswith("cuda:"):
            provider_options = {}
            if ":" in device:
                provider_options["device_id"] = int(device.split(":", 1)[1])
            return "CUDAExecutionProvider", provider_options
        raise ValueError(f"ONNX backend does not support device '{device}'. Use 'cpu' or 'cuda'.")

    def _strip_token_type_ids(self) -> None:
        """Remove ``token_type_ids`` from forward kwargs for models that do not accept it."""
        import inspect

        model = self.pipeline.model
        if "token_type_ids" not in inspect.signature(model.forward).parameters:
            original_forward = model.forward

            def _forward(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("token_type_ids", None)
                return original_forward(*args, **kwargs)

            model.forward = _forward

    def _resolve_onnx_dir(self) -> str:
        """Directory where exported ONNX artifacts are cached for this model."""
        safe_name = self.name.replace("/", os.sep)
        return os.path.join(os.path.expanduser("~"), ".cache", "mezzoguard", "onnx", safe_name)

    def _predict(self, inputs: list[str], **kwargs: Any) -> list:
        """Run the pipeline on ``inputs`` with an LRU cache keyed by input text.

        When caching is enabled, only inputs not already in the cache are
        passed to the pipeline; results for the rest are replayed from the
        cache, preserving the original batch order (including duplicates).
        """
        if not _config.cache_enabled:
            return self.pipeline(inputs, **kwargs)

        unique_keys = list(dict.fromkeys(inputs))
        cached = {key: self._cache.get(key) for key in unique_keys}
        misses = [key for key, value in cached.items() if value is None]
        fresh = dict(zip(misses, self.pipeline(misses, **kwargs), strict=True)) if misses else {}
        by_text = {**fresh, **{key: value for key, value in cached.items() if value is not None}}
        for key, value in fresh.items():
            self._cache.put(key, value)
        return [by_text[text] for text in inputs]

    def cache_clear(self) -> None:
        self._cache.clear()

    def cache_info(self) -> dict[str, Any]:
        return self._cache.info()

    def eject_model(self) -> None:
        if self.pipeline is not None:
            if hasattr(self.pipeline, "model"):
                del self.pipeline.model
            del self.pipeline
            self.pipeline = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GuardModel(Model):
    """Base Guard Model class.

    Batch methods are the primitives; single-text methods delegate to them with a
    one-element batch so behavior stays identical between the two.
    """

    def __init__(self, name: str, task: Literal["text-classification"], **kwargs: Any):
        super().__init__(name=name, task=task, **kwargs)

    @abstractmethod
    def batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]: ...

    @abstractmethod
    async def async_batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]: ...

    @abstractmethod
    def batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]: ...

    @abstractmethod
    async def async_batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]: ...

    def scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult:
        return self.batch_scan([text], max_seq_length, overlap)[0]

    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult:
        return (await self.async_batch_scan([text], max_seq_length, overlap))[0]

    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> str | tuple[str, list[Span]]:
        return self.batch_redact(
            [text],
            max_seq_length,
            overlap,
            replace,
            policy=policy,
            confidence=confidence,
            return_spans=return_spans,
            **kwargs,
        )[0]

    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> str | tuple[str, list[Span]]:
        return (
            await self.async_batch_redact(
                [text],
                max_seq_length,
                overlap,
                replace,
                policy=policy,
                confidence=confidence,
                return_spans=return_spans,
                **kwargs,
            )
        )[0]

    @abstractmethod
    def redact_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        **kwargs: Any,
    ) -> Callable: ...

    @abstractmethod
    def scan_before_exec(
        self, param: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP, **kwargs: Any
    ) -> Callable: ...
