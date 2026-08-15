from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from ._types import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, BaseResult

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

        try:
            if self.use_onnx:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer

                export_dir = self._resolve_onnx_dir()
                if os.path.isdir(export_dir):
                    tokenizer = AutoTokenizer.from_pretrained(export_dir)
                    model = ORTModelForSequenceClassification.from_pretrained(export_dir, export=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(self.name)
                    model = ORTModelForSequenceClassification.from_pretrained(self.name, export=True)
                    tokenizer.save_pretrained(export_dir)
                    model.save_pretrained(export_dir)
                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, dtype=self.dtype)
            elif self.torch_compile:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.name)
                model = AutoModelForSequenceClassification.from_pretrained(self.name, torch_dtype=self.dtype)
                model.forward = torch.compile(model.forward, mode=self.compile_mode, dynamic=True)
                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, dtype=self.dtype)
            else:
                self.pipeline = pipeline(self.task, model=self.name, dtype=self.dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.name}': {e}") from e

    def _resolve_onnx_dir(self) -> str:
        """Directory where exported ONNX artifacts are cached for this model."""
        safe_name = self.name.replace("/", os.sep)
        return os.path.join(os.path.expanduser("~"), ".cache", "mezzoguard", "onnx", safe_name)

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
    """Base Guard Model class"""

    def __init__(self, name: str, task: Literal["text-classification"], **kwargs: Any):
        super().__init__(name=name, task=task, **kwargs)

    @abstractmethod
    def scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult: ...

    @abstractmethod
    def scan_many(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]: ...

    @abstractmethod
    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        **kwargs: Any,
    ) -> str: ...

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

    @abstractmethod
    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult: ...

    @abstractmethod
    async def async_scan_many(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]: ...

    @abstractmethod
    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        **kwargs: Any,
    ) -> str: ...
