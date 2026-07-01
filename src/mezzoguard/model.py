from abc import abstractmethod, ABC
from typing import Literal, Callable, Any

import torch
from transformers import pipeline

from ._types import BaseResult, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_OVERLAP, DEFAULT_REDACTED_LABEL, PIPELINE_TASK


class Model(ABC):
    """Base Model Class"""
    def __init__(
            self,
            name: str,
            task: Literal["text-classification"],
            dtype: Literal[
                "auto",
                "fp32",
                "fp16",
                "bf16",
                "int8"
            ] = "auto",
            torch_compile: bool = False,
            compile_mode: str = "default",
            use_onnx: bool = False
    ):
        if torch_compile and use_onnx:
            raise ValueError("torch_compile and use_onnx cannot be used together")

        self.name = name
        self.task = task
        self.pipeline: pipeline | None = None
        self.dtype = dtype
        match self.dtype:
            case "auto":
                self.torch_dtype = torch.float16
            case "fp32":
                self.torch_dtype = torch.float32
            case "fp16":
                self.torch_dtype = torch.float16
            case "bf16":
                self.torch_dtype = torch.bfloat16 if not use_onnx else torch.float16
            case "int8":
                self.torch_dtype = torch.int8

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

    def _split_tokens_into_chunks(self, text: str, max_seq_length: int, overlap: int = 0) -> list:
        tokens = self.tokenize(text)
        chunks = []
        step = max_seq_length - overlap

        if step <= 0:
            raise ValueError(f"Invalid parameters: overlap ({overlap}) must be less than max_seq_length ({max_seq_length})")

        if not tokens:
            return chunks

        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + max_seq_length]
            chunks.append(chunk)
            if i + max_seq_length >= len(tokens):
                break
        return chunks

    def _is_batch_tokens(self, tokenized_text: list[str] | list[list[str]]) -> bool:
        if not tokenized_text:
            return False

        first_element = tokenized_text[0]
        return isinstance(first_element, list)

    def _predict_tokenized_text(self, tokenized_text: list[str] | list[list[str]]):
        token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(tokenized_text)
        decoded_chunk = self.pipeline.tokenizer.decode(token_ids, skip_special_tokens=True)
        result = self.pipeline(decoded_chunk)
        if not self._is_batch_tokens(tokenized_text):
            result = result[0]
        else:
            result = [res[0] for res in result]
        return result

    def _predict_tokenized_text_topk_none(self, tokenized_text: list[str]):
        token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(tokenized_text)
        decoded_chunk = self.pipeline.tokenizer.decode(token_ids, skip_special_tokens=True)
        result = self.pipeline(decoded_chunk, top_k=None)
        return result

    def _reform_tokenized_chunk(self, chunk: list[str]) -> str:
        token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(chunk)
        reformed_text = self.pipeline.tokenizer.decode(token_ids, skip_special_tokens=True)
        return reformed_text

    def tokenize(self, text: str) -> list[str]:
        self.load_model()
        return self.pipeline.tokenizer.tokenize(text)

    def get_token_length(self, text: str) -> int:
        self.load_model()
        tokens = self.tokenize(text)
        return len(tokens)

    def load_model(self) -> None:
        if self.pipeline: return
        try:
            if self.use_onnx:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer
                from huggingface_hub import HfApi

                tokenizer = AutoTokenizer.from_pretrained(self.name)
                api = HfApi()
                repo_files = api.list_repo_files(self.name)

                onnx_folder = f"onnx/"
                resolved_dtype = "fp16" if self.dtype == "auto" else self.dtype
                onnx_subfolder = f"model_{resolved_dtype}"
                model_folder = f"{onnx_folder}{onnx_subfolder}"
                has_onnx = any(f.startswith("onnx/") for f in repo_files)
                has_subfolder = any(f.startswith(f"{onnx_folder}{onnx_subfolder}/") for f in repo_files)

                if has_onnx and has_subfolder:
                    model = ORTModelForSequenceClassification.from_pretrained(self.name, subfolder=model_folder, file_name="model.onnx")
                else:
                    print("No ONNX model found in the repository. Exporting to ONNX...")
                    model = ORTModelForSequenceClassification.from_pretrained(self.name, export=True)

                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, torch_dtype=self.torch_dtype)
            elif self.torch_compile:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(self.name)
                model = AutoModelForSequenceClassification.from_pretrained(self.name, torch_dtype=self.torch_dtype)
                model.forward = torch.compile(model.forward, mode=self.compile_mode, dynamic=True)
                self.pipeline = pipeline(self.task, model=model, tokenizer=tokenizer, torch_dtype=self.torch_dtype)
            else:
                self.pipeline = pipeline(self.task, model=self.name, torch_dtype=self.torch_dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.name}': {e}") from e
        return

    def eject_model(self) -> None:
        if self.pipeline is not None:
            if hasattr(self.pipeline, 'model'):
                del self.pipeline.model
            del self.pipeline
            self.pipeline = None


class GuardModel(Model):
    """Base Guard Model class"""
    def __init__(
            self,
            name: str,
            task: Literal["text-classification"],
            **kwargs: Any
    ):
        super().__init__(name=name, task=task, **kwargs)

    @abstractmethod
    def scan(
            self,
            text: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult: ...

    @abstractmethod
    def redact(
            self,
            text: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP,
            replace: str = DEFAULT_REDACTED_LABEL,
            **kwargs: Any
    ) -> str: ...

    @abstractmethod
    def redact_before_exec(
            self,
            param: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP,
            replace: str = DEFAULT_REDACTED_LABEL,
            **kwargs: Any
    ) -> Callable: ...

    @abstractmethod
    def scan_before_exec(
            self,
            param: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP,
            **kwargs: Any
    ) -> Callable: ...

    @abstractmethod
    async def async_scan(
            self,
            text: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult: ...

    @abstractmethod
    async def async_redact(
            self,
            text: str,
            max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
            overlap: int = DEFAULT_OVERLAP,
            replace: str = DEFAULT_REDACTED_LABEL,
            **kwargs: Any
    ) -> str: ...
