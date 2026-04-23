import inspect
from abc import abstractmethod, ABC
from typing import Optional, Union, Literal, Callable

from transformers import pipeline

from .results import Result


class Model(ABC):
    def __init__(
            self,
            name: str,
            task: Union[
                Literal["text-classification"]
            ]
    ):
        self.name = name
        self.task = task
        self.pipeline: Optional[pipeline] = None

    @classmethod
    @abstractmethod
    def _from_prediction(cls, results: list):...

    def _split_tokens_into_chunks(self, text: str, max_seq_length: int, overlap: int=0):
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

    def _is_batch_tokens(self, tokenized_text: Union[list[str], list[list[str]]]) -> bool:
        if not tokenized_text:
            return False

        first_element = tokenized_text[0]
        return isinstance(first_element, list)

    def _predict_tokenized_text(self, tokenized_text: Union[list[str], list[list[str]]]):
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
        if not self._is_batch_tokens(tokenized_text):
            result = result
        else:
            result = [res for res in result]
        return result

    def _reform_tokenized_chunk(self, chunk: list[str]) -> str:
        token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(chunk)
        reformed_text = self.pipeline.tokenizer.decode(token_ids, skip_special_tokens=True)
        return reformed_text

    def tokenize(self, text: str):
        self.load_model()
        return self.pipeline.tokenizer.tokenize(text)

    def get_token_length(self, text: str) -> int:
        self.load_model()
        tokens = self.tokenize(text)
        return len(tokens)

    def load_model(self) -> None:
        if not self.pipeline:
            self.pipeline = pipeline(self.task, model=self.name)
        return

class GuardModel(Model):

    @abstractmethod
    def scan(
            self,
            text: str,
            max_seq_length: int = 64,
            overlap: int = 16
    ) -> Result: ...

    @abstractmethod
    def redact(
            self,
            text: str,
            max_seq_length: int = 64,
            overlap: int = 16,
            replace: str = "[REDACTED]",
            confidence: float = 0.5
    ) -> str: ...

    @abstractmethod
    def redact_before_exec(
            self,
            param: str,
            max_seq_length: int = 64,
            overlap: int = 16,
            replace: str = "[REDACTED]",
            confidence: float = 0.5
    ) -> Callable: ...

    @abstractmethod
    def scan_before_exec(
            self,
            param: str,
            max_seq_length: int = 64,
            overlap: int = 16,
            confidence: float=0.5
    ) -> Callable: ...



