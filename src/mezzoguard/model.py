from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional

from transformers import pipeline

from src.mezzoguard.resultmaker import ResultMaker, Result


class PromptGuardModel:
    def __init__(self, name: str):
        self.name = name
        self.pipeline: Optional[pipeline] = None

    def _split_text_with_tokenizer(self, text: str):
        self.load_model()
        return self.pipeline.tokenizer.tokenize(text)

    def _split_tokens_into_chunks(self, text: str, max_seq_length: int, overlap: int=0):
        tokens = self._split_text_with_tokenizer(text)
        chunks = []
        step = max_seq_length - overlap
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + max_seq_length]
            chunks.append(chunk)
            if i + max_seq_length >= len(tokens):
                break
        return chunks

    def _predict_tokenized_text(self, tokenized_text: list[str]):
        token_ids = self.pipeline.tokenizer.convert_tokens_to_ids(tokenized_text)
        decoded_chunk = self.pipeline.tokenizer.decode(token_ids, skip_special_tokens=True)
        result = self.pipeline(decoded_chunk)[0]
        return result

    def load_model(self) -> None:
        if not self.pipeline:
            self.pipeline = pipeline("text-classification", model=self.name)
        return

    def get_token_length(self, text: str) -> int:
        self.load_model()
        tokens = self.pipeline.tokenizer.tokenize(text)
        return len(tokens)

    def scan(self, text: str, max_seq_length: int=64, overlap: int=16) -> Result:
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        return ResultMaker.from_prediction(results)
