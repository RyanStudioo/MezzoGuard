from concurrent.futures import ThreadPoolExecutor

from mezzoguard.model import Model


class ContentGuardModel(Model):
    def __init__(self, name: str):
        super().__init__(name, task="text-classification")

    @classmethod
    def _from_prediction(cls, results: list[dict]):
        return results


    def scan(self, text: str, max_seq_length: int=64, overlap: int=8):
        self.load_model()
        chunks = self._split_tokens_into_chunks(text, max_seq_length, overlap)
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._predict_tokenized_text_topk_none, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        return self._from_prediction(results)



