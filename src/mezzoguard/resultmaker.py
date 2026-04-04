from dataclasses import dataclass

@dataclass
class Result:
    _chunks: list[dict]
    label: str
    confidence: float

class ResultMaker:

    @classmethod
    def from_prediction(cls, chunks: list[dict]) -> Result:
        if len(chunks) == len([i for i in chunks if i["label"] == "safe"]):
            label = "safe"
            confidence = min([i["score"] for i in chunks])
            return Result(chunks, label, confidence)
        unsafe_chunks = [i for i in chunks if i["label"] == "unsafe"]
        label = "unsafe"
        confidence = max([i["score"] for i in unsafe_chunks])
        return Result(chunks, label, confidence)