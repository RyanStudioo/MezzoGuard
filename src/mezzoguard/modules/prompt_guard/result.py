from mezzoguard.results import Result

class PromptGuardResult(Result):
    accepted_labels = ["safe", "unsafe"]

    def __init__(self, chunks: list[dict], label: str, confidence: float):
        self._chunks = chunks
        self.label = label
        self.confidence = confidence

        if self.label not in self.accepted_labels:
            raise ValueError(f"Invalid label: {self.label}. Accepted labels are: {self.accepted_labels}")

    def is_safe(self, threshold: float=0.5) -> bool:
        if self.label == "safe":
            return True
        elif self.label == "unsafe":
            if self.confidence < threshold:
                return True
            else:
                return False
        else:
            raise ValueError(f"Invalid label: {self.label}")