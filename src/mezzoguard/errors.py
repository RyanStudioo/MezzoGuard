class UnsafePromptError(Exception):
    """Raised when the model deems a prompt unsafe"""

    def __init__(self, confidence: float):
        super().__init__(f"Prompt was flagged as unsafe by the model with a confidence of {confidence:.2f}")


class UnsafeContentError(Exception):
    """Raised when the model deems content unsafe"""

    def __init__(self, categories: list[dict[str, float]]):
        super().__init__(f"Content was flagged as unsafe by the model with the following categories: {categories}")


class MezzoGuardServerError(Exception):
    """Raised when a remote Mezzo Guard server responds with an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Mezzo Guard server returned {status_code}: {detail}")
