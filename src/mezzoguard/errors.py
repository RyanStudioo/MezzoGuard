class UnsafeInputError(Exception):
    """Raised when the model deems input unsafe.

    ``categories`` contains one ``{category: score}`` entry per violated
    category. Prompt-style results are represented as ``{"unsafe": score}``.
    """

    def __init__(self, categories: list[dict[str, float]]):
        self.categories = categories
        super().__init__(f"Input was flagged as unsafe by the model with the following categories: {categories}")


class MezzoGuardServerError(Exception):
    """Raised when a remote Mezzo Guard server responds with an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Mezzo Guard server returned {status_code}: {detail}")
