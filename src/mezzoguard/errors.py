class UnsafePromptError(Exception):
    """Raised when the model deems a prompt unsafe"""
    def __init__(self, confidence: float):
        super().__init__(f"Prompt was flagged as unsafe by the model with a confidence of {confidence:.2f}")