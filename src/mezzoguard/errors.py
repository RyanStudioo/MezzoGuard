class UnsafePromptError(Exception):
    """Raised when a prompt is deemed unsafe by the model"""
    def __init__(self, confidence: float):
        super().__init__(f"Prompt was flagged as unsafe by the model with a confidence of {confidence:.2f}")