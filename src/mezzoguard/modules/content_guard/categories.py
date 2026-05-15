from enum import Enum

class ModerationCategory(Enum):
    DIVISIVE = "divisive"
    HATE_SPEECH = "hate-speech"
    SELF_HARM = "self-harm"
    SEXUAL = "sexual"
    TOXIC = "toxic"
    VIOLENCE = "violence"

class ContentGuardCheck:
    def __init__(self, category: ModerationCategory, threshold: float):
        self.category = category
        self.threshold = threshold