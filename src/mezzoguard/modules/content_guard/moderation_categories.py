from enum import Enum

class ModerationCategory(Enum):
    DIVISIVE = "Divisive"
    HATE_SPEECH = "Hate Speech"
    SELF_HARM = "Self-Harm"
    SEXUAL = "Sexual"
    TOXIC = "Toxic"
    VIOLENCE = "Violence"

class ContentGuardCheck:
    def __init__(self, category: ModerationCategory, threshold: float):
        self.category = category
        self.threshold = threshold