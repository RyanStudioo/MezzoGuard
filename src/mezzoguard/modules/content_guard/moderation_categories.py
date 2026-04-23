from enum import Enum

class ContentGuardModerationCategories(Enum):
    SEXUAL = "Sexual"
    VIOLENCE = "Violence"
    HATE_SPEECH = "Hate Speech"
    TOXIC = "Toxic"
    DIVISIVE = "Divisive"
    SELF_HARM = "Self-Harm"

class ContentGuardCategory:
    def __init__(self, category: ContentGuardModerationCategories, confidence: float):
        self.category = category
        self.confidence = confidence