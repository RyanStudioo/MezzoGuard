from mezzoguard.base_classes import Config
from mezzoguard.content_guard import Category


class ContentGuardConfig(Config):
    def __init__(self, mappings: dict[str, Category]):
        super().__init__("content_guard")
        self.mappings = mappings

    def get_category_for_label(self, label: str) -> Category:
        if label not in self.mappings:
            raise ValueError(f"Label '{label}' not found in mappings")
        return self.mappings[label]

    def get_labels_for_category(self, category: Category) -> list[str]:
        valid = []
        for label, cat in self.mappings.items():
                if cat == category:
                    valid.append(label)
        return valid

CONFIGS = {
    "mezzo-content-guard-preview": ContentGuardConfig(
        mappings={
            "Divisive": Category.DIVISIVE,
            "Hate Speech": Category.HATE_SPEECH,
            "Self-Harm": Category.SELF_HARM,
            "Sexual": Category.SEXUAL,
            "Toxic": Category.TOXIC,
            "Violence": Category.VIOLENCE
        }
    ),
    "mezzo-content-guard-v1": ContentGuardConfig(
        mappings={
            "hate-speech": Category.HATE_SPEECH,
            "self-harm": Category.SELF_HARM,
            "sexual": Category.SEXUAL,
            "toxic": Category.TOXIC,
            "violence": Category.VIOLENCE
        }
    )
}

MODELS_CONFIG = {
    "RyanStudio/Mezzo-Content-Guard-Large-Preview": CONFIGS["mezzo-content-guard-preview"],
    "RyanStudio/Mezzo-Content-Guard-Large": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-Base": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-Small": CONFIGS["mezzo-content-guard-v1"],
}

__all__ = ["ContentGuardConfig", "MODELS_CONFIG"]