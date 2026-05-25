from ..base_classes import BaseConfig
from .categories import Category


class ContentGuardConfig(BaseConfig):
    def __init__(self, mappings: dict[str, Category]):
        super().__init__("content_guard")
        self.mappings = mappings
        self._normalized_mappings = {
            self._normalize_label(label): category for label, category in mappings.items()
        }

    def _normalize_label(self, label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def get_category_for_label(self, label: str) -> Category:
        if label in self.mappings:
            return self.mappings[label]
        normalized_label = self._normalize_label(label)
        if normalized_label in self._normalized_mappings:
            return self._normalized_mappings[normalized_label]
        raise ValueError(f"Label '{label}' not found in mappings")

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