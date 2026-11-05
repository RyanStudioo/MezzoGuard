from ..base_classes import BaseConfig


class ContentGuardConfig(BaseConfig):
    """Configurations for Content Guard"""

    def __init__(self, categories: list[str]):
        super().__init__("content_guard")
        self.categories = categories
        self._normalized_categories = {self._normalize_label(category): category for category in categories}

    def _normalize_label(self, label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def get_category_for_label(self, label: str) -> str:
        if label in self.categories:
            return label
        normalized_label = self._normalize_label(label)
        if normalized_label in self._normalized_categories:
            return self._normalized_categories[normalized_label]
        raise ValueError(f"Label '{label}' not found in categories")


CONFIGS = {
    "mezzo-content-guard-preview": ContentGuardConfig(
        categories=["divisive", "hate-speech", "self-harm", "sexual", "toxic", "violence"]
    ),
    "mezzo-content-guard-v1": ContentGuardConfig(
        categories=["hate-speech", "self-harm", "sexual", "toxic", "violence"]
    ),
}

MODELS_CONFIG = {
    "RyanStudio/Mezzo-Content-Guard-Large-Preview": CONFIGS["mezzo-content-guard-preview"],
    "RyanStudio/Mezzo-Content-Guard-Large": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-Base": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-Small": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Large": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Base": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Small": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Tiny": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Nano": CONFIGS["mezzo-content-guard-v1"],
    "RyanStudio/Mezzo-Content-Guard-v1.5-Nano-Preview": CONFIGS["mezzo-content-guard-v1"],
}

__all__ = ["ContentGuardConfig", "MODELS_CONFIG"]
