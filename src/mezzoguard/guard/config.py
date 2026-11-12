from __future__ import annotations


class GuardConfig:
    """Category configuration for the unified guard.

    Only scored categories are listed. A model label that is not configured
    (for example ``"safe"`` from a binary prompt classifier) is ignored.
    """

    def __init__(self, categories: list[str]):
        self.categories = [category for category in categories if self._normalize_label(category) != "safe"]
        self._normalized_categories = {self._normalize_label(category): category for category in self.categories}

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def get_category_for_label(self, label: str) -> str:
        if label in self.categories:
            return label
        normalized_label = self._normalize_label(label)
        if normalized_label in self._normalized_categories:
            return self._normalized_categories[normalized_label]
        raise ValueError(f"Label '{label}' not found in categories")

    def to_dict(self) -> dict:
        return {"categories": self.categories}


CONFIGS = {
    "unsafe": GuardConfig(categories=["unsafe"]),
    "mezzo-content-guard-preview": GuardConfig(
        categories=["divisive", "hate-speech", "self-harm", "sexual", "toxic", "violence"]
    ),
    "mezzo-content-guard-v1": GuardConfig(categories=["hate-speech", "self-harm", "sexual", "toxic", "violence"]),
}

MODELS_CONFIG = {
    "RyanStudio/Mezzo-Prompt-Guard-v2-Large": CONFIGS["unsafe"],
    "RyanStudio/Mezzo-Prompt-Guard-v2-Base": CONFIGS["unsafe"],
    "RyanStudio/Mezzo-Prompt-Guard-v2-Small": CONFIGS["unsafe"],
    "meta-llama/Llama-Prompt-Guard-2-22M": CONFIGS["unsafe"],
    "meta-llama/Llama-Prompt-Guard-2-86M": CONFIGS["unsafe"],
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


__all__ = ["CONFIGS", "MODELS_CONFIG", "GuardConfig"]
