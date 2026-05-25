from ..base_classes import Config
from .categories import Category


class PromptGuardConfig(Config):
    def __init__(
        self, mappings: dict[str, Category], safe_category: Category = Category.SAFE
    ):
        super().__init__("prompt_guard")
        self.mappings = mappings
        self.safe_category = safe_category

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
    "safe-unsafe": PromptGuardConfig(
        mappings={
            "safe": Category.SAFE,
            "unsafe": Category.UNSAFE,
        },
        safe_category=Category.SAFE,
    ),
}

MODELS_CONFIG = {
    "RyanStudio/Mezzo-Prompt-Guard-v2-Large": CONFIGS["safe-unsafe"],
    "RyanStudio/Mezzo-Prompt-Guard-v2-Base": CONFIGS["safe-unsafe"],
    "RyanStudio/Mezzo-Prompt-Guard-v2-Small": CONFIGS["safe-unsafe"],
    "meta-llama/Llama-Prompt-Guard-2-22M": CONFIGS["safe-unsafe"],
    "meta-llama/Llama-Prompt-Guard-2-86M": CONFIGS["safe-unsafe"],
}

__all__ = ["PromptGuardConfig", "MODELS_CONFIG"]