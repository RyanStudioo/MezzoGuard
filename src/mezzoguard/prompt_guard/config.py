from ..base_classes import BaseConfig


class PromptGuardConfig(BaseConfig):
    """Configurations for Prompt Guard"""

    def __init__(self, categories: list[str], safe_category: str = "safe"):
        super().__init__("prompt_guard")
        self.categories = categories
        self.safe_category = safe_category

    def get_category_for_label(self, label: str) -> str:
        if label not in self.categories:
            raise ValueError(f"Label '{label}' not found in categories")
        return label


CONFIGS = {
    "safe-unsafe": PromptGuardConfig(
        categories=["safe", "unsafe"],
        safe_category="safe",
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
