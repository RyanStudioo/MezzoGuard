from ...base_classes import Config
from ...modules.content_guard.categories import Category
from ...modules.prompt_guard.categories import Category


class PromptGuardConfig(Config):
	def __init__(self, mappings: dict[str, Category]):
		super().__init__()
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

MODEL_SERIES = {
	"Mezzo-Prompt-Guard-v2": PromptGuardConfig(
		mappings={
			"safe": Category.SAFE,
			"unsafe": Category.UNSAFE
		}
	)
}

MODELS_CONFIG = {
	"RyanStudio/Mezzo-Prompt-Guard-v2-Large": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
	"RyanStudio/Mezzo-Prompt-Guard-v2-Base": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
	"RyanStudio/Mezzo-Prompt-Guard-v2-Small": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
}

__all__ = ["PromptGuardConfig", "MODELS_CONFIG"]