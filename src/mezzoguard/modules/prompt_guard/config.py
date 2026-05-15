from ...config import Config
from ...modules.content_guard.categories import ModerationCategory
from ...modules.prompt_guard.categories import PromptGuardCategory


class PromptGuardConfig(Config):
	def __init__(self, mappings: dict[str, PromptGuardCategory]):
		super().__init__()
		self.mappings = mappings

	def get_category_for_label(self, label: str) -> PromptGuardCategory:
		if label not in self.mappings:
			raise ValueError(f"Label '{label}' not found in mappings")
		return self.mappings[label]

	def get_labels_for_category(self, category: PromptGuardCategory) -> list[str]:
		valid = []
		for label, cat in self.mappings.items():
				if cat == category:
					valid.append(label)
		return valid

MODEL_SERIES = {
	"Mezzo-Prompt-Guard-v2": PromptGuardConfig(
		mappings={
			"safe": PromptGuardCategory.SAFE,
			"unsafe": PromptGuardCategory.UNSAFE
		}
	)
}

MODELS_CONFIG = {
	"RyanStudio/Mezzo-Prompt-Guard-v2-Large": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
	"RyanStudio/Mezzo-Prompt-Guard-v2-Base": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
	"RyanStudio/Mezzo-Prompt-Guard-v2-Small": MODEL_SERIES["Mezzo-Prompt-Guard-v2"],
}