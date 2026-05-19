from mezzoguard.base_classes import Config


class ContentGuardConfig(Config):
	def __init__(self):
		super().__init__("prompt_guard")

__all__ = ["ContentGuardConfig"]