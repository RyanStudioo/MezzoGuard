from enum import Enum


class PromptGuardCategories(Enum):
    UNSAFE = "unsafe"
    SAFE = "safe"