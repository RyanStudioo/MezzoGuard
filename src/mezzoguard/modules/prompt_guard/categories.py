from enum import Enum


class PromptGuardCategory(Enum):
    UNSAFE = "unsafe"
    SAFE = "safe"