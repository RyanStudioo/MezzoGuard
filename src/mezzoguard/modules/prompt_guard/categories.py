from enum import Enum


class Category(Enum):
    UNSAFE = "unsafe"
    SAFE = "safe"

__all__ = ["Category"]