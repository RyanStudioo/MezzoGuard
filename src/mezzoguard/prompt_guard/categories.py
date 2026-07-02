from .._types import BaseCategory


class Category(BaseCategory):
    UNSAFE = "unsafe"
    SAFE = "safe"

__all__ = ["Category"]