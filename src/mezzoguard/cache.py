from collections import OrderedDict
from typing import Any

from .config import cache_maxsize


class TextLRUCache:
    def __init__(self, maxsize: int = cache_maxsize):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self._hits += 1
            return self.cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    def info(self) -> dict[str, Any]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self.cache), "maxsize": self.maxsize}
