from collections import OrderedDict
from typing import Any

from mezzoguard import BaseCategory


class TextLRUCache:
    def __init__(self, maxsize: int=128):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, BaseCategory] = OrderedDict()

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: BaseCategory):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
