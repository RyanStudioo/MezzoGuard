from typing import Any, Callable

from . import CONTENTGUARD
from .preset_models import PROMPTGUARD
from .prompt_guard import guard as pg


class AutoGuard:
    def __init__(
            self,
            prompt_guard: pg.Guard=pg.Guard(PROMPTGUARD.MEZZO_PROMPT_GUARD_V2_BASE),
            content_guard: pg.Guard=pg.Guard(CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE)
    ):
        self.prompt_guard = prompt_guard
        self.content_guard = content_guard
