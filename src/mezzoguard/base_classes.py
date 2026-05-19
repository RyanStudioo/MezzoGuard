from dataclasses import dataclass
from typing import Literal


class Config:
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):\
        self.model_type = model_type


@dataclass
class BaseResult:
    pass
