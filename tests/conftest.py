import re

import pytest

from mezzoguard.base_classes import ModelConfig
from mezzoguard.guard import Guard
from mezzoguard.model import Model


class OffsetEncoding:
    def __init__(self, offsets):
        self.offset_mapping = offsets


class FakeTokenizer:
    """Tokenizes on whitespace with accurate character offsets."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        return OffsetEncoding([(m.start(), m.end()) for m in re.finditer(r"\S+", text)])

    def tokenize(self, text):
        return [m.group(0) for m in re.finditer(r"\S+", text)]


class FakePipeline:
    """Text-classification pipeline that flags text based on markers.

    mode "prompt": top-1 style output (list of dicts for a list input).
    mode "content": top_k=None style output (list of list of dicts).
    """

    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode

    def _prompt_prediction(self, text):
        unsafe = any(marker in text.lower() for marker in ("kill", "weapon", "steal"))
        if unsafe:
            return {"label": "unsafe", "score": 0.99}
        return {"label": "safe", "score": 0.97}

    def _content_predictions(self, text):
        low = text.lower()
        scores = {
            "divisive": 0.05,
            "hate-speech": 0.05,
            "self-harm": 0.05,
            "sexual": 0.05,
            "toxic": 0.05,
            "violence": 0.05,
        }
        if "kill" in low:
            scores["violence"] = 0.98
            scores["toxic"] = 0.8
        if "fuck" in low:
            scores["sexual"] = 0.97
            scores["hate-speech"] = 0.3
        if "stupid" in low:
            scores["hate-speech"] = 0.95
            scores["toxic"] = 0.9
        if "hurt" in low or "suicide" in low:
            scores["self-harm"] = 0.99
        if "fight" in low:
            scores["violence"] = 0.9
        return [{"label": label, "score": score} for label, score in scores.items()]

    def __call__(self, texts, top_k=None):
        single = isinstance(texts, str)
        inputs = [texts] if single else texts
        if self.mode == "prompt":
            outputs = [self._prompt_prediction(t) for t in inputs]
            return outputs[0] if single else outputs
        outputs = [self._content_predictions(t) for t in inputs]
        return outputs[0] if single else outputs


@pytest.fixture
def prompt_guard(monkeypatch):
    pipeline = FakePipeline(FakeTokenizer(), mode="prompt")
    make_guard(monkeypatch, pipeline)
    return Guard("RyanStudio/Mezzo-Prompt-Guard-v2-Small")


@pytest.fixture
def content_guard(monkeypatch):
    pipeline = FakePipeline(FakeTokenizer(), mode="content")
    make_guard(monkeypatch, pipeline)
    return Guard("RyanStudio/Mezzo-Content-Guard-Small")


def make_guard(monkeypatch, pipeline):
    """Patch config discovery and model loading so no network access is needed."""

    def _no_config(cls, model_name):
        return None

    def _fake_load_model(self):
        self.pipeline = pipeline
        self.dtype = "float32"

    monkeypatch.setattr(ModelConfig, "from_model_name", classmethod(_no_config))
    monkeypatch.setattr(ModelConfig, "get_deprecation_from_readme", classmethod(_no_config))
    monkeypatch.setattr(Model, "load_model", _fake_load_model)
