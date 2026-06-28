import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Self

from ._types import (
    BaseResult,
    Category,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
)

CONFIG_FILENAME = ".mezzoguard"


class ModelConfig:
    """Schema for model-specific configuration loaded from .mezzoguard files."""

    def __init__(
        self,
        model_type: Literal["prompt_guard", "content_guard"],
        mappings: dict[str, str],
        safe_category: str | None = None,
        default_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        default_overlap: int = DEFAULT_OVERLAP,
        default_replace: str = DEFAULT_REDACTED_LABEL,
        deprecated: bool = False,
        deprecated_message: str = "",
        replacement: str = "",
    ):
        self.model_type = model_type
        self.mappings = mappings
        self.safe_category = safe_category
        self.default_max_seq_length = default_max_seq_length
        self.default_overlap = default_overlap
        self.default_replace = default_replace
        self.deprecated = deprecated
        self.deprecated_message = deprecated_message
        self.replacement = replacement

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(
            model_type=data["model_type"],
            mappings=data["mappings"],
            safe_category=data.get("safe_category"),
            default_max_seq_length=data.get("default_max_seq_length", DEFAULT_MAX_SEQ_LENGTH),
            default_overlap=data.get("default_overlap", DEFAULT_OVERLAP),
            default_replace=data.get("default_replace", DEFAULT_REDACTED_LABEL),
        )

    @classmethod
    def from_file(cls, path: str) -> "ModelConfig | None":
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Failed to read config file {path}: {e}")
            return None
        return cls.from_dict(data)

    @classmethod
    def _parse_readme(cls, content: str) -> dict | None:
        """Parse YAML frontmatter from README.md for new_version field."""
        content = content.strip()
        if not content.startswith("---"):
            return None

        end = content.find("---", 3)
        if end == -1:
            return None

        frontmatter = content[3:end].strip()
        for line in frontmatter.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "new_version" and value:
                return {
                    "deprecated": True,
                    "replacement": value,
                    "deprecated_message": f"This model is deprecated. Use {value} instead.",
                }
        return None

    @classmethod
    def _is_local_path(cls, model_name: str) -> bool:
        return os.path.exists(model_name)

    @classmethod
    def get_deprecation_from_readme(cls, model_name: str) -> dict | None:
        """Parse README.md from model repo for new_version deprecation field."""
        if cls._is_local_path(model_name):
            readme_path = os.path.join(model_name, "README.md")
            if os.path.isfile(readme_path):
                try:
                    with open(readme_path, "r", encoding="utf-8") as f:
                        return cls._parse_readme(f.read())
                except OSError:
                    return None
        else:
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id=model_name,
                    filename="README.md",
                    repo_type="model",
                )
                with open(downloaded, "r", encoding="utf-8") as f:
                    return cls._parse_readme(f.read())
            except (OSError, ImportError):
                return None
        return None

    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelConfig | None":
        config = None

        if cls._is_local_path(model_name):
            local_config = os.path.join(model_name, CONFIG_FILENAME)
            config = cls.from_file(local_config)
        else:
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id=model_name,
                    filename=CONFIG_FILENAME,
                    repo_type="model",
                )
                config = cls.from_file(downloaded)
            except (OSError, ImportError):
                pass

        readme_info = cls.get_deprecation_from_readme(model_name)

        if readme_info and config:
            config.deprecated = readme_info["deprecated"]
            config.replacement = readme_info["replacement"]
            config.deprecated_message = readme_info["deprecated_message"]

        return config

    def to_dict(self) -> dict:
        d = {
            "model_type": self.model_type,
            "mappings": self.mappings,
            "default_max_seq_length": self.default_max_seq_length,
            "default_overlap": self.default_overlap,
            "default_replace": self.default_replace,
        }
        if self.safe_category is not None:
            d["safe_category"] = self.safe_category
        return d

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseConfig:
    """Base Config class"""
    def __init__(self, model_type: Literal["prompt_guard", "content_guard"]):
        self.model_type = model_type


class PolicyResult(BaseResult):
    def __init__(self, scores: dict[Category, float], violated: dict[Category, bool], categories: list[Category]):
        self.scores = scores
        self.violated = violated
        self.categories = categories

    def __bool__(self) -> bool:
        return self.is_unsafe()

    def __repr__(self) -> str:
        violated = [k.name for k, v in self.violated.items() if v]
        safe = [k.name for k, v in self.violated.items() if not v]
        return f"PolicyResult(safe={safe}, violated={violated})"

    def is_safe(self) -> bool:
        return not any(self.violated.values())

    def is_unsafe(self) -> bool:
        return any(self.violated.values())

    def get_violated_categories(self) -> list[Category]:
        return [category for category, violated in self.violated.items() if violated]


class BasePolicy(ABC):
    """Base Policy class"""
    def __init__(self):
        self._mapping: dict[Category, float] = {}

    def add_threshold(self, category: Category, threshold: float) -> Self:
        """Add a threshold to a category"""
        self._mapping[category] = threshold
        return self

    def get_threshold(self, category: Category) -> float:
        """Get the threshold of a category"""
        if category not in self._mapping:
            return 0.0
        return self._mapping[category]

    def evaluate(self, result: BaseResult, **kwargs) -> PolicyResult:
        """Evaluate a result from a guard scan"""
        violated: dict[Category, bool] = {}
        for key, value in result.scores.items():
            threshold = self.get_threshold(key)
            violated[key] = value >= threshold
        categories = list(result.scores.keys())
        return PolicyResult(scores=result.scores, violated=violated, categories=categories)


def _init_guard_config(name: str, CategoryEnum: type, ConfigClass: type, models_config: dict):
    """Shared config-loading logic for Guard __init__ methods.

    Returns (model_config, config) tuple.
    """
    from . import _types

    model_config = ModelConfig.from_model_name(name)
    config = models_config.get(name, None)

    if model_config and not config:
        mappings = {}
        for label, cat_str in model_config.mappings.items():
            try:
                cat = CategoryEnum(cat_str)
            except ValueError:
                cat = _types.Category(cat_str)
            mappings[label] = cat

        safe_cat = None
        if getattr(model_config, "safe_category", None):
            try:
                safe_cat = CategoryEnum(model_config.safe_category)
            except ValueError:
                safe_cat = _types.Category(model_config.safe_category)

        if safe_cat is not None:
            config = ConfigClass(mappings=mappings, safe_category=safe_cat)
        else:
            config = ConfigClass(mappings=mappings)

    if not config:
        warnings.warn(
            f"No preset config found for model {name}. You may need to provide a custom config."
        )

    return model_config, config


def _make_redact_before_exec(redact_method, async_redact_method):
    """Shared decorator factory for redact_before_exec."""
    import asyncio
    import functools
    import inspect

    def decorator(param, max_seq_length, overlap, replace, policy, confidence):
        def outer_decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments.get(param)
                    if value is not None:
                        redacted = await async_redact_method(
                            value, max_seq_length, overlap, replace,
                            policy=policy, confidence=confidence,
                        )
                        bound.arguments[param] = redacted
                    return await func(*bound.args, **bound.kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments.get(param)
                    if value is not None:
                        redacted = redact_method(
                            value, max_seq_length, overlap, replace,
                            policy=policy, confidence=confidence,
                        )
                        bound.arguments[param] = redacted
                    return func(*bound.args, **bound.kwargs)
                return wrapper
        return outer_decorator
    return decorator


def _make_scan_before_exec(scan_method, async_scan_method, on_unsafe):
    """Shared decorator factory for scan_before_exec.

    on_unsafe: callable(result, confidence) that raises if unsafe.
    """
    import asyncio
    import functools
    import inspect

    def decorator(param, max_seq_length, overlap, confidence):
        def outer_decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments.get(param)
                    if value is not None:
                        result = await async_scan_method(value, max_seq_length, overlap)
                        on_unsafe(result, confidence)
                    return await func(*bound.args, **bound.kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments.get(param)
                    if value is not None:
                        result = scan_method(value, max_seq_length, overlap)
                        on_unsafe(result, confidence)
                    return func(*bound.args, **bound.kwargs)
                return wrapper
        return outer_decorator
    return decorator
