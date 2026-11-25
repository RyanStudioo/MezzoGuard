from __future__ import annotations

import functools
import json
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._types import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
    Span,
)
from .errors import UnsafeInputError

if TYPE_CHECKING:
    from .guard import Policy, Result

CONFIG_FILENAME = ".mezzoguard"


class ModelConfig:
    """Schema for model-specific configuration loaded from .mezzoguard files."""

    def __init__(
        self,
        categories: list[str],
        default_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        default_overlap: int = DEFAULT_OVERLAP,
        default_replace: str = DEFAULT_REDACTED_LABEL,
        deprecated: bool = False,
        deprecated_message: str = "",
        replacement: str = "",
    ):
        self.categories = categories
        self.default_max_seq_length = default_max_seq_length
        self.default_overlap = default_overlap
        self.default_replace = default_replace
        self.deprecated = deprecated
        self.deprecated_message = deprecated_message
        self.replacement = replacement

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        categories = data.get("categories")
        if not isinstance(categories, list):
            raise ValueError("model config must contain a 'categories' list")
        return cls(
            categories=categories,
            default_max_seq_length=data.get("default_max_seq_length", DEFAULT_MAX_SEQ_LENGTH),
            default_overlap=data.get("default_overlap", DEFAULT_OVERLAP),
            default_replace=data.get("default_replace", DEFAULT_REDACTED_LABEL),
        )

    @classmethod
    def from_file(cls, path: str) -> ModelConfig | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Failed to read config file {path}: {e}", stacklevel=2)
            return None
        try:
            return cls.from_dict(data)
        except ValueError as e:
            warnings.warn(f"Invalid config file {path}: {e}", stacklevel=2)
            return None

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
    @functools.lru_cache(maxsize=128)
    def get_deprecation_from_readme(cls, model_name: str) -> dict | None:
        """Parse README.md from model repo for new_version deprecation field."""
        if cls._is_local_path(model_name):
            readme_path = os.path.join(model_name, "README.md")
            if os.path.isfile(readme_path):
                try:
                    with open(readme_path, encoding="utf-8") as f:
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
                with open(downloaded, encoding="utf-8") as f:
                    return cls._parse_readme(f.read())
            except (OSError, ImportError):
                return None
        return None

    @classmethod
    @functools.lru_cache(maxsize=128)
    def from_model_name(cls, model_name: str) -> ModelConfig | None:
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
        return {
            "categories": self.categories,
            "default_max_seq_length": self.default_max_seq_length,
            "default_overlap": self.default_overlap,
            "default_replace": self.default_replace,
        }

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class GuardBase(ABC):
    """Common interface implemented by local and remote guards.

    Batch methods are the primitive operations. The single-text methods and
    ``*_before_exec`` decorators are concrete adapters built on top of them, so
    local and remote guards share exactly one implementation.
    """

    @abstractmethod
    def batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]: ...

    @abstractmethod
    async def async_batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[Result]: ...

    @abstractmethod
    def batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: Policy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]: ...

    @abstractmethod
    async def async_batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: Policy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]: ...

    def scan(self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP) -> Result:
        return self.batch_scan([text], max_seq_length, overlap)[0]

    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> Result:
        return (await self.async_batch_scan([text], max_seq_length, overlap))[0]

    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: Policy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> str | tuple[str, list[Span]]:
        return self.batch_redact(
            [text],
            max_seq_length,
            overlap,
            replace,
            policy=policy,
            confidence=confidence,
            return_spans=return_spans,
            **kwargs,
        )[0]

    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[str, str] = DEFAULT_REDACTED_LABEL,
        policy: Policy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> str | tuple[str, list[Span]]:
        return (
            await self.async_batch_redact(
                [text],
                max_seq_length,
                overlap,
                replace,
                policy=policy,
                confidence=confidence,
                return_spans=return_spans,
                **kwargs,
            )
        )[0]

    def _on_unsafe(self, result: Result, confidence: float) -> None:
        violated = [{category: score} for category, score in result.scores.items() if score >= confidence]
        if violated:
            raise UnsafeInputError(violated)

    def redact_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: Policy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Callable:
        return _make_redact_before_exec(self.redact, self.async_redact)(
            param, max_seq_length, overlap, replace, policy, confidence
        )

    def scan_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Callable:
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe)(
            param, max_seq_length, overlap, confidence
        )


def _redact_text_with_spans(
    text: str,
    spans: list[tuple[int, int]],
    flags: list[tuple[bool, str | None]],
    replace: str | dict[str, str],
) -> tuple[str, list[Span]]:
    """Shared redaction core used by the guard.

    Rebuilds ``text`` chunk by chunk, replacing flagged runs of consecutive
    chunks with a single replacement. ``spans`` are ``(start_char, end_char)``
    pairs into the original text (see ``Model._chunk_text``); consecutive chunks
    may overlap, and overlapping regions are only emitted once, so the result
    preserves the original text exactly except for flagged regions.
    """
    if isinstance(replace, str):

        def replacement_for(category: str | None) -> str:
            return replace
    else:

        def replacement_for(category: str | None) -> str:
            return replace.get(category, DEFAULT_REDACTED_LABEL)

    parts = []
    redaction_spans = []
    cursor = 0
    i = 0
    n = len(spans)
    while i < n:
        start, end = spans[i]
        if start == end or end <= cursor:
            i += 1
            continue
        flagged, category = flags[i]
        if not flagged:
            parts.append(text[cursor:end])
            cursor = end
            i += 1
        else:
            merge_end = end
            j = i
            while j + 1 < n and flags[j + 1][0] and spans[j + 1][1] > merge_end:
                j += 1
                merge_end = spans[j][1]
            replacement = replacement_for(category)
            parts.append(replacement)
            redaction_spans.append(Span(start=cursor, end=merge_end, category=category, replacement=replacement))
            cursor = merge_end
            i = j + 1
    return "".join(parts), redaction_spans


def _warn_if_deprecated(model_config: ModelConfig | None, name: str) -> None:
    """Emit the standard deprecation warning for a model, if any."""
    if model_config is not None and model_config.deprecated:
        warnings.warn(
            model_config.deprecated_message,
            DeprecationWarning,
            stacklevel=3,
        )
        return

    readme_deprecation = ModelConfig.get_deprecation_from_readme(name)
    if readme_deprecation:
        warnings.warn(
            readme_deprecation["deprecated_message"],
            DeprecationWarning,
            stacklevel=3,
        )


def _make_redact_before_exec(redact_method, async_redact_method):
    """Shared decorator factory for redact_before_exec."""
    import functools
    import inspect

    def decorator(param, max_seq_length, overlap, replace, policy, confidence):
        def outer_decorator(func):
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments.get(param)
                    if value is not None:
                        redacted = await async_redact_method(
                            value,
                            max_seq_length,
                            overlap,
                            replace,
                            policy=policy,
                            confidence=confidence,
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
                            value,
                            max_seq_length,
                            overlap,
                            replace,
                            policy=policy,
                            confidence=confidence,
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
    import functools
    import inspect

    def decorator(param, max_seq_length, overlap, confidence):
        def outer_decorator(func):
            if inspect.iscoroutinefunction(func):

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
