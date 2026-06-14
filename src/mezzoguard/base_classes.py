import json
import os
from abc import ABC, abstractmethod
from typing import Literal, Self, Any, Optional

from ._types import BaseResult, Category

CONFIG_FILENAME = ".mezzoguard"


class ModelConfig:
    """Schema for model-specific configuration loaded from .mezzoguard files."""

    def __init__(
        self,
        model_type: Literal["prompt_guard", "content_guard"],
        mappings: dict[str, str],
        safe_category: Optional[str] = None,
        default_max_seq_length: int = 64,
        default_overlap: int = 16,
        default_replace: str = "[REDACTED]",
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
            default_max_seq_length=data.get("default_max_seq_length", 64),
            default_overlap=data.get("default_overlap", 16),
            default_replace=data.get("default_replace", "[REDACTED]"),
        )

    @classmethod
    def from_file(cls, path: str) -> Optional["ModelConfig"]:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def _parse_readme(cls, content: str) -> Optional[dict]:
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
    def get_deprecation_from_readme(cls, model_name: str) -> Optional[dict]:
        """Parse README.md from model repo for new_version deprecation field."""
        if cls._is_local_path(model_name):
            readme_path = os.path.join(model_name, "README.md")
            if os.path.isfile(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    return cls._parse_readme(f.read())
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
            except Exception:
                return None
        return None

    @classmethod
    def from_model_name(cls, model_name: str) -> Optional["ModelConfig"]:
        config = None

        # 1. Load .mezzoguard config
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
            except Exception:
                pass

        # 2. Check README.md for deprecation (new_version field)
        readme_info = cls.get_deprecation_from_readme(model_name)

        # 3. Apply deprecation info from README to config
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
        self._mapping = {}

    def add_threshold(self, category: Any, threshold: float) -> Self:
        """Add a threshold to a category"""
        self._mapping[category] = threshold
        return self

    def get_threshold(self, category: Any) -> float:
        """Get the threshold of a category"""
        if category not in self._mapping:
            return 0.0
        return self._mapping[category]

    @abstractmethod
    def evaluate(self, result: BaseResult, **kwargs) -> PolicyResult:
        """Evaluate a result from a guard scan"""
        raise NotImplementedError