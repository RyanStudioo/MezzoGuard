from __future__ import annotations

from typing import Any

from ._types import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OVERLAP,
    DEFAULT_REDACTED_LABEL,
    BaseCategory,
    BaseResult,
    Span,
)
from .base_classes import (
    BasePolicy,
    _make_redact_before_exec,
    _make_scan_before_exec,
)
from .content_guard import Category as ContentCategory
from .content_guard import Result as ContentResult
from .errors import MezzoGuardServerError, UnsafeContentError, UnsafePromptError
from .prompt_guard import Category as PromptCategory
from .prompt_guard import Result as PromptResult


def _require_httpx():
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(
            'API guards require httpx, which is not installed. Install it with `pip install "mezzo-guard[client]"`.'
        ) from e
    return httpx


def _category_from_name(enum_cls: type[BaseCategory], name: str) -> Any:
    try:
        return enum_cls(name)
    except ValueError:
        return name


def _handle_response(response: Any) -> Any:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        raise MezzoGuardServerError(response.status_code, detail)
    return response.json()


class APIGuard:
    """Base class for guards that call a Mezzo Guard HTTP server.

    Mirrors the local ``Guard`` API (``scan``, ``batch_scan``, ``redact``,
    ``batch_redact`` and their async variants) so it can be swapped in without
    loading a model locally. Use ``APIPromptGuard`` or ``APIContentGuard``.
    """

    model_type: str
    _category_enum: type[BaseCategory]
    _result_cls: type[BaseResult]

    def __init__(self, base_url: str, timeout: float = 30.0):
        httpx = _require_httpx()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client: Any = None
        self._info: dict | None = None

    def __enter__(self) -> APIGuard:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> APIGuard:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.aclose()
        return False

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        self._client.close()

    @property
    def info(self) -> dict:
        if self._info is None:
            self._info = self._request("GET", "/v1/info")
        return self._info

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        return _handle_response(response)

    async def _async_request(self, method: str, path: str, **kwargs: Any) -> Any:
        if self._async_client is None:
            httpx = _require_httpx()
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        response = await self._async_client.request(method, f"{self.base_url}{path}", **kwargs)
        return _handle_response(response)

    def _scan_payload(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> dict:
        return {
            "texts": texts,
            "max_seq_length": max_seq_length,
            "overlap": overlap,
            "include_chunks": True,
        }

    def _redact_payload(
        self,
        texts: list[str],
        max_seq_length: int,
        overlap: int,
        replace: str | dict[BaseCategory, str],
        policy: BasePolicy | None,
        confidence: float,
        return_spans: bool,
    ) -> dict:
        payload: dict[str, Any] = {
            "texts": texts,
            "max_seq_length": max_seq_length,
            "overlap": overlap,
            "confidence": confidence,
            "return_spans": return_spans,
        }
        if isinstance(replace, dict):
            payload["replace"] = DEFAULT_REDACTED_LABEL
            payload["replacements"] = {getattr(cat, "value", str(cat)): value for cat, value in replace.items()}
        else:
            payload["replace"] = replace
        if policy is not None:
            payload["policy"] = policy.to_dict()
        return payload

    def _build_result(self, payload: dict) -> BaseResult:
        scores = {_category_from_name(self._category_enum, name): score for name, score in payload["scores"].items()}
        chunks = payload.get("chunks")
        return self._result_cls(chunks=chunks if chunks is not None else [], scores=scores)

    def _build_spans(self, payload: list[dict]) -> list[Span]:
        spans = []
        for item in payload:
            category = item.get("category")
            spans.append(
                Span(
                    start=item["start"],
                    end=item["end"],
                    category=_category_from_name(self._category_enum, category) if category is not None else None,
                    replacement=item.get("replacement", ""),
                )
            )
        return spans

    def scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult:
        return self.batch_scan([text], max_seq_length, overlap)[0]

    def batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]:
        payload = self._request("POST", "/v1/scan", json=self._scan_payload(texts, max_seq_length, overlap))
        return [self._build_result(item) for item in payload["results"]]

    async def async_scan(
        self, text: str, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> BaseResult:
        return (await self.async_batch_scan([text], max_seq_length, overlap))[0]

    async def async_batch_scan(
        self, texts: list[str], max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, overlap: int = DEFAULT_OVERLAP
    ) -> list[BaseResult]:
        payload = await self._async_request("POST", "/v1/scan", json=self._scan_payload(texts, max_seq_length, overlap))
        return [self._build_result(item) for item in payload["results"]]

    def redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[BaseCategory, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
    ) -> str | tuple[str, list[Span]]:
        return self.batch_redact(
            [text],
            max_seq_length,
            overlap,
            replace,
            policy=policy,
            confidence=confidence,
            return_spans=return_spans,
        )[0]

    def batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[BaseCategory, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]:
        payload = self._request(
            "POST",
            "/v1/redact",
            json=self._redact_payload(texts, max_seq_length, overlap, replace, policy, confidence, return_spans),
        )
        return self._build_redact_outputs(payload["results"], return_spans)

    async def async_redact(
        self,
        text: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[BaseCategory, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
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
            )
        )[0]

    async def async_batch_redact(
        self,
        texts: list[str],
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str | dict[BaseCategory, str] = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        return_spans: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[tuple[str, list[Span]]]:
        payload = await self._async_request(
            "POST",
            "/v1/redact",
            json=self._redact_payload(texts, max_seq_length, overlap, replace, policy, confidence, return_spans),
        )
        return self._build_redact_outputs(payload["results"], return_spans)

    def _build_redact_outputs(
        self, results: list[dict], return_spans: bool
    ) -> list[str] | list[tuple[str, list[Span]]]:
        outputs: list[Any] = []
        for item in results:
            if return_spans:
                outputs.append((item["text"], self._build_spans(item.get("spans") or [])))
            else:
                outputs.append(item["text"])
        return outputs

    def _on_unsafe(self, result: BaseResult, confidence: float) -> None:
        raise NotImplementedError

    def redact_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        replace: str = DEFAULT_REDACTED_LABEL,
        policy: BasePolicy | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        return _make_redact_before_exec(self.redact, self.async_redact)(
            param, max_seq_length, overlap, replace, policy, confidence
        )

    def scan_before_exec(
        self,
        param: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        overlap: int = DEFAULT_OVERLAP,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        return _make_scan_before_exec(self.scan, self.async_scan, self._on_unsafe)(
            param, max_seq_length, overlap, confidence
        )


class APIPromptGuard(APIGuard):
    """Prompt guard that calls a Mezzo Guard HTTP server."""

    model_type = "prompt_guard"
    _category_enum = PromptCategory
    _result_cls = PromptResult

    def _on_unsafe(self, result: BaseResult, confidence: float) -> None:
        unsafe_score = result.scores.get(PromptCategory.UNSAFE, 0.0)
        if unsafe_score >= confidence:
            raise UnsafePromptError(unsafe_score)


class APIContentGuard(APIGuard):
    """Content guard that calls a Mezzo Guard HTTP server."""

    model_type = "content_guard"
    _category_enum = ContentCategory
    _result_cls = ContentResult

    def _on_unsafe(self, result: BaseResult, confidence: float) -> None:
        violated = [{cat.value: score} for cat, score in result.scores.items() if score >= confidence]
        if violated:
            raise UnsafeContentError(violated)


__all__ = [
    "APIContentGuard",
    "APIGuard",
    "APIPromptGuard",
]
