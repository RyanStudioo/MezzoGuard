import threading
import time

import pytest
import uvicorn

from mezzoguard import APIGuard, GuardBase, Policy
from mezzoguard.errors import UnsafeInputError
from mezzoguard.server import create_app


def start_server(app) -> tuple[uvicorn.Server, threading.Thread, str]:
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("uvicorn server failed to start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def prompt_url(prompt_guard):
    server, thread, url = start_server(create_app(prompt_guard))
    yield url
    stop_server(server, thread)


@pytest.fixture
def content_url(content_guard):
    server, thread, url = start_server(create_app(content_guard))
    yield url
    stop_server(server, thread)


def test_api_info(prompt_url):
    with APIGuard(prompt_url) as guard:
        assert isinstance(guard, GuardBase)
        assert guard.info["categories"] == ["unsafe"]


def test_remote_scan_matches_local(prompt_url):
    with APIGuard(prompt_url) as client:
        result = client.scan("kill the process")
    assert result.is_unsafe()
    assert result.scores["unsafe"] == 0.99
    assert result.chunks == [[{"label": "unsafe", "score": 0.99}]]


def test_remote_batch_scan(content_url):
    with APIGuard(content_url) as client:
        results = client.batch_scan(["I will kill you", "hello"])
    assert results[0].scores["violence"] == 0.98
    assert results[1].scores["violence"] == 0.05


def test_remote_redact_with_spans(content_url):
    with APIGuard(content_url) as client:
        redacted, spans = client.redact("I will kill you with a weapon", return_spans=True)
    assert "[REDACTED]" in redacted
    assert spans[0].category == "violence"
    assert spans[0].start == 0


def test_remote_redact_custom_replacements(content_url):
    with APIGuard(content_url) as client:
        redacted = client.redact("I will kill you", replace={"violence": "[VIOLENCE]"})
    assert redacted == "[VIOLENCE]"


def test_remote_redact_policy(content_url):
    with APIGuard(content_url) as client:
        untouched = client.redact("I will kill you", policy=Policy(default_threshold=0.999))
    assert untouched == "I will kill you"


def test_remote_scan_before_exec(prompt_url):
    client = APIGuard(prompt_url)

    @client.scan_before_exec(param="prompt", confidence=0.5)
    def chat(prompt: str) -> str:
        return prompt

    assert chat("hello") == "hello"
    with pytest.raises(UnsafeInputError):
        chat("kill")
    client.close()


async def test_remote_async_scan(prompt_url):
    client = APIGuard(prompt_url)
    try:
        results = await client.async_batch_scan(["hello", "kill"])
        assert results[0].is_unsafe() is False
        assert results[1].is_unsafe() is True
    finally:
        await client.aclose()


async def test_remote_async_redact(content_url):
    client = APIGuard(content_url)
    try:
        redacted = await client.async_redact("I will kill you")
        assert redacted == "[REDACTED]"
    finally:
        await client.aclose()
