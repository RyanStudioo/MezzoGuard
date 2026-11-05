from fastapi.testclient import TestClient

from mezzoguard.server import create_app


def make_client(guard, model_type: str, **kwargs) -> TestClient:
    return TestClient(create_app(guard, model_type=model_type, **kwargs))


def test_healthz(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["model"] == "RyanStudio/Mezzo-Prompt-Guard-v2-Small"
    assert body["model_type"] == "prompt_guard"


def test_info_lists_categories_and_defaults(content_guard):
    with make_client(content_guard, "content_guard") as client:
        response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "content_guard"
    assert body["backend"] == "torch"
    assert body["categories"] == ["hate-speech", "self-harm", "sexual", "toxic", "violence"]
    assert body["default_max_seq_length"] == 64
    assert body["default_overlap"] == 16
    assert body["default_replace"] == "[REDACTED]"


def test_scan_prompt_scores_and_violations(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.post("/v1/scan", json={"texts": ["kill the process", "hello there"]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["scores"] == {"unsafe": 0.99}
    assert results[0]["violated"] == ["unsafe"]
    assert results[0]["unsafe"] is True
    assert results[1]["scores"] == {"unsafe": 0.0}
    assert results[1]["violated"] == []
    assert results[1]["unsafe"] is False


def test_scan_confidence_threshold(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.post("/v1/scan", json={"texts": ["kill"], "confidence": 0.999})
    assert response.json()["results"][0]["violated"] == []


def test_scan_policy_overrides_confidence(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.post("/v1/scan", json={"texts": ["kill"], "policy": {"unsafe": 0.999}})
    assert response.json()["results"][0]["violated"] == []


def test_scan_content_multiple_violations(content_guard):
    with make_client(content_guard, "content_guard") as client:
        response = client.post("/v1/scan", json={"texts": ["I will kill you"]})
    result = response.json()["results"][0]
    assert result["scores"]["violence"] == 0.98
    assert set(result["violated"]) == {"toxic", "violence"}
    assert result["unsafe"] is True


def test_scan_include_chunks(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        without = client.post("/v1/scan", json={"texts": ["kill"]}).json()["results"][0]
        with_chunks = client.post("/v1/scan", json={"texts": ["kill"], "include_chunks": True}).json()["results"][0]
    assert without["chunks"] is None
    assert with_chunks["chunks"] == [{"label": "unsafe", "score": 0.99}]


def test_redact_content_spans(content_guard):
    with make_client(content_guard, "content_guard") as client:
        response = client.post("/v1/redact", json={"texts": ["I will kill you"], "return_spans": True})
    result = response.json()["results"][0]
    assert result["text"] == "[REDACTED]"
    assert result["spans"] == [{"start": 0, "end": 15, "category": "violence", "replacement": "[REDACTED]"}]


def test_redact_custom_replacements(content_guard):
    with make_client(content_guard, "content_guard") as client:
        response = client.post(
            "/v1/redact",
            json={"texts": ["I will kill you"], "replacements": {"violence": "[VIOLENCE]"}},
        )
    assert response.json()["results"][0]["text"] == "[VIOLENCE]"


def test_redact_prompt(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.post("/v1/redact", json={"texts": ["please kill it", "hello"]})
    results = response.json()["results"]
    assert results[0]["text"] == "[REDACTED]"
    assert results[1]["text"] == "hello"


def test_batch_size_limit(prompt_guard):
    with make_client(prompt_guard, "prompt_guard", max_batch_texts=1) as client:
        response = client.post("/v1/scan", json={"texts": ["a", "b"]})
    assert response.status_code == 413


def test_text_length_limit(prompt_guard):
    with make_client(prompt_guard, "prompt_guard", max_text_chars=3) as client:
        response = client.post("/v1/scan", json={"texts": ["much too long"]})
    assert response.status_code == 413


def test_empty_batch_rejected(prompt_guard):
    with make_client(prompt_guard, "prompt_guard") as client:
        response = client.post("/v1/scan", json={"texts": []})
    assert response.status_code == 422
