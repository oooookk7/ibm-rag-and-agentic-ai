import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

# Ensure tests can import app module when run from project root.
LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import app as app_module

BASE_URL = "http://127.0.0.1:5000"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _post_json(url, payload, timeout=2.0):
    import json

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        return exc.code, body


def test_generate_glm5_success(client, monkeypatch):
    def fake_glm5_response(system_prompt, user_prompt):
        assert isinstance(system_prompt, str)
        assert user_prompt == "hello"
        return {"summary": "ok", "sentiment": 90, "response": "hi there"}

    monkeypatch.setattr(app_module, "glm5_response", fake_glm5_response)

    resp = client.post("/generate", json={"message": "hello", "model": "glm5"})
    assert resp.status_code == 200

    payload = resp.get_json()
    assert payload["summary"] == "ok"
    assert payload["sentiment"] == 90
    assert payload["response"] == "hi there"
    assert "duration" in payload
    assert isinstance(payload["duration"], float)


def test_generate_minimax_success(client, monkeypatch):
    def fake_minimax_response(system_prompt, user_prompt):
        assert user_prompt == "need help"
        return {"summary": "minimax", "sentiment": 70, "response": "handled by minimax"}

    monkeypatch.setattr(app_module, "minimax_response", fake_minimax_response)

    resp = client.post("/generate", json={"message": "need help", "model": "minimax"})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["summary"] == "minimax"
    assert payload["response"] == "handled by minimax"


def test_generate_qwen_success(client, monkeypatch):
    def fake_qwen_response(system_prompt, user_prompt):
        assert user_prompt == "what is this"
        return {"summary": "qwen", "sentiment": 55, "response": "handled by qwen"}

    monkeypatch.setattr(app_module, "qwen_response", fake_qwen_response)

    resp = client.post("/generate", json={"message": "what is this", "model": "qwen"})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["summary"] == "qwen"
    assert payload["response"] == "handled by qwen"


def test_generate_missing_input_returns_400(client):
    resp = client.post("/generate", json={"message": "", "model": "glm5"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()

    resp = client.post("/generate", json={"message": "hello"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_generate_invalid_model_returns_400(client):
    resp = client.post("/generate", json={"message": "hello", "model": "unknown"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_generate_backend_exception_returns_500(client, monkeypatch):
    def broken_glm5_response(system_prompt, user_prompt):
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(app_module, "glm5_response", broken_glm5_response)

    resp = client.post("/generate", json={"message": "hello", "model": "glm5"})
    assert resp.status_code == 500
    payload = resp.get_json()
    assert "error" in payload
    assert "upstream failure" in payload["error"]


def test_app_process_generate_endpoint():
    try:
        status, body = _post_json(
            f"{BASE_URL}/generate",
            {"message": "Hello from e2e test", "model": "invalid-model"},
            timeout=5.0,
        )
    except URLError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("Environment blocks localhost HTTP calls in tests.")
        pytest.fail(
            f"Failed to call running Flask app at {BASE_URL}. "
            f"Start server first with `python3 app.py`. Details: {exc}"
        )

    assert status == 400
    assert "Invalid model selection" in body
