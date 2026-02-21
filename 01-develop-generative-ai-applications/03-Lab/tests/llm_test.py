import sys
from pathlib import Path

# Ensure tests can import app modules when run from project root.
LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import model
from model import glm5_response, minimax_response, qwen_response


SYSTEM_PROMPT = "You are a helpful assistant who provides concise and accurate answers."
USER_PROMPT = "What is the capital of Canada? Include one cool fact."


def _assert_valid_payload(result):
    assert isinstance(result, dict), "Model response must be a dict."
    assert "summary" in result, "Missing 'summary' key."
    assert "sentiment" in result, "Missing 'sentiment' key."
    assert "response" in result, "Missing 'response' key."

    assert isinstance(result["summary"], str), "'summary' must be a string."
    assert isinstance(result["response"], str), "'response' must be a string."
    assert result["summary"].strip(), "'summary' must not be empty."
    assert result["response"].strip(), "'response' must not be empty."

    assert isinstance(result["sentiment"], int), "'sentiment' must be an int."
    assert 0 <= result["sentiment"] <= 100, "'sentiment' must be within [0, 100]."


def _print_result(model_name, result):
    print(f"\n[{model_name}] result:")
    print(result)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def create(self, **kwargs):
        return _FakeCompletion(
            "{\"summary\":\"Ottawa is the capital.\",\"sentiment\":84,\"response\":\"The capital of Canada is Ottawa. It has the world's largest skating rink in winter.\"}"
        )


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeInferenceClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _FakeChat()


class _FakeSettings:
    huggingface_api_token = "test-token"
    glm5_model_id = "zai-org/GLM-5"
    glm5_provider = "novita"
    minimax_model_id = "MiniMaxAI/MiniMax-M2.5"
    qwen_model_id = "Qwen/Qwen3-235B-A22B"


def test_glm5_response(monkeypatch):
    monkeypatch.setattr(model, "InferenceClient", _FakeInferenceClient)
    monkeypatch.setattr(model, "settings", _FakeSettings())
    result = glm5_response(SYSTEM_PROMPT, USER_PROMPT)
    _print_result("GLM-5", result)
    _assert_valid_payload(result)


def test_minimax_response(monkeypatch):
    monkeypatch.setattr(model, "InferenceClient", _FakeInferenceClient)
    monkeypatch.setattr(model, "settings", _FakeSettings())
    result = minimax_response(SYSTEM_PROMPT, USER_PROMPT)
    _print_result("MiniMax-M2.5", result)
    _assert_valid_payload(result)


def test_qwen_response(monkeypatch):
    monkeypatch.setattr(model, "InferenceClient", _FakeInferenceClient)
    monkeypatch.setattr(model, "settings", _FakeSettings())
    result = qwen_response(SYSTEM_PROMPT, USER_PROMPT)
    _print_result("Qwen3-235B-A22B", result)
    _assert_valid_payload(result)
