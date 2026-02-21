import json
import re

from huggingface_hub import InferenceClient
from config import settings


def _extract_json_object(text):
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _normalize_response(raw_content):
    try:
        parsed = json.loads(_extract_json_object(raw_content))
        summary = str(parsed.get("summary", "")).strip() or "No summary available."
        sentiment = parsed.get("sentiment", 50)
        try:
            sentiment = int(sentiment)
        except (TypeError, ValueError):
            sentiment = 50
        sentiment = max(0, min(100, sentiment))
        response = str(parsed.get("response", "")).strip() or str(raw_content).strip()
        return {"summary": summary, "sentiment": sentiment, "response": response}
    except Exception:
        text = str(raw_content or "").strip()
        return {
            "summary": text[:160] + ("..." if len(text) > 160 else ""),
            "sentiment": 50,
            "response": text or "No response generated.",
        }


def _call_hf_model(system_prompt, user_prompt, model_id, provider=None):
    token = settings.huggingface_api_token
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment.")

    print("\n" + "-" * 15)
    print(f"[model.py] invoking model={model_id}, provider={provider or 'default'}")

    client_kwargs = {"model": model_id, "token": token}
    if provider:
        client_kwargs["provider"] = provider
    client = InferenceClient(**client_kwargs)

    instruction = (
        "Return ONLY valid JSON with this exact schema:\n"
        '{"summary":"string","sentiment":0-100,"response":"string"}\n'
        "No markdown, no extra text."
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\nUser message:\n{user_prompt}"},
        ],
        max_tokens=400,
        temperature=0.2,
        top_p=0.9,
        extra_body={
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        },
    )

    normalized = _normalize_response(completion.choices[0].message.content)
    print(f"[model.py] normalized response from {model_id}: {normalized}")
    return normalized


def glm5_response(system_prompt, user_prompt):
    return _call_hf_model(
        system_prompt,
        user_prompt,
        settings.glm5_model_id,
        provider=settings.glm5_provider,
    )


def minimax_response(system_prompt, user_prompt):
    return _call_hf_model(system_prompt, user_prompt, settings.minimax_model_id)


def qwen_response(system_prompt, user_prompt):
    return _call_hf_model(system_prompt, user_prompt, settings.qwen_model_id)
