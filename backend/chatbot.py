"""
Chat assistant: online only — Gemini -> Groq -> OpenRouter. No offline FAQ.
Requires internet + GOOGLE_API_KEY / GEMINI_API_KEY, or GROQ_API_KEY, or OPENROUTER_API_KEY.
"""
import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_MODELS = (
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-7b-instruct:free",
)

# Best-first on Google's free AI Studio tier (names tried in order).
GEMINI_MODEL_NAMES = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)


def build_system_instruction(extra_context: str | None = None) -> str:
    base = (
        'You are a helpful assistant inside "Pneumonia AI Lab", a university demo that classifies '
        "chest X-rays with DenseNet121 features, an SVM, and Grad-CAM visualizations.\n"
        "- Always reply in clear English (users may type in any language; still answer in English unless they "
        "explicitly ask for another language).\n"
        "- Answer completely and accurately: use short sections or bullet points when helpful.\n"
        "- Medical topics: general education only; never diagnose; say personal care must follow their doctor.\n"
        "- If a report summary is attached, explain it educationally (what the demo shows, limits), not as medical advice.\n"
        "- Answer questions about current date/time and general knowledge directly and accurately."
    )
    if extra_context and extra_context.strip():
        return base + "\n\n--- Attached context ---\n" + extra_context.strip()
    return base


def followup_suggestions(has_patient_context: bool) -> list[str]:
    if has_patient_context:
        pool = [
            "What does the confidence score mean here?",
            "What are the limits of this AI result?",
            "How should Grad-CAM be interpreted for this case?",
            "What symptoms warrant urgent care?",
        ]
    else:
        pool = [
            "What is pneumonia?",
            "What should a patient with pneumonia do in general?",
            "What are common pneumonia symptoms?",
            "How do DenseNet121 and SVM work in this app?",
            "What is Grad-CAM and why use it?",
            "How can pneumonia be prevented?",
        ]
    return random.sample(pool, min(3, len(pool)))


def _gemini_response_text(resp: Any) -> str:
    t = (getattr(resp, "text", None) or "").strip()
    if t:
        return t
    try:
        cand = resp.candidates[0]
        parts = getattr(cand.content, "parts", None) or []
        return "".join(getattr(p, "text", "") for p in parts).strip()
    except (IndexError, AttributeError, TypeError, ValueError):
        return ""


def _history_for_gemini(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in history:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        content = content[:6000]
        if role == "user":
            out.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            out.append({"role": "model", "parts": [content]})
    return out


def _gemini_reply(
    user_message: str,
    history: list[dict[str, Any]],
    system_instruction: str,
) -> str | None:
    raw = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    key = raw.strip().strip('"').strip("'")
    if not key or not user_message.strip():
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        gem_hist = _history_for_gemini(history)
        for model_name in GEMINI_MODEL_NAMES:
            try:
                try:
                    model = genai.GenerativeModel(
                        model_name,
                        system_instruction=system_instruction,
                    )
                except TypeError:
                    model = genai.GenerativeModel(model_name)
                chat = model.start_chat(history=gem_hist)
                resp = chat.send_message(
                    user_message[:4000],
                    generation_config={
                        "max_output_tokens": 1024,
                        "temperature": 0.35,
                    },
                )
                text = _gemini_response_text(resp)
                if text:
                    return re.sub(r"\n{3,}", "\n\n", text)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _openai_style_chat(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 75,
) -> str | None:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.35,
        }
    ).encode()
    hdrs = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in url:
        hdrs["HTTP-Referer"] = os.environ.get(
            "OPENROUTER_HTTP_REFERER", "http://127.0.0.1:5000"
        )
        hdrs["X-Title"] = "Pneumonia AI Lab"
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        text = text.strip()
        return re.sub(r"\n{3,}", "\n\n", text) if text else None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def _groq_reply(
    user_message: str,
    history: list[dict[str, Any]],
    system_instruction: str,
) -> str | None:
    key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not key or not user_message.strip():
        return None
    messages: list[dict[str, str]] = [{"role": "system", "content": system_instruction}]
    for m in history:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content[:6000]})
    messages.append({"role": "user", "content": user_message[:4000]})
    return _openai_style_chat(
        "https://api.groq.com/openai/v1/chat/completions",
        key,
        "llama-3.3-70b-versatile",
        messages,
    )


def _openrouter_reply(
    user_message: str,
    history: list[dict[str, Any]],
    system_instruction: str,
) -> str | None:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
    if not key or not user_message.strip():
        return None
    messages: list[dict[str, str]] = [{"role": "system", "content": system_instruction}]
    for m in history:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content[:6000]})
    messages.append({"role": "user", "content": user_message[:4000]})
    for model in OPENROUTER_MODELS:
        text = _openai_style_chat(
            "https://openrouter.ai/api/v1/chat/completions",
            key,
            model,
            messages,
            timeout=90,
        )
        if text:
            return text
    return None


def get_chat_reply(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    extra_context: str | None = None,
) -> str | None:
    """Returns assistant text, or None if no provider succeeded (caller should show an error)."""
    history = history or []
    msg = (user_message or "").strip()
    if not msg:
        return None

    system_instruction = build_system_instruction(extra_context)

    text = _gemini_reply(msg, history, system_instruction)
    if text:
        return text
    text = _groq_reply(msg, history, system_instruction)
    if text:
        return text
    text = _openrouter_reply(msg, history, system_instruction)
    if text:
        return text
    return None
