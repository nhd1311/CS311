import os
import re
import logging
from typing import Dict, List, Optional

from openai import OpenAI

import requests

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_DEBUG = (os.getenv("LLM_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}

logger = logging.getLogger(__name__)

_openai_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        base_url = LLM_BASE_URL.strip() or "https://api.openai.com/v1"
        # NOTE: This expects an OpenAI-compatible endpoint.
        _openai_client = OpenAI(base_url=base_url, api_key=LLM_API_KEY)
    return _openai_client


def generate_answer(
    question: str,
    contexts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.2,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generate an answer using either:

    - Hugging Face Inference API (if LLM_BASE_URL points to api-inference.huggingface.co)
    - OpenAI-compatible Chat Completions API (otherwise)

    Falls back to a retrieval-only answer if the LLM is unreachable/misconfigured.
    """

    prompt = _build_prompt(question, contexts, messages=messages)
    try:
        base = (LLM_BASE_URL or "").strip().rstrip("/")
        if "api-inference.huggingface.co" in base:
            return _hf_inference_generate(prompt, max_tokens=max_tokens, temperature=temperature)
        if not LLM_MODEL:
            raise RuntimeError("LLM_MODEL is not configured")
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Do not leak secrets. Provide a short diagnostic in logs to help
        # users fix configuration issues (invalid key, wrong model, endpoint down, etc.).
        try:
            base = (LLM_BASE_URL or "").strip().rstrip("/")
            model = (LLM_MODEL or "").strip()
            msg = f"LLM call failed; falling back to retrieval-only. base_url={base!r}, model={model!r}, err={type(e).__name__}: {e}"
            if LLM_DEBUG:
                logger.exception(msg)
            else:
                logger.warning(msg)
        except Exception:
            # Avoid breaking user flow due to logging errors.
            pass
        return _fallback_answer(question, contexts)


def _hf_inference_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    base = (LLM_BASE_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("LLM_BASE_URL is empty")
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not configured")
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not configured")

    url = f"{base}/models/{LLM_MODEL}"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    # HF may return 503 while loading model
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise RuntimeError(f"HF Inference API error: {err}")

    data = resp.json()
    # Common format: [{"generated_text": "..."}]
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"]).strip()
    # Sometimes: {"generated_text": "..."}
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    raise RuntimeError(f"Unexpected HF response: {data}")


def _build_prompt(
    question: str,
    contexts: List[str],
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    history = ""
    if messages:
        lines = []
        for m in messages[-10:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant", "system"}:
                role = "user"
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[role]
            lines.append(f"{tag}: {content}")
        if lines:
            history = "\n".join(lines)

    ctx_block = "\n\n".join([f"[Doc {i+1}] {c}" for i, c in enumerate(contexts)])
    return (
        "You are a fashion shopping assistant. Always reply in English.\n"
        "Goal: help the user find suitable products using the available dataset (RAG).\n\n"
        "HARD RULES (to stay realistic):\n"
        "1) Use ONLY information from PRODUCT CONTEXT. Never invent: price, brand, material, availability/stock, shipping/returns policies.\n"
        "   - If something is missing, explicitly say it is not available in the dataset.\n"
        "2) Do NOT paste raw HTML or long descriptions. Provide short summaries only.\n"
        "3) When recommending, provide 2–4 options. Each option must include: product name + product ID + (price if available) + (color if available) + one short reason.\n"
        "4) Do NOT ask the user follow-up questions. If key details are missing, make conservative assumptions and state them briefly.\n"
        "5) End with one short next-step suggestion (not a question).\n\n"
        "Suggested answer format:\n"
        "- Quick summary: ...\n"
        "- Top picks (2–4):\n"
        "  1) ...\n"
        "  2) ...\n"
        "- Next step: ...\n\n"
        + (f"Conversation history:\n{history}\n\n" if history else "")
        + f"PRODUCT CONTEXT (from the database):\n{ctx_block}\n\n"
        + f"User request: {question}\n"
        "Answer (English):"
    )


def _fallback_answer(question: str, contexts: List[str]) -> str:
    # Retrieval-only fallback that still behaves like a chat assistant.
    if not contexts:
        return (
            "I couldn’t find a good match in the current dataset right now. "
            "Here’s what I can do anyway:\n"
            "- Explain what’s currently indexed: this bot can only recommend items that were ingested into the database.\n"
            "- Suggest query rewrites you can try immediately (no extra details needed):\n"
            "  • Use simple keywords: \"men shirt white\", \"women dress black\", \"sneakers blue\"\n"
            "  • Try broader category terms: \"t-shirt\", \"jeans\", \"jacket\"\n"
            "  • Avoid long sentences; keep 3–6 keywords\n"
            "- If you haven’t ingested data yet, run the ingest scripts and retry.\n\n"
            "Next step: try a shorter keyword query or ingest the dataset into ChromaDB."
        )

    def _shorten(text: str, max_len: int = 140) -> str:
        s = re.sub(r"<[^>]+>", " ", text or "")
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > max_len:
            return s[: max_len - 1].rstrip() + "…"
        return s

    return (
        "(Fallback mode: answering from retrieved results without the LLM.)\n"
        "Here are a few closest matches from the dataset (see the product cards below):\n"
        + "\n".join([f"- Pick {i+1}: {_shorten(c)}" for i, c in enumerate(contexts[:3])])
    )
