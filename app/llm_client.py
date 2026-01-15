import os
import re
import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

import requests

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_DEBUG = (os.getenv("LLM_DEBUG", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}

# Nếu bật, API sẽ gọi LLM (đã cấu hình) để sinh `answer` ngay cả khi đã có `products`.
#
# Hành vi mặc định (tắt) là xác định: `answer` được dựng từ `products` để đảm bảo
# nhất quán với thẻ sản phẩm trong UI và tránh bịa/hallucination.
USE_LLM_ANSWER = (os.getenv("USE_LLM_ANSWER", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}

logger = logging.getLogger(__name__)

_openai_client = None


def _format_price_short(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        n = float(val)
    except Exception:
        s = str(val).strip()
        return s or None
    if not (n >= 0):
        return None
    if abs(n - round(n)) < 1e-9:
        return f"${int(round(n))}"
    return f"${n:.2f}"


def _answer_from_products(question: str, products: List[Dict[str, Any]]) -> str:
    # Đảm bảo thứ tự ổn định và không trùng lặp.
    seen = set()
    picks: List[Dict[str, Any]] = []
    for p in products or []:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if pid is None:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        picks.append(p)
        if len(picks) >= 4:
            break

    if not picks:
        return (
            "I couldn’t find sufficiently relevant products in the current dataset for that request. "
            "Try using 2–6 simple keywords (category + color + gender), or loosen constraints like budget."
        )

    lines: List[str] = []
    q = (question or "").strip()
    if q:
        lines.append(f"Quick summary: Here are the closest matches for: {q}.")
    else:
        lines.append("Quick summary: Here are the closest matches I found in the dataset.")
    lines.append("")
    lines.append("Top picks:")

    for i, p in enumerate(picks, start=1):
        name = (p.get("name") or str(p.get("id") or "")).strip()
        pid = p.get("id")
        price = _format_price_short(p.get("price")) or "price not listed"
        color = (p.get("color") or "").strip()
        gender = (p.get("gender") or "").strip()
        cat = (p.get("category") or "").strip()
        sub = (p.get("subcategory") or "").strip()
        usage = (p.get("usage") or "").strip()

        attrs = []
        if price:
            attrs.append(str(price))
        if color:
            attrs.append(color)
        if gender:
            attrs.append(gender)
        if cat or sub:
            attrs.append(" / ".join([x for x in [cat, sub] if x]))
        attr_txt = " - " + " - ".join(attrs) if attrs else ""

        reason_parts = []
        if usage:
            reason_parts.append(f"Good for {usage.lower()}.")
        if sub and not usage:
            reason_parts.append(f"Matches {sub.lower()}.")
        if not reason_parts:
            reason_parts.append("A close match from the dataset.")
        reason = " ".join(reason_parts)

        lines.append(f"{i}) {name} ({pid}){attr_txt} - {reason}")

    lines.append("")
    lines.append("Next step: Open a product image card to verify style and details.")
    return "\n".join(lines).strip()


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        base_url = LLM_BASE_URL.strip() or "https://api.openai.com/v1"
        # LƯU Ý: Đầu cuối (endpoint) cần tương thích chuẩn OpenAI.
        _openai_client = OpenAI(base_url=base_url, api_key=LLM_API_KEY)
    return _openai_client


def generate_answer(
    question: str,
    contexts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.2,
    messages: Optional[List[Dict[str, str]]] = None,
    products: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Sinh câu trả lời theo một trong hai cách:

    - Hugging Face Inference API (nếu LLM_BASE_URL trỏ tới api-inference.huggingface.co)
    - OpenAI-compatible Chat Completions API (nếu không)

    Nếu LLM không truy cập được/cấu hình sai, sẽ fallback sang trả lời chỉ dựa trên retrieval.
    """

    # QUAN TRỌNG: UI dựng thẻ sản phẩm từ `products` mà API trả về.
    # Để tránh lệch giữa nội dung assistant và các thẻ đang hiển thị,
    # ta sinh câu trả lời một cách xác định từ `products` bất cứ khi nào có thể,
    # trừ khi bật USE_LLM_ANSWER.
    if products and not USE_LLM_ANSWER:
        return _answer_from_products(question, products)

    # Nếu người dùng bật LLM answer nhưng chưa cấu hình LLM,
    # fallback sang output xác định để UI vẫn nhất quán.
    if products and USE_LLM_ANSWER and not (LLM_API_KEY and LLM_MODEL):
        return _answer_from_products(question, products)

    prompt = _build_prompt(question, contexts, messages=messages, products=products)
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
        # Không làm lộ bí mật. Chỉ ghi chẩn đoán ngắn trong log để người dùng
        # sửa cấu hình (key sai, model sai, endpoint down, v.v.).
        try:
            base = (LLM_BASE_URL or "").strip().rstrip("/")
            model = (LLM_MODEL or "").strip()
            msg = f"LLM call failed; falling back to retrieval-only. base_url={base!r}, model={model!r}, err={type(e).__name__}: {e}"
            if LLM_DEBUG:
                logger.exception(msg)
            else:
                logger.warning(msg)
        except Exception:
            # Tránh làm gián đoạn luồng người dùng vì lỗi logging.
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
    # HF có thể trả 503 trong lúc đang tải model
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise RuntimeError(f"HF Inference API error: {err}")

    data = resp.json()
    # Định dạng phổ biến: [{"generated_text": "..."}]
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"]).strip()
    # Đôi khi: {"generated_text": "..."}
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    raise RuntimeError(f"Unexpected HF response: {data}")


def _build_prompt(
    question: str,
    contexts: List[str],
    messages: Optional[List[Dict[str, str]]] = None,
    products: Optional[List[Dict[str, Any]]] = None,
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

    def _json_slim(v: Any) -> str:
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    ctx_block = "\n\n".join([f"[Doc {i+1}] {c}" for i, c in enumerate(contexts)])

    # UI dựng thẻ sản phẩm từ `products` mà API trả về.
    # Để câu trả lời của assistant nhất quán với các thẻ đó, ta cung cấp cùng
    # một danh sách có cấu trúc ở đây và coi nó là nguồn chuẩn.
    products_block = ""
    if products:
        # Chỉ giữ các trường mà ta muốn LLM trích dẫn.
        safe_products = []
        for p in products:
            if not isinstance(p, dict):
                continue
            safe_products.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "price": p.get("price"),
                    "color": p.get("color"),
                    "gender": p.get("gender"),
                    "category": p.get("category"),
                    "subcategory": p.get("subcategory"),
                    "usage": p.get("usage"),
                    "image_url": p.get("image_url"),
                }
            )
        if safe_products:
            products_block = (
                "PRODUCT LIST (authoritative; must match the product cards exactly):\n"
                + _json_slim(safe_products)
                + "\n\n"
            )
    return (
        "You are a fashion shopping assistant. Always reply in English.\n"
        "Goal: help the user find suitable products using the available dataset (RAG).\n\n"
        "HARD RULES (to stay realistic):\n"
        "1) Use ONLY information from PRODUCT CONTEXT and PRODUCT LIST. Never invent: price, brand, material, availability/stock, shipping/returns policies.\n"
        "   - If something is missing, explicitly say it is not available in the dataset.\n"
        "2) If PRODUCT LIST is provided, you MUST NOT contradict it.\n"
        "   - Only recommend items from PRODUCT LIST.\n"
        "   - Any price/color/gender/category you mention MUST exactly match PRODUCT LIST. If price is null/missing, say 'price not listed'.\n"
        "3) Do NOT paste raw HTML or long descriptions. Provide short summaries only.\n"
        "4) When recommending, provide 2–4 options. Each option must include: product name + product ID + (price if available) + (color if available) + one short reason.\n"
        "5) Do NOT ask the user follow-up questions. If key details are missing, make conservative assumptions and state them briefly.\n"
        "6) End with one short next-step suggestion (not a question).\n\n"
        "Suggested answer format:\n"
        "- Quick summary: ...\n"
        "- Top picks (2–4):\n"
        "  1) ...\n"
        "  2) ...\n"
        "- Next step: ...\n\n"
        + (f"Conversation history:\n{history}\n\n" if history else "")
        + products_block
        + f"PRODUCT CONTEXT (from the database):\n{ctx_block}\n\n"
        + f"User request: {question}\n"
        "Answer (English):"
    )


def _fallback_answer(question: str, contexts: List[str]) -> str:
    # Fallback chỉ dựa trên retrieval nhưng vẫn trả lời theo kiểu trợ lý chat.
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
