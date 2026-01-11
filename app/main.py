import os
import re
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    IngestRequest,
    QueryRequest,
    IngestImageRequest,
    ChatRequest,
)
from .rag import ingest, retrieve, get_items_by_ids
from .image_rag import ingest_images, retrieve_by_image_file
from .llm_client import generate_answer


def _env_flag(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


# Chat behavior toggles.
# Defaults are intentionally permissive: always answer without extra conditions.
ASK_FOLLOWUPS = _env_flag("ASK_FOLLOWUPS", "0")
STRICT_EXACT_LOOKUP = _env_flag("STRICT_EXACT_LOOKUP", "0")


_STOPWORDS = {
    # English
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "under",
    "with",
    "you",
    "your",
    # Vietnamese (minimal)
    "ao",
    "quan",
    "vay",
    "dam",
    "giay",
    "dep",
    "tui",
    "vi",
    "mu",
    "kinh",
    "do",
    "cho",
    "toi",
    "minh",
    "ban",
    "co",
    "la",
    "va",
    "voi",
    "trong",
    "duoi",
    "tren",
    "mot",
    "nhung",
    "nay",
}

_GENERIC_TOKENS = {
    # Common attributes
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "brown",
    "grey",
    "gray",
    "navy",
    "beige",
    "cream",
    "maroon",
    "women",
    "woman",
    "men",
    "man",
    "unisex",
    "kids",
    "kid",
    # Size/fit
    "size",
    "fit",
    "slim",
    "regular",
    "oversize",
    "oversized",
    "xs",
    "s",
    "m",
    "l",
    "xl",
    "xxl",
    "xxxl",
    # Very generic catalog words
    "shirt",
    "tshirt",
    "tee",
    "top",
    "pant",
    "pants",
    "trouser",
    "trousers",
    "jean",
    "jeans",
    "dress",
    "skirt",
    "jacket",
    "coat",
    "shoe",
    "shoes",
    "sneaker",
    "sneakers",
}


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _distinctive_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _GENERIC_TOKENS]


def _looks_like_specific_product_lookup(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        # English
        "do you have",
        "do you sell",
        "looking for",
        "find ",
        "search ",
        "exact",
        "same as",
        "this product",
        "this item",
        "model",
        "sku",
        "code",
        # Vietnamese
        "co ban",
        "co mau",
        "con mau",
        "tim ",
        "kiem ",
        "san pham",
        "mau nay",
        "giong",
        "y nhu",
        "chinh xac",
        "hinh nay",
        "anh nay",
    ]
    return any(k in t for k in keywords)


def _max_token_overlap_ratio(query: str, candidates: List[str]) -> float:
    q = set(_tokenize(query))
    if not q:
        return 1.0
    best = 0.0
    for c in candidates:
        ct = set(_tokenize(c))
        if not ct:
            continue
        best = max(best, len(q & ct) / len(q))
    return best


def _max_distinctive_match_count(query: str, candidates: List[str]) -> int:
    q = set(_distinctive_tokens(query))
    if not q:
        return 0
    best = 0
    for c in candidates:
        ct = set(_distinctive_tokens(c))
        best = max(best, len(q & ct))
    return best


def _clean_text_snippet(text: str, max_len: int = 160) -> str:
    # Remove basic HTML tags to keep snippets readable.
    s = re.sub(r"<[^>]+>", " ", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def _infer_name_from_text(text: str) -> str:
    # Best-effort: take a short title-like prefix.
    s = _clean_text_snippet(text, max_len=120)
    # Many rows start with the product name; keep the first ~10-14 words.
    words = s.split()
    return " ".join(words[:14]).strip()


def _parse_price_number(val: Any) -> Optional[float]:
    """Best-effort parse for numeric price values stored in metadata."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            v = float(val)
            return v if v >= 0 else None
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    # Remove common decorations like "$", commas, and whitespace.
    s = s.replace(",", "")
    s = re.sub(r"[^0-9.]+", "", s)
    if not s:
        return None
    try:
        v = float(s)
        return v if v >= 0 else None
    except Exception:
        return None


def _extract_max_budget_usd(text: str) -> Optional[float]:
    """Extract a max budget (USD) from free-form text.

    Supports patterns like: "under 70", "below $40", "up to 100", "<= 50".
    Returns the smallest found number (most restrictive), if any.
    """
    t = (text or "").lower()
    nums: List[float] = []

    patterns = [
        r"\bunder\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bbelow\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bless\s+than\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bup\s*to\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bmax(?:imum)?\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\b<=\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\b<\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        # Vietnamese-ish
        r"\bdưới\s*(\d+(?:\.\d{1,2})?)\b",
        r"\btoi\s*da\s*(\d+(?:\.\d{1,2})?)\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, t):
            try:
                nums.append(float(m.group(1)))
            except Exception:
                continue

    if not nums:
        return None
    # Ignore obviously nonsensical budgets.
    nums = [n for n in nums if 0 < n < 1_000_000]
    return min(nums) if nums else None


def _normalize_article_type(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    # Keep alphanumerics only to make matching robust (e.g., "T-shirts" -> "tshirts").
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _desired_article_types_from_query(text: str) -> Optional[set[str]]:
    """Infer desired articleType values from the user query.

    Returns a set of normalized articleType strings (see _normalize_article_type), or None.
    """
    t = (text or "").lower()

    # T-shirt intent (must be checked before generic "shirt")
    if re.search(r"\b(t\s*-?\s*shirt|tshirt|tee)\b", t) or any(k in t for k in ["ao thun", "áo thun", "ao phong", "áo phông"]):
        # Dataset uses "Tshirts" for t-shirts.
        return {"tshirts"}

    # Shirt intent (only if not asking for t-shirt)
    if re.search(r"\bshirt\b", t) or any(k in t for k in ["ao so mi", "áo sơ mi", "so mi", "sơ mi"]):
        return {"shirts"}

    return None

app = FastAPI(title="Fashion RAG")

# CORS for demo web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # IMPORTANT: wildcard origins cannot be combined with credentials in browsers.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_items(req: IngestRequest):
    count = ingest(req.items)
    return {"ingested": count}


@app.post("/query")
def query_items(req: QueryRequest):
    results = retrieve(req.query, top_k=req.top_k, filters=req.filters)
    return {"results": [r.dict() for r in results]}


@app.post("/ingest_image")
def ingest_image_items(req: IngestImageRequest):
    try:
        count = ingest_images(req.items)
        return {"ingested": count}
    except Exception as e:
        import traceback
        print(f"[ERROR] ingest_image failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image/upload")
async def search_by_image_upload(file: UploadFile = File(...), top_k: int = 3):
    try:
        data = await file.read()
        results = retrieve_by_image_file(data, top_k=top_k, filters=None)

        # Enrich image-search results with product names from the text collection.
        # This lets the UI show a human-friendly name instead of only the id.
        ids = [r.id for r in results]
        by_id = get_items_by_ids(ids)
        for r in results:
            meta = r.metadata or {}
            # Prefer existing name/title in image metadata.
            name = meta.get("name") or meta.get("title")
            if not name:
                info = by_id.get(str(r.id)) or {}
                tmeta = (info.get("metadata") or {}) if isinstance(info, dict) else {}
                ttext = (info.get("text") or "") if isinstance(info, dict) else ""
                name = tmeta.get("name") or tmeta.get("title") or _infer_name_from_text(ttext)
            if name:
                meta["name"] = name
            # Ensure we always include an image_url if present in either collection.
            if not meta.get("image_url"):
                info = by_id.get(str(r.id)) or {}
                tmeta = (info.get("metadata") or {}) if isinstance(info, dict) else {}
                if tmeta.get("image_url"):
                    meta["image_url"] = tmeta.get("image_url")
            r.metadata = meta

        return {"results": [r.dict() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    # Multi-turn chat: use last user message as the query if provided.
    query = (req.query or "").strip()
    if not query and req.messages:
        for m in reversed(req.messages):
            if (m.role or "").strip() == "user" and (m.content or "").strip():
                query = m.content.strip()
                break
    if not query:
        # Be permissive: return a helpful response instead of failing.
        return {
            "answer": "Bạn muốn hỏi gì hoặc muốn tìm sản phẩm nào? Hãy nhập câu hỏi/mô tả ngắn để mình hỗ trợ.",
            "products": [],
            "sources": [],
        }

    def _conversation_text() -> str:
        parts: List[str] = []
        if req.messages:
            for m in req.messages[-10:]:
                if (m.role or "") == "user" and (m.content or "").strip():
                    parts.append(m.content.strip())
        parts.append(query)
        return "\n".join(parts)

    def _has_budget(text: str) -> bool:
        t = text.lower()
        # Budget intent words (VN + EN)
        if any(k in t for k in ["ngân sách", "budget", "under", "below", "max", "maximum", "up to", "tối đa", "khoảng", "tầm", "dưới", "<"]):
            if re.search(r"\d", t):
                return True
        # USD-style patterns: $40, 40$, 40 usd
        if re.search(r"\$\s*\d+(?:\.\d{1,2})?", t):
            return True
        if re.search(r"\b\d+(?:\.\d{1,2})?\s*\$\b", t):
            return True
        if re.search(r"\b\d+(?:\.\d{1,2})?\s*(usd|dollars?)\b", t):
            return True
        # VN currency-ish markers
        return bool(re.search(r"\b\d{2,}\s*(k|nghìn|tr|triệu|đ|vnd)\b", t))

    def _has_occasion(text: str) -> bool:
        t = text.lower()
        keywords = [
            # English
            "work",
            "office",
            "formal",
            "business",
            "interview",
            "party",
            "wedding",
            "date",
            "casual",
            "everyday",
            "travel",
            "vacation",
            "gym",
            "workout",
            "sport",
            "sports",
            "school",
            # Vietnamese
            "đi làm",
            "công sở",
            "đi tiệc",
            "dự tiệc",
            "đi chơi",
            "dạo phố",
            "đi học",
            "du lịch",
            "tập gym",
            "thể thao",
            "đám cưới",
            "phỏng vấn",
            "ở nhà",
        ]
        return any(k in t for k in keywords)

    def _has_size_or_fit(text: str) -> bool:
        t = text.lower()
        if any(k in t for k in ["size", "form", "fit", "oversize", "regular", "slim", "vừa", "rộng", "ôm"]):
            return True
        # sizes like S/M/L/XL, numeric waist sizes 26-40
        if re.search(r"\b(xs|s|m|l|xl|xxl|xxxl)\b", t):
            return True
        if re.search(r"\b(2[6-9]|3[0-9]|40)\b", t):
            return True
        return False

    def _follow_up_questions(text: str) -> List[str]:
        qs: List[str] = []
        if not _has_occasion(text):
            qs.append("What’s the occasion (work, casual, party, travel, etc.)?")
        if not _has_budget(text):
            qs.append("What’s your max budget? (USD/VND is fine)")
        if not _has_size_or_fit(text):
            qs.append("What size do you wear and what fit do you prefer (slim/regular/oversized)?")
        # Keep it light: ask at most 2 questions per turn.
        return qs[:2]

    # Budget-aware retrieval: if the user specifies a max budget, enforce it.
    budget_max = _extract_max_budget_usd(query)

    desired_types = _desired_article_types_from_query(query)

    # Merge caller-provided filters with inferred budget constraint.
    effective_filters: Dict[str, Any] = dict(req.filters or {})
    if budget_max is not None:
        # Prefer numeric compare in Chroma if supported by the installed version.
        # We'll also post-filter below as a safety net.
        price_filter = effective_filters.get("price")
        if isinstance(price_filter, dict):
            # Merge/override $lte with the most restrictive cap.
            cur = _parse_price_number(price_filter.get("$lte"))
            effective_filters["price"] = {"$lte": budget_max if cur is None else min(cur, budget_max)}
        else:
            effective_filters["price"] = {"$lte": budget_max}

    # Retrieve more candidates when we need to filter (to avoid ending up with < top_k).
    candidate_k = req.top_k
    if budget_max is not None or desired_types is not None:
        candidate_k = min(max(req.top_k * 8, req.top_k), 50)

    # RAG: retrieve text docs
    docs = retrieve(query, top_k=candidate_k, filters=effective_filters)

    # Safety: ensure returned products do not exceed the budget.
    if budget_max is not None:
        filtered = []
        for d in docs:
            price = _parse_price_number((d.metadata or {}).get("price"))
            if price is None:
                continue
            if price <= budget_max + 1e-9:
                filtered.append(d)
        docs = filtered[: req.top_k]

        if not docs:
            return {
                "answer": (
                    f"I couldn’t find any items with a listed price under ${budget_max:g} in the dataset. "
                    "Try increasing the budget or using broader keywords."
                ),
                "products": [],
                "sources": [],
            }

    # Product-type intent filter (e.g., user asked for T-shirt, don't return Shirts).
    if desired_types is not None:
        filtered = []
        for d in docs:
            at = _normalize_article_type((d.metadata or {}).get("articleType"))
            if at and at in desired_types:
                filtered.append(d)
        docs = filtered[: req.top_k]

        if not docs:
            # Be honest: we can't satisfy the exact type constraint with the current filters.
            wanted = ", ".join(sorted(desired_types))
            msg = f"I couldn't find any items matching the requested type ({wanted}) with the current constraints."
            if budget_max is not None:
                msg += f" Budget cap: ${budget_max:g}."
            return {"answer": msg, "products": [], "sources": []}
    contexts = [d.text for d in docs]

    # Build structured product cards for UI rendering.
    products = []
    for d in docs:
        meta = d.metadata or {}
        name = meta.get("name") or meta.get("title") or _infer_name_from_text(d.text)
        products.append(
            {
                "id": d.id,
                "name": name,
                "image_url": meta.get("image_url"),
                "price": meta.get("price"),
                "color": meta.get("color"),
                "gender": meta.get("gender"),
                "category": meta.get("category"),
                "subcategory": meta.get("subcategory"),
                "snippet": _clean_text_snippet(d.text, max_len=180),
            }
        )
    msg_history = [m.dict() for m in req.messages] if req.messages else None

    # Optional strict behavior: if the user is likely asking for an exact item
    # that isn't in our catalog, say that clearly (but still return closest matches).
    if STRICT_EXACT_LOOKUP and products and _looks_like_specific_product_lookup(query):
        candidate_texts = [
            f"{p.get('name','')} {p.get('category','')} {p.get('subcategory','')} {p.get('color','')}"
            for p in products
        ]
        q_dist = _distinctive_tokens(query)
        best_distinctive_matches = _max_distinctive_match_count(query, candidate_texts)
        min_required = 2 if len(set(q_dist)) >= 3 else 1
        if q_dist and best_distinctive_matches < min_required:
            answer = (
                "I can't find that exact product in my dataset. "
                "However, here are the closest similar items I do have:"
            )
            return {
                "answer": answer,
                "products": products,
                "sources": [d.dict() for d in docs],
            }

    # Otherwise, generate with LLM (or fallback)
    answer = generate_answer(
        query,
        contexts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        messages=msg_history,
    )

    # Optional: ask follow-ups. Disabled by default to keep chat frictionless.
    if ASK_FOLLOWUPS:
        followups = _follow_up_questions(_conversation_text())
        if followups:
            answer = (
                f"{answer}\n\n"
                "Before I lock in the best picks, quick questions:\n"
                + "\n".join([f"- {q}" for q in followups])
            )
    return {
        "answer": answer,
        "products": products,
        "sources": [d.dict() for d in docs],
    }
