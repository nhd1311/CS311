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


def _env_float(name: str, default: str) -> float:
    v = (os.getenv(name, default) or "").strip()
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: str) -> int:
    v = (os.getenv(name, default) or "").strip()
    try:
        return int(v)
    except Exception:
        return int(default)


# Chat behavior toggles.
# Defaults are intentionally permissive: always answer without extra conditions.
ASK_FOLLOWUPS = _env_flag("ASK_FOLLOWUPS", "0")
STRICT_EXACT_LOOKUP = _env_flag("STRICT_EXACT_LOOKUP", "0")

# Language behavior.
# This project is English-only: the API rejects non-English user queries.
ENGLISH_ONLY = True

# Relevance gating for retrieval results.
# Note: Chroma returns distances (lower is better). In this project we often
# observe L2 distances roughly ~0.6–0.9 for good matches and >~1.3 for weak ones.
# Tune via env if your metric/model differs.
RAG_MAX_DISTANCE = _env_float("RAG_MAX_DISTANCE", "1.0")
RAG_MIN_TOKEN_OVERLAP = _env_float("RAG_MIN_TOKEN_OVERLAP", "0.12")
RAG_MIN_DISTINCTIVE_MATCHES = _env_int("RAG_MIN_DISTINCTIVE_MATCHES", "1")


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

_QUERY_HINT_TOKENS = {
    # Budget / price
    "budget",
    "price",
    "cheap",
    "affordable",
    "under",
    "below",
    "max",
    "maximum",
    "usd",
    "dollar",
    "dollars",
    # Common shopping intent
    "buy",
    "need",
    "want",
    "looking",
    "find",
    "search",
    # Occasion hints
    "work",
    "office",
    "formal",
    "business",
    "interview",
    "casual",
    "party",
    "wedding",
    "travel",
    "gym",
    "sport",
    "sports",
    # Outfit intent
    "outfit",
    "outfits",
    "look",
}


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _looks_like_english_query(text: str) -> bool:
    """Best-effort check for whether the user query is English.

    This project is English-only. We reject:
    - Any non-ASCII text (common for non-English input)
    - Any text that a language detector does not classify as English
    """
    s = (text or "").strip()
    if not s:
        return True

    # Reject non-ASCII characters first.
    # (This immediately blocks Vietnamese input with diacritics.)
    if re.search(r"[^\x00-\x7F]", s):
        return False

    # Keyword-style product searches are very common (e.g. "men black sneakers under $80").
    # Many language detectors perform poorly on such short, noun-heavy inputs.
    tokens = re.findall(r"[a-z0-9]+", s.lower())
    alpha = [t for t in tokens if t.isalpha()]
    if alpha and any((t in _GENERIC_TOKENS) or (t in _QUERY_HINT_TOKENS) for t in alpha):
        return True

    # If it's short and doesn't look like a catalog/search query, be strict.
    if len(s) < 20 or len(alpha) < 4:
        return False

    try:
        from langdetect import DetectorFactory, detect_langs

        # Make detection deterministic.
        DetectorFactory.seed = 0
        langs = detect_langs(s)
        if not langs:
            return True

        for cand in langs:
            if getattr(cand, "lang", None) == "en" and float(getattr(cand, "prob", 0.0)) >= 0.50:
                return True
        return False
    except Exception:
        return False


def _distinctive_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _GENERIC_TOKENS]


def _relevance_filter_docs(
    query: str, docs: List[Any], max_distance: Optional[float] = None
) -> List[Any]:
    """Filter retrieved docs to avoid returning irrelevant items.

    Uses a combination of embedding distance threshold and light lexical checks.
    Distance is the primary gate (lower is better). Lexical checks help avoid
    edge cases where the vector DB returns weak matches.
    """

    if not docs:
        return []

    q_tokens = set(_tokenize(query))
    q_dist = set(_distinctive_tokens(query))

    # If the query has no usable tokens (e.g., non-latin), rely on distance only.
    use_overlap = bool(q_tokens) and RAG_MIN_TOKEN_OVERLAP > 0
    use_distinctive = bool(q_dist) and RAG_MIN_DISTINCTIVE_MATCHES > 0

    max_dist = RAG_MAX_DISTANCE if max_distance is None else float(max_distance)

    kept = []
    for d in docs:
        # 1) Distance gate (primary)
        try:
            dist = float(getattr(d, "score", 1e9))
        except Exception:
            dist = 1e9

        if max_dist > 0 and dist > max_dist:
            continue

        # 2) Lexical overlap gate (secondary)
        if use_overlap:
            # Include metadata text for matching, when present.
            meta = getattr(d, "metadata", None) or {}
            meta_text = " ".join([str(v) for v in meta.values() if v is not None])
            text = f"{getattr(d, 'text', '')} {meta_text}".strip()
            dtoks = set(_tokenize(text))
            overlap = (len(q_tokens & dtoks) / max(1, len(q_tokens)))
            if overlap < RAG_MIN_TOKEN_OVERLAP:
                continue

        if use_distinctive:
            meta = getattr(d, "metadata", None) or {}
            meta_text = " ".join([str(v) for v in meta.values() if v is not None])
            text = f"{getattr(d, 'text', '')} {meta_text}".strip()
            dt = set(_distinctive_tokens(text))
            if len(q_dist & dt) < RAG_MIN_DISTINCTIVE_MATCHES:
                continue

        kept.append(d)

    return kept


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
    ]
    return any(k in t for k in keywords)


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


def _extract_min_budget_usd(text: str) -> Optional[float]:
    """Extract a minimum budget (USD) from free-form text.

    Supports patterns like: "above 200", "over $150", "at least 80", ">= 50", "> 40".
    Returns the largest found number (most restrictive), if any.
    """
    t = (text or "").lower()
    nums: List[float] = []

    patterns = [
        r"\babove\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bover\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bmore\s+than\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bat\s+least\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\bminimum\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\b>=\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
        r"\b>\s*\$?\s*(\d+(?:\.\d{1,2})?)\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, t):
            try:
                nums.append(float(m.group(1)))
            except Exception:
                continue

    if not nums:
        return None
    nums = [n for n in nums if 0 < n < 1_000_000]
    return max(nums) if nums else None


def _normalize_article_type(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    # Keep alphanumerics only to make matching robust (e.g., "T-shirts" -> "tshirts").
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _normalize_color(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    # Keep letters only for robust comparisons.
    s = re.sub(r"[^a-z]+", "", s)
    if s == "grey":
        return "gray"
    return s


def _desired_colors_from_query(text: str) -> Optional[set[str]]:
    """Infer desired colors from the user query.

    Returns a set of canonical dataset color names (Title Case), or None.
    If a color is explicitly mentioned, we will enforce it strictly.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # Map normalized tokens to canonical color labels used in metadata.
    # Note: the dataset commonly uses "Grey".
    color_map: Dict[str, str] = {
        # English
        "black": "Black",
        "white": "White",
        "red": "Red",
        "blue": "Blue",
        "green": "Green",
        "yellow": "Yellow",
        "pink": "Pink",
        "purple": "Purple",
        "orange": "Orange",
        "brown": "Brown",
        "gray": "Grey",
        "grey": "Grey",
        "silver": "Silver",
        "gold": "Gold",
        "beige": "Beige",
        "cream": "Cream",
        "navy": "Navy Blue",
        "maroon": "Maroon",
    }

    toks = re.findall(r"[a-z]+", t)
    found: set[str] = set()
    for tok in toks:
        key = _normalize_color(tok)
        if key in color_map:
            found.add(color_map[key])

    return found or None


def _filter_docs_by_color(docs: List[Any], desired_colors: set[str]) -> List[Any]:
    """Strictly keep only docs whose metadata.color matches a requested color."""
    if not docs:
        return []
    allowed = {_normalize_color(c) for c in desired_colors}
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        c = _normalize_color(meta.get("color"))
        if not c:
            # If we can't verify the color, don't return it when the user requested one.
            continue
        if c in allowed:
            kept.append(d)
    return kept


def _normalize_subcategory(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    return s


def _desired_subcategories_from_query(text: str) -> Optional[set[str]]:
    """Infer desired product subcategory from the user query.

    Returns canonical dataset subcategory labels (e.g., "Bottomwear", "Topwear"),
    or None if no clear intent is detected.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # If user explicitly says bottomwear/topwear, respect that.
    # Also infer bottomwear from common garment words.
    # NOTE: Keep this conservative: only infer when it's strongly implied.
    tt = t
    if re.search(r"\bbottom\s*wear\b", tt) or any(k in tt for k in ["trousers", "pants", "jeans", "shorts", "skirt", "leggings", "chinos"]):
        return {"Bottomwear"}
    if re.search(r"\btop\s*wear\b", tt) or any(k in tt for k in ["shirt", "tshirt", "t-shirt", "tee", "top", "jacket", "hoodie", "sweater", "coat"]):
        # Only infer Topwear if it is explicitly requested; garment words are too broad.
        # We keep this strict to avoid blocking mixed queries.
        if re.search(r"\btop\s*wear\b", tt):
            return {"Topwear"}

    # If user explicitly says footwear/bags, we can infer those too.
    if re.search(r"\bfoot\s*wear\b", tt) or any(k in tt for k in ["shoes", "sneakers", "sandals", "flipflops", "boots"]):
        # Dataset uses many footwear subcategories; don't over-restrict.
        return None
    if any(k in tt for k in ["bag", "handbag", "backpack"]):
        # Bags subcategory exists, but users often mix accessory terms.
        return {"Bags"}

    return None


def _filter_docs_by_subcategory(docs: List[Any], desired_subcategories: set[str]) -> List[Any]:
    """Strictly keep only docs whose metadata.subcategory matches requested subcategory."""
    if not docs:
        return []
    allowed = {_normalize_subcategory(s) for s in desired_subcategories}
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        sub = _normalize_subcategory(meta.get("subcategory"))
        if not sub:
            continue
        if sub in allowed:
            kept.append(d)
    return kept


def _normalize_usage(val: Any) -> str:
    # Dataset values include: Casual, Sports, Ethnic, Formal, Smart Casual, Party, Travel, Home
    s = ("" if val is None else str(val)).strip().lower()
    # Keep letters only to normalize spaces/hyphens ("Smart Casual" -> "smartcasual").
    s = re.sub(r"[^a-z]+", "", s)
    return s


def _desired_usages_from_query(text: str) -> Optional[set[str]]:
    """Infer desired `usage` labels from the user query.

    Returns a set of canonical dataset usage labels, or None if no occasion intent is detected.
    This is used for strict filtering ("lọc cứng") when the user clearly asks for an occasion.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    tt = t
    wanted: set[str] = set()

    # Sports / gym
    if any(k in tt for k in ["gym", "workout", "sport", "sports", "training", "run", "running"]):
        wanted.add("Sports")

    # Travel
    if any(k in tt for k in ["travel", "trip", "vacation", "journey", "tour"]):
        wanted.add("Travel")

    # Home / loungewear
    if any(k in tt for k in ["home", "at home", "loungewear", "pajama", "pjs"]):
        wanted.add("Home")

    # Ethnic / traditional
    if any(k in tt for k in ["ethnic", "traditional", "festival"]):
        wanted.add("Ethnic")

    # Party / events
    if any(k in tt for k in ["party", "club", "date", "prom", "wedding", "reception"]):
        wanted.add("Party")

    # Formal / office / interview
    office_like = any(
        k in tt
        for k in [
            "formal",
            "business",
            "office",
            "work",
            "interview",
            "meeting",
        ]
    )
    if office_like:
        wanted.add("Formal")

    # Casual / everyday
    casual_like = any(
        k in tt
        for k in [
            "casual",
            "everyday",
            "daily",
            "street",
            "hangout",
        ]
    )

    # If the user explicitly mixes casual + office, map to Smart Casual (and keep Formal as a fallback).
    if casual_like and office_like:
        wanted.add("Smart Casual")
    elif casual_like:
        wanted.add("Casual")

    return wanted or None


def _filter_docs_by_usage(docs: List[Any], desired_usages: set[str]) -> List[Any]:
    """Strictly keep only docs whose metadata.usage matches the requested usage labels."""
    if not docs:
        return []
    allowed = {_normalize_usage(u) for u in desired_usages}
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        u = _normalize_usage(meta.get("usage"))
        if not u:
            # If we can't verify the usage, don't return it when the user requested one.
            continue
        if u in allowed:
            kept.append(d)
    return kept


def _usage_query_hints(desired_usages: set[str]) -> str:
    """Return extra query terms to help retrieval surface the requested usage."""
    hints: List[str] = []
    for u in desired_usages:
        nu = _normalize_usage(u)
        if nu == "formal":
            hints.extend(["formal", "business", "office", "work", "interview"])
        elif nu == "smartcasual":
            hints.extend(["smart casual", "business casual", "office"])
        elif nu == "casual":
            hints.extend(["casual", "everyday", "daily", "street"])
        elif nu == "sports":
            hints.extend(["sports", "sport", "gym", "workout", "training"])
        elif nu == "party":
            hints.extend(["party", "wedding", "event", "date"])
        elif nu == "travel":
            hints.extend(["travel", "trip", "vacation"])
        elif nu == "ethnic":
            hints.extend(["ethnic", "traditional", "festival"])
        elif nu == "home":
            hints.extend(["home", "loungewear", "pajama"])
    # De-dupe while preserving order.
    seen = set()
    out: List[str] = []
    for h in hints:
        hh = h.strip().lower()
        if not hh or hh in seen:
            continue
        seen.add(hh)
        out.append(h)
    return " ".join(out)


def _usage_seed_query(desired_usages: set[str]) -> str:
    """Return a lightweight, catalog-oriented seed query for vague occasion requests."""
    seeds: List[str] = []
    for u in desired_usages:
        nu = _normalize_usage(u)
        if nu == "formal":
            seeds.extend(["shirt", "trousers", "pants", "shoes", "tie"])
        elif nu == "smartcasual":
            seeds.extend(["shirt", "chinos", "loafers", "shoes"])
        elif nu == "casual":
            seeds.extend(["tshirt", "jeans", "sneakers"])
        elif nu == "sports":
            seeds.extend(["tshirt", "shorts", "sneakers", "shoes"])
        elif nu == "party":
            seeds.extend(["dress", "heels", "shoes", "clutch", "handbag"])
        elif nu == "travel":
            seeds.extend(["backpack", "shoes", "sneakers", "tshirt"])
        elif nu == "ethnic":
            seeds.extend(["kurta", "saree", "ethnic"])
        elif nu == "home":
            seeds.extend(["pajama", "loungewear", "shorts"])

    # De-dupe while preserving order.
    seen = set()
    out: List[str] = []
    for s in seeds:
        ss = s.strip().lower()
        if not ss or ss in seen:
            continue
        seen.add(ss)
        out.append(s)
    return " ".join(out)


def _with_where_constraint(filters: Optional[Dict[str, Any]], constraint: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new Chroma where-filter with an additional constraint.

    - If filters is empty -> just the constraint.
    - If filters already uses an operator at the top-level -> wrap with $and.
    - If filters is a simple field dict -> merge by wrapping into $and to be safe.
    """
    base = dict(filters or {})
    if not base:
        return dict(constraint)
    if any(str(k).startswith("$") for k in base.keys()):
        return {"$and": [base, dict(constraint)]}
    # Multiple simple constraints are normalized later, but wrapping here avoids edge cases.
    return {"$and": [{k: v} for k, v in base.items()] + [dict(constraint)]}


def _looks_like_outfit_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = [
        # English
        "outfit",
        "outfits",
        "complete look",
        "full look",
        "set",
        "combo",
        "mix and match",
    ]
    return any(k in t for k in keywords)


def _wants_innerwear(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = [
        # English
        "innerwear",
        "underwear",
        "brief",
        "briefs",
        "boxer",
        "boxers",
        "trunk brief",
    ]
    return any(k in t for k in keywords)


def _filter_out_innerwear(docs: List[Any]) -> List[Any]:
    if not docs:
        return []
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        if str(meta.get("subcategory") or "").strip().lower() == "innerwear":
            continue
        kept.append(d)
    return kept


def _assemble_outfit_docs(docs: List[Any], budget_max: Optional[float], top_k: int) -> List[Any]:
    """Select a small, outfit-like set of items.

    Goal: prefer a mix (Topwear + Bottomwear + Footwear) when possible.
    If budget_max is provided, try to keep the total under that budget.
    """
    if not docs:
        return []

    # Bucket candidates by (meta.category/subcategory)
    topwear: List[Any] = []
    bottomwear: List[Any] = []
    footwear: List[Any] = []
    other: List[Any] = []

    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        cat = str(meta.get("category") or "").strip().lower()
        sub = str(meta.get("subcategory") or "").strip().lower()

        if sub == "topwear":
            topwear.append(d)
        elif sub == "bottomwear":
            bottomwear.append(d)
        elif cat == "footwear":
            footwear.append(d)
        else:
            other.append(d)

    def _price_of(doc: Any) -> Optional[float]:
        meta = getattr(doc, "metadata", None) or {}
        return _parse_price_number(meta.get("price"))

    def _sort_key_affordable(doc: Any):
        # Prefer priced items (cheaper first), then more relevant (lower distance).
        p = _price_of(doc)
        try:
            dist = float(getattr(doc, "score", 1e9))
        except Exception:
            dist = 1e9
        return (1 if p is None else 0, 1e18 if p is None else p, dist)

    def _sort_key_relevant(doc: Any):
        try:
            dist = float(getattr(doc, "score", 1e9))
        except Exception:
            dist = 1e9
        p = _price_of(doc)
        return (dist, 1e18 if p is None else p)

    # Sort buckets.
    if budget_max is not None:
        topwear.sort(key=_sort_key_affordable)
        bottomwear.sort(key=_sort_key_affordable)
        footwear.sort(key=_sort_key_affordable)
        other.sort(key=_sort_key_affordable)
    else:
        topwear.sort(key=_sort_key_relevant)
        bottomwear.sort(key=_sort_key_relevant)
        footwear.sort(key=_sort_key_relevant)
        other.sort(key=_sort_key_relevant)

    selected: List[Any] = []
    remaining = budget_max

    def _try_pick(bucket: List[Any]) -> None:
        nonlocal remaining
        for cand in bucket:
            if cand in selected:
                continue
            if remaining is None:
                selected.append(cand)
                return
            price = _price_of(cand)
            if price is None:
                continue
            if price <= remaining + 1e-9:
                selected.append(cand)
                remaining -= price
                return

    # Outfit-like picks.
    _try_pick(topwear)
    _try_pick(bottomwear)
    _try_pick(footwear)

    # Fill remaining slots with other relevant/affordable items.
    while len(selected) < max(2, min(top_k, 4)):
        before = len(selected)
        _try_pick(other)
        if len(selected) == before:
            # As a last resort, pick by relevance (ignoring budget) to avoid empty results.
            pool = [d for d in docs if d not in selected]
            pool.sort(key=_sort_key_relevant)
            if pool:
                selected.append(pool[0])
            else:
                break

    return selected[:top_k]


def _merge_docs_keep_best(*doc_lists: List[Any]) -> List[Any]:
    """Merge QueryResult-like objects by id, keeping the one with the lowest distance score."""
    best_by_id: Dict[str, Any] = {}
    for lst in doc_lists:
        for d in lst or []:
            doc_id = str(getattr(d, "id", ""))
            if not doc_id:
                continue
            try:
                dist = float(getattr(d, "score", 1e9))
            except Exception:
                dist = 1e9
            cur = best_by_id.get(doc_id)
            if cur is None:
                best_by_id[doc_id] = d
                continue
            try:
                cur_dist = float(getattr(cur, "score", 1e9))
            except Exception:
                cur_dist = 1e9
            if dist < cur_dist:
                best_by_id[doc_id] = d
    return list(best_by_id.values())


def _desired_article_types_from_query(text: str) -> Optional[set[str]]:
    """Infer desired articleType values from the user query.

    Returns a set of normalized articleType strings (see _normalize_article_type), or None.
    """
    t = (text or "").lower()

    # T-shirt intent (must be checked before generic "shirt")
    if re.search(r"\b(t\s*-?\s*shirt|tshirt|tee)\b", t):
        # Dataset uses "Tshirts" for t-shirts.
        return {"tshirts"}

    # Shirt intent (only if not asking for t-shirt)
    if re.search(r"\bshirt\b", t):
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
    if ENGLISH_ONLY and not _looks_like_english_query(req.query):
        return {
            "results": [],
            "error": "English only: please rephrase your request in English.",
        }
    # Retrieve candidates, then apply post-filters.
    results = retrieve(req.query, top_k=req.top_k, filters=req.filters)
    desired_usages = _desired_usages_from_query(req.query)
    relevance_query = req.query
    if desired_usages is not None:
        # Help retrieval surface the requested usage before strict filtering.
        hints = _usage_query_hints(desired_usages)
        seed = _usage_seed_query(desired_usages)
        aug = " ".join([req.query, hints, seed]).strip()
        if hints:
            more = retrieve(aug, top_k=min(max(req.top_k * 8, req.top_k), 120), filters=req.filters)
            results = _merge_docs_keep_best(results, more)

        # For relevance gating, only require overlap with the usage label(s).
        # This avoids false negatives when hints/seeds add many tokens.
        relevance_query = " ".join(sorted(desired_usages))

        # Hard-filter support: explicitly retrieve within each requested usage.
        # This helps when semantic similarity doesn't naturally surface Formal/Party/etc. items.
        k_each = min(max(req.top_k * 8, req.top_k), 120)
        per_usage = []
        for u in desired_usages:
            fu = _with_where_constraint(req.filters, {"usage": u})
            per_usage.append(retrieve(aug, top_k=k_each, filters=fu))
        if per_usage:
            results = _merge_docs_keep_best(results, *per_usage)
    relaxed_dist = None
    if desired_usages is not None:
        # Occasion/usage is a hard constraint; allow a looser distance threshold.
        relaxed_dist = max(RAG_MAX_DISTANCE, 3.0)
    results = _relevance_filter_docs(relevance_query, results, max_distance=relaxed_dist)
    # Outfit intent: avoid returning innerwear unless explicitly requested.
    if _looks_like_outfit_request(req.query) and not _wants_innerwear(req.query):
        results = _filter_out_innerwear(results)
    if desired_usages is not None:
        results = _filter_docs_by_usage(results, desired_usages)
    desired_colors = _desired_colors_from_query(req.query)
    if desired_colors is not None:
        results = _filter_docs_by_color(results, desired_colors)
    desired_subcats = _desired_subcategories_from_query(req.query)
    if desired_subcats is not None:
        results = _filter_docs_by_subcategory(results, desired_subcats)
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
            "answer": "What would you like to find? Please enter a short English query (e.g., 'men black sneakers under $80').",
            "products": [],
            "sources": [],
        }

    if ENGLISH_ONLY and not _looks_like_english_query(query):
        return {
            "answer": "I don't understand your question. Please rewrite it.",
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
        # Budget intent words
        if any(k in t for k in ["budget", "under", "below", "max", "maximum", "up to", "<"]):
            if re.search(r"\d", t):
                return True
        # USD-style patterns: $40, 40$, 40 usd
        if re.search(r"\$\s*\d+(?:\.\d{1,2})?", t):
            return True
        if re.search(r"\b\d+(?:\.\d{1,2})?\s*\$\b", t):
            return True
        if re.search(r"\b\d+(?:\.\d{1,2})?\s*(usd|dollars?)\b", t):
            return True
        return False

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
        ]
        return any(k in t for k in keywords)

    def _has_size_or_fit(text: str) -> bool:
        t = text.lower()
        if any(k in t for k in ["size", "form", "fit", "oversize", "regular", "slim"]):
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
            qs.append("What’s your max budget? (USD)")
        if not _has_size_or_fit(text):
            qs.append("What size do you wear and what fit do you prefer (slim/regular/oversized)?")
        # Keep it light: ask at most 2 questions per turn.
        return qs[:2]

    # Budget-aware retrieval: enforce min/max budget when specified.
    budget_min = _extract_min_budget_usd(query)
    budget_max = _extract_max_budget_usd(query)
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        # Conflicting constraints; prefer the stricter one (treat as invalid max).
        budget_max = None

    desired_types = _desired_article_types_from_query(query)
    desired_colors = _desired_colors_from_query(query)
    desired_subcats = _desired_subcategories_from_query(query)
    desired_usages = _desired_usages_from_query(query)
    is_outfit = _looks_like_outfit_request(query)
    wants_innerwear = _wants_innerwear(query)

    # Merge caller-provided filters with inferred budget constraint.
    effective_filters: Dict[str, Any] = dict(req.filters or {})
    if budget_min is not None or budget_max is not None:
        # Prefer numeric compare in Chroma if supported by the installed version.
        # We'll also post-filter below as a safety net.
        price_filter = effective_filters.get("price")
        if isinstance(price_filter, dict):
            out = dict(price_filter)
            if budget_min is not None:
                cur = _parse_price_number(out.get("$gte"))
                out["$gte"] = budget_min if cur is None else max(cur, budget_min)
            if budget_max is not None:
                cur = _parse_price_number(out.get("$lte"))
                out["$lte"] = budget_max if cur is None else min(cur, budget_max)
            effective_filters["price"] = out
        else:
            out: Dict[str, Any] = {}
            if budget_min is not None:
                out["$gte"] = budget_min
            if budget_max is not None:
                out["$lte"] = budget_max
            effective_filters["price"] = out

    # Retrieve more candidates when we need to filter (to avoid ending up with < top_k).
    candidate_k = req.top_k
    if budget_min is not None or budget_max is not None or desired_types is not None or desired_colors is not None or desired_subcats is not None or desired_usages is not None:
        # Usage filtering can be especially brittle if we don't retrieve enough candidates.
        candidate_k = min(max(req.top_k * 10, req.top_k), 120)

    # RAG: retrieve text docs
    # For outfit requests, broaden retrieval so we get a mix of items (top/bottom/footwear).
    if is_outfit and desired_subcats is None and not wants_innerwear:
        k_each = max(min(candidate_k, 25), req.top_k)
        docs_base = retrieve(query, top_k=k_each, filters=effective_filters)
        docs_top = retrieve(query + " topwear shirt tshirt jacket", top_k=k_each, filters=effective_filters)
        docs_bottom = retrieve(query + " bottomwear pants jeans trousers chinos", top_k=k_each, filters=effective_filters)
        docs_foot = retrieve(query + " footwear shoes sneakers", top_k=k_each, filters=effective_filters)
        docs = _merge_docs_keep_best(docs_base, docs_top, docs_bottom, docs_foot)
    else:
        docs = retrieve(query, top_k=candidate_k, filters=effective_filters)

    relevance_query = query

    # If the user asked for an occasion/usage, run an additional retrieval pass with usage hints
    # so we have relevant candidates to filter strictly.
    if desired_usages is not None:
        hints = _usage_query_hints(desired_usages)
        seed = _usage_seed_query(desired_usages)
        aug = " ".join([query, hints, seed]).strip()
        if hints:
            docs_usage = retrieve(aug, top_k=candidate_k, filters=effective_filters)
            docs = _merge_docs_keep_best(docs, docs_usage)

        # For relevance gating, only require overlap with the usage label(s).
        # This avoids false negatives when hints/seeds add many tokens.
        relevance_query = " ".join(sorted(desired_usages))

        # Hard-filter support: explicitly retrieve within each requested usage.
        k_each = max(min(candidate_k, 60), req.top_k)
        per_usage = []
        for u in desired_usages:
            fu = _with_where_constraint(effective_filters, {"usage": u})
            per_usage.append(retrieve(aug, top_k=k_each, filters=fu))
        if per_usage:
            docs = _merge_docs_keep_best(docs, *per_usage)

    # Strict relevance gate: do not return weak matches.
    relaxed_dist = None
    if desired_usages is not None:
        # Occasion/usage is a hard constraint; allow a looser distance threshold.
        relaxed_dist = max(RAG_MAX_DISTANCE, 3.0)
    docs = _relevance_filter_docs(relevance_query, docs, max_distance=relaxed_dist)

    if not docs:
        # Avoid calling the LLM with an empty/irrelevant context.
        return {
            "answer": (
                "I couldn’t find sufficiently relevant products in the current dataset for that request. "
                "Try using 2–6 simple keywords (category + color + gender), or loosen constraints like budget."
            ),
            "products": [],
            "sources": [],
        }

    # If the user asked for outfits, do not suggest innerwear unless they asked for it.
    if is_outfit and not wants_innerwear:
        docs = _filter_out_innerwear(docs)
        if not docs:
            return {
                "answer": (
                    "I couldn’t find relevant outfit items (excluding innerwear) for that request in the dataset. "
                    "Try using keywords like a specific top (t-shirt/shirt) or bottom (jeans/chinos) plus color/budget."
                ),
                "products": [],
                "sources": [],
            }

    # Occasion intent filter (strict): if the user asked for an occasion (usage), don't return other usages.
    if desired_usages is not None:
        docs = _filter_docs_by_usage(docs, desired_usages)
        if not docs:
            wanted = ", ".join(sorted(desired_usages))
            return {
                "answer": (
                    f"I couldn’t find sufficiently relevant items for the requested occasion/usage ({wanted}) in the dataset. "
                    "Try using broader occasion wording (e.g., casual/formal/sports/travel) or relaxing other constraints like color/budget."
                ),
                "products": [],
                "sources": [],
            }

    # Subcategory intent filter (strict): e.g. user asked for bottomwear, don't return topwear.
    if desired_subcats is not None:
        docs = _filter_docs_by_subcategory(docs, desired_subcats)
        if not docs:
            wanted = ", ".join(sorted(desired_subcats))
            return {
                "answer": (
                    f"I couldn’t find sufficiently relevant items in the requested category ({wanted}) in the dataset. "
                    "Try removing the category constraint or using broader keywords."
                ),
                "products": [],
                "sources": [],
            }

    # If this is an outfit request and the user didn't already force a subcategory,
    # try to assemble a small mix (Topwear + Bottomwear + Footwear) within budget.
    if is_outfit and desired_subcats is None and not wants_innerwear:
        assembled = _assemble_outfit_docs(docs, budget_max=budget_max, top_k=req.top_k)
        if assembled:
            docs = assembled

    # Color intent filter (strict): if the user asked for a color, don't return other colors.
    if desired_colors is not None:
        docs = _filter_docs_by_color(docs, desired_colors)
        docs = docs[: req.top_k]
        if not docs:
            wanted = ", ".join(sorted(desired_colors))
            return {
                "answer": (
                    f"I couldn’t find sufficiently relevant items in the requested color ({wanted}) in the dataset. "
                    "Try removing the color constraint or using broader keywords."
                ),
                "products": [],
                "sources": [],
            }

    # Safety: ensure returned products obey the budget range.
    if budget_min is not None or budget_max is not None:
        filtered = []
        for d in docs:
            price = _parse_price_number((d.metadata or {}).get("price"))
            if price is None:
                continue
            if budget_min is not None and price < budget_min - 1e-9:
                continue
            if budget_max is not None and price > budget_max + 1e-9:
                continue
            filtered.append(d)
        docs = filtered[: req.top_k]

        if not docs:
            return {
                "answer": (
                    "I couldn’t find any items with a listed price in the requested range in the dataset. "
                    "Try adjusting the budget range or using broader keywords."
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

    # Final safety: never return more than top_k items.
    docs = docs[: req.top_k]
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
                "usage": meta.get("usage"),
                "snippet": _clean_text_snippet(d.text, max_len=180),
            }
        )

    outfit = None
    if is_outfit and products:
        # Build a structured "bundle" while keeping backward-compatible `products`.
        def _is_top(p: Dict[str, Any]) -> bool:
            return str(p.get("subcategory") or "").strip().lower() == "topwear"

        def _is_bottom(p: Dict[str, Any]) -> bool:
            return str(p.get("subcategory") or "").strip().lower() == "bottomwear"

        def _is_footwear(p: Dict[str, Any]) -> bool:
            return str(p.get("category") or "").strip().lower() == "footwear" or str(p.get("subcategory") or "").strip().lower() == "shoes"

        top_item = next((p for p in products if _is_top(p)), None)
        bottom_item = next((p for p in products if _is_bottom(p)), None)
        footwear_item = next((p for p in products if _is_footwear(p)), None)

        picked_ids = {p["id"] for p in [top_item, bottom_item, footwear_item] if p and p.get("id") is not None}
        accessories = [p for p in products if p.get("id") not in picked_ids]

        def _sum_price(ps: List[Dict[str, Any]]) -> Optional[float]:
            total = 0.0
            seen_any = False
            for pp in ps:
                v = _parse_price_number(pp.get("price"))
                if v is None:
                    continue
                seen_any = True
                total += v
            return total if seen_any else None

        outfit_items = [p for p in [top_item, bottom_item, footwear_item] if p] + accessories
        outfit = {
            "topwear": top_item,
            "bottomwear": bottom_item,
            "footwear": footwear_item,
            "accessories": accessories,
            "estimated_total_price": _sum_price(outfit_items),
        }
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
        products=products,
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
    resp = {
        "answer": answer,
        "products": products,
        "sources": [d.dict() for d in docs],
    }
    if outfit is not None:
        resp["outfit"] = outfit
    return resp
