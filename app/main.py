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


# Các cờ (toggle) hành vi chat.
# Mặc định được đặt “thoáng”: luôn trả lời, không thêm điều kiện ràng buộc.
ASK_FOLLOWUPS = _env_flag("ASK_FOLLOWUPS", "0")
STRICT_EXACT_LOOKUP = _env_flag("STRICT_EXACT_LOOKUP", "0")

# Hành vi ngôn ngữ.
# Project này chỉ dùng tiếng Anh: API từ chối truy vấn không phải tiếng Anh.
ENGLISH_ONLY = True

# Cơ chế “gating” độ liên quan cho kết quả retrieval.
# Lưu ý: Chroma trả về distance (càng thấp càng tốt). Trong project này thường
# thấy L2 distance khoảng ~0.6–0.9 cho match tốt và >~1.3 cho match yếu.
# Có thể tinh chỉnh qua env nếu metric/model của bạn khác.
RAG_MAX_DISTANCE = _env_float("RAG_MAX_DISTANCE", "1.0")
RAG_MIN_TOKEN_OVERLAP = _env_float("RAG_MIN_TOKEN_OVERLAP", "0.12")
RAG_MIN_DISTINCTIVE_MATCHES = _env_int("RAG_MIN_DISTINCTIVE_MATCHES", "1")


_STOPWORDS = {
    # Tiếng Anh
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
    # Thuộc tính phổ biến
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
    # Các từ rất chung trong catalog
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
    # Ngân sách / giá
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
    # Ý định mua sắm phổ biến
    "buy",
    "need",
    "want",
    "looking",
    "find",
    "search",
    # Gợi ý theo dịp sử dụng
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
    # Ý định “outfit”
    "outfit",
    "outfits",
    "look",
}


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _looks_like_english_query(text: str) -> bool:
    """Kiểm tra “best-effort” xem truy vấn của người dùng có phải tiếng Anh hay không.

    Project này chỉ hỗ trợ tiếng Anh. Ta sẽ từ chối:
    - Bất kỳ text không phải ASCII (thường gặp ở ngôn ngữ khác)
    - Bất kỳ text mà bộ phát hiện ngôn ngữ không phân loại là tiếng Anh
    """
    s = (text or "").strip()
    if not s:
        return True

    # Từ chối ký tự không phải ASCII trước.
    # (Điều này chặn ngay tiếng Việt có dấu.)
    if re.search(r"[^\x00-\x7F]", s):
        return False

    # Truy vấn kiểu “từ khoá” rất phổ biến (vd: "men black sneakers under $80").
    # Nhiều bộ detect ngôn ngữ hoạt động kém với input ngắn, nhiều danh từ.
    tokens = re.findall(r"[a-z0-9]+", s.lower())
    alpha = [t for t in tokens if t.isalpha()]
    if alpha and any((t in _GENERIC_TOKENS) or (t in _QUERY_HINT_TOKENS) for t in alpha):
        return True

    # Nếu chuỗi ngắn và không giống truy vấn catalog/tìm kiếm, hãy nghiêm ngặt.
    if len(s) < 20 or len(alpha) < 4:
        return False

    try:
        from langdetect import DetectorFactory, detect_langs

        # Làm cho detection mang tính quyết định (deterministic).
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
    """Lọc các doc truy xuất để tránh trả về item không liên quan.

    Dùng kết hợp ngưỡng distance theo embedding và một số kiểm tra lexical nhẹ.
    Distance là cổng chính (càng thấp càng tốt). Kiểm tra lexical giúp tránh
    các trường hợp edge case khi vector DB trả về match yếu.
    """

    if not docs:
        return []

    q_tokens = set(_tokenize(query))
    q_dist = set(_distinctive_tokens(query))

    # Nếu query không có token hữu ích (vd: non-latin), chỉ dựa vào distance.
    use_overlap = bool(q_tokens) and RAG_MIN_TOKEN_OVERLAP > 0
    use_distinctive = bool(q_dist) and RAG_MIN_DISTINCTIVE_MATCHES > 0

    max_dist = RAG_MAX_DISTANCE if max_distance is None else float(max_distance)

    kept = []
    for d in docs:
        # 1) Cổng distance (chính)
        try:
            dist = float(getattr(d, "score", 1e9))
        except Exception:
            dist = 1e9

        if max_dist > 0 and dist > max_dist:
            continue

        # 2) Cổng overlap lexical (phụ)
        if use_overlap:
            # Gộp thêm text từ metadata để so khớp, nếu có.
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
    # Loại bỏ thẻ HTML cơ bản để snippet dễ đọc.
    s = re.sub(r"<[^>]+>", " ", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def _infer_name_from_text(text: str) -> str:
    # Best-effort: lấy một tiền tố giống tiêu đề, ngắn gọn.
    s = _clean_text_snippet(text, max_len=120)
    # Nhiều dòng bắt đầu bằng tên sản phẩm; giữ ~10–14 từ đầu.
    words = s.split()
    return " ".join(words[:14]).strip()


def _parse_price_number(val: Any) -> Optional[float]:
    """Parse “best-effort” giá (số) được lưu trong metadata."""
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
    # Loại bỏ các ký tự trang trí thường gặp như "$", dấu phẩy và khoảng trắng.
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
    """Trích xuất ngân sách tối đa (USD) từ text tự do.

    Hỗ trợ mẫu như: "under 70", "below $40", "up to 100", "<= 50".
    Trả về số nhỏ nhất tìm được (ràng buộc chặt nhất), nếu có.
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
    # Bỏ qua các ngân sách vô lý.
    nums = [n for n in nums if 0 < n < 1_000_000]
    return min(nums) if nums else None


def _extract_min_budget_usd(text: str) -> Optional[float]:
    """Trích xuất ngân sách tối thiểu (USD) từ text tự do.

    Hỗ trợ mẫu như: "above 200", "over $150", "at least 80", ">= 50", "> 40".
    Trả về số lớn nhất tìm được (ràng buộc chặt nhất), nếu có.
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
    # Chỉ giữ chữ/số để so khớp ổn định (vd: "T-shirts" -> "tshirts").
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _normalize_color(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    # Chỉ giữ chữ để so sánh ổn định.
    s = re.sub(r"[^a-z]+", "", s)
    if s == "grey":
        return "gray"
    return s


def _desired_colors_from_query(text: str) -> Optional[set[str]]:
    """Suy ra màu mong muốn từ truy vấn của người dùng.

    Trả về một tập màu chuẩn theo dataset (Title Case), hoặc None.
    Nếu người dùng nhắc rõ màu, ta sẽ lọc nghiêm ngặt theo màu đó.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # Ánh xạ token đã chuẩn hoá sang nhãn màu chuẩn được dùng trong metadata.
    # Lưu ý: dataset thường dùng "Grey".
    color_map: Dict[str, str] = {
        # Tiếng Anh
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
    """Chỉ giữ doc có metadata.color khớp đúng với màu được yêu cầu."""
    if not docs:
        return []
    allowed = {_normalize_color(c) for c in desired_colors}
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        c = _normalize_color(meta.get("color"))
        if not c:
            # Nếu không xác minh được màu, không trả về khi người dùng có yêu cầu màu.
            continue
        if c in allowed:
            kept.append(d)
    return kept


def _normalize_subcategory(val: Any) -> str:
    s = ("" if val is None else str(val)).strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    return s


def _desired_subcategories_from_query(text: str) -> Optional[set[str]]:
    """Suy ra subcategory mong muốn từ truy vấn của người dùng.

    Trả về nhãn subcategory chuẩn theo dataset (vd: "Bottomwear", "Topwear"),
    hoặc None nếu không phát hiện được ý định rõ ràng.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # Nếu người dùng nói rõ bottomwear/topwear, tôn trọng điều đó.
    # Có thể suy bottomwear từ một số từ trang phục phổ biến.
    # LƯU Ý: Giữ bảo thủ: chỉ suy ra khi thật sự hàm ý mạnh.
    tt = t
    if re.search(r"\bbottom\s*wear\b", tt) or any(k in tt for k in ["trousers", "pants", "jeans", "shorts", "skirt", "leggings", "chinos"]):
        return {"Bottomwear"}
    if re.search(r"\btop\s*wear\b", tt) or any(k in tt for k in ["shirt", "tshirt", "t-shirt", "tee", "top", "jacket", "hoodie", "sweater", "coat"]):
        # Chỉ suy Topwear nếu người dùng yêu cầu rõ; từ trang phục quá rộng.
        # Giữ nghiêm để tránh chặn các truy vấn “trộn”.
        if re.search(r"\btop\s*wear\b", tt):
            return {"Topwear"}

    # Nếu người dùng nói rõ footwear/bags, ta có thể suy ra thêm.
    if re.search(r"\bfoot\s*wear\b", tt) or any(k in tt for k in ["shoes", "sneakers", "sandals", "flipflops", "boots"]):
        # Dataset có nhiều subcategory footwear; không nên siết quá.
        return None
    if any(k in tt for k in ["bag", "handbag", "backpack"]):
        # Subcategory Bags có tồn tại, nhưng người dùng hay trộn thuật ngữ phụ kiện.
        return {"Bags"}

    return None


def _filter_docs_by_subcategory(docs: List[Any], desired_subcategories: set[str]) -> List[Any]:
    """Chỉ giữ doc có metadata.subcategory khớp đúng với subcategory yêu cầu."""
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
    # Giá trị trong dataset gồm: Casual, Sports, Ethnic, Formal, Smart Casual, Party, Travel, Home
    s = ("" if val is None else str(val)).strip().lower()
    # Chỉ giữ chữ để chuẩn hoá khoảng trắng/dấu gạch ("Smart Casual" -> "smartcasual").
    s = re.sub(r"[^a-z]+", "", s)
    return s


def _desired_usages_from_query(text: str) -> Optional[set[str]]:
    """Suy ra nhãn `usage` mong muốn từ truy vấn.

    Trả về tập nhãn usage chuẩn theo dataset, hoặc None nếu không phát hiện ý định theo dịp.
    Dùng cho lọc nghiêm ngặt ("lọc cứng") khi người dùng hỏi rõ theo dịp.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    tt = t
    wanted: set[str] = set()

    # Thể thao / gym
    if any(k in tt for k in ["gym", "workout", "sport", "sports", "training", "run", "running"]):
        wanted.add("Sports")

    # Du lịch
    if any(k in tt for k in ["travel", "trip", "vacation", "journey", "tour"]):
        wanted.add("Travel")

    # Ở nhà / đồ mặc nhà
    if any(k in tt for k in ["home", "at home", "loungewear", "pajama", "pjs"]):
        wanted.add("Home")

    # Dân tộc / truyền thống
    if any(k in tt for k in ["ethnic", "traditional", "festival"]):
        wanted.add("Ethnic")

    # Tiệc / sự kiện
    if any(k in tt for k in ["party", "club", "date", "prom", "wedding", "reception"]):
        wanted.add("Party")

    # Trang trọng / công sở / phỏng vấn
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

    # Casual / thường ngày
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

    # Nếu người dùng trộn casual + office, ánh xạ sang Smart Casual (và vẫn có Formal dự phòng).
    if casual_like and office_like:
        wanted.add("Smart Casual")
    elif casual_like:
        wanted.add("Casual")

    return wanted or None


def _filter_docs_by_usage(docs: List[Any], desired_usages: set[str]) -> List[Any]:
    """Chỉ giữ doc có metadata.usage khớp đúng với usage được yêu cầu."""
    if not docs:
        return []
    allowed = {_normalize_usage(u) for u in desired_usages}
    kept: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        u = _normalize_usage(meta.get("usage"))
        if not u:
            # Nếu không xác minh được usage, không trả về khi người dùng có yêu cầu.
            continue
        if u in allowed:
            kept.append(d)
    return kept


def _usage_query_hints(desired_usages: set[str]) -> str:
    """Trả về các từ khoá bổ sung để retrieval ưu tiên đúng usage được yêu cầu."""
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
    # Khử trùng lặp nhưng vẫn giữ thứ tự.
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
    """Tạo seed query ngắn gọn theo kiểu catalog cho các yêu cầu theo dịp còn mơ hồ."""
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

    # Khử trùng lặp nhưng vẫn giữ thứ tự.
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
    """Tạo where-filter mới của Chroma bằng cách thêm một ràng buộc.

    - Nếu filters rỗng -> chỉ dùng constraint.
    - Nếu filters đã dùng toán tử ở top-level -> bọc bằng $and.
    - Nếu filters là dict trường đơn giản -> gộp bằng cách bọc $and để an toàn.
    """
    base = dict(filters or {})
    if not base:
        return dict(constraint)
    if any(str(k).startswith("$") for k in base.keys()):
        return {"$and": [base, dict(constraint)]}
    # Nhiều ràng buộc đơn sẽ được chuẩn hoá sau, nhưng bọc ở đây giúp tránh edge case.
    return {"$and": [{k: v} for k, v in base.items()] + [dict(constraint)]}


def _looks_like_outfit_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = [
        # Tiếng Anh
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
        # Tiếng Anh
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
    """Chọn một tập item nhỏ theo kiểu “outfit”.

    Mục tiêu: ưu tiên tổ hợp (Topwear + Bottomwear + Footwear) khi có thể.
    Nếu có budget_max, cố gắng giữ tổng chi phí không vượt ngân sách.
    """
    if not docs:
        return []

    # Nhóm ứng viên theo (meta.category/subcategory)
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
        # Ưu tiên item có giá (rẻ trước), sau đó ưu tiên độ liên quan (distance thấp).
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

    # Sắp xếp các nhóm.
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

    # Các lựa chọn kiểu “outfit”.
    _try_pick(topwear)
    _try_pick(bottomwear)
    _try_pick(footwear)

    # Điền các slot còn lại bằng item khác phù hợp/rẻ.
    while len(selected) < max(2, min(top_k, 4)):
        before = len(selected)
        _try_pick(other)
        if len(selected) == before:
            # Phương án cuối: chọn theo độ liên quan (bỏ qua ngân sách) để tránh rỗng.
            pool = [d for d in docs if d not in selected]
            pool.sort(key=_sort_key_relevant)
            if pool:
                selected.append(pool[0])
            else:
                break

    return selected[:top_k]


def _merge_docs_keep_best(*doc_lists: List[Any]) -> List[Any]:
    """Gộp các object kiểu QueryResult theo id, giữ bản có distance thấp nhất."""
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
    """Suy ra các giá trị articleType mong muốn từ truy vấn.

    Trả về tập articleType đã chuẩn hoá (xem _normalize_article_type), hoặc None.
    """
    t = (text or "").lower()

    # Ý định T-shirt (phải kiểm tra trước "shirt" chung)
    if re.search(r"\b(t\s*-?\s*shirt|tshirt|tee)\b", t):
        # Dataset dùng "Tshirts" cho t-shirt.
        return {"tshirts"}

    # Ý định Shirt (chỉ khi không hỏi t-shirt)
    if re.search(r"\bshirt\b", t):
        return {"shirts"}

    return None

app = FastAPI(title="Fashion RAG")

# CORS cho web UI demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # QUAN TRỌNG: allow_origins="*" không thể đi kèm credentials trong trình duyệt.
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
    # Lấy các ứng viên, rồi áp dụng các bộ lọc hậu xử lý.
    results = retrieve(req.query, top_k=req.top_k, filters=req.filters)
    desired_usages = _desired_usages_from_query(req.query)
    relevance_query = req.query
    if desired_usages is not None:
        # Giúp retrieval ưu tiên đúng usage trước khi lọc nghiêm ngặt.
        hints = _usage_query_hints(desired_usages)
        seed = _usage_seed_query(desired_usages)
        aug = " ".join([req.query, hints, seed]).strip()
        if hints:
            more = retrieve(aug, top_k=min(max(req.top_k * 8, req.top_k), 120), filters=req.filters)
            results = _merge_docs_keep_best(results, more)

        # Khi “gating” độ liên quan, chỉ yêu cầu trùng token với nhãn usage.
        # Tránh false negative khi hints/seeds thêm nhiều token.
        relevance_query = " ".join(sorted(desired_usages))

        # Hỗ trợ lọc cứng: truy xuất riêng theo từng usage được yêu cầu.
        # Hữu ích khi độ tương đồng ngữ nghĩa không tự nhiên đưa Formal/Party/... lên.
        k_each = min(max(req.top_k * 8, req.top_k), 120)
        per_usage = []
        for u in desired_usages:
            fu = _with_where_constraint(req.filters, {"usage": u})
            per_usage.append(retrieve(aug, top_k=k_each, filters=fu))
        if per_usage:
            results = _merge_docs_keep_best(results, *per_usage)
    relaxed_dist = None
    if desired_usages is not None:
        # Occasion/usage là ràng buộc cứng; cho phép ngưỡng distance lỏng hơn.
        relaxed_dist = max(RAG_MAX_DISTANCE, 3.0)
    results = _relevance_filter_docs(relevance_query, results, max_distance=relaxed_dist)
    # Ý định outfit: tránh trả về innerwear trừ khi người dùng yêu cầu rõ.
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

        # Bổ sung (enrich) kết quả tìm theo ảnh bằng tên sản phẩm từ text collection.
        # Giúp UI hiển thị tên thân thiện thay vì chỉ có id.
        ids = [r.id for r in results]
        by_id = get_items_by_ids(ids)
        for r in results:
            meta = r.metadata or {}
            # Ưu tiên name/title đã có trong metadata ảnh.
            name = meta.get("name") or meta.get("title")
            if not name:
                info = by_id.get(str(r.id)) or {}
                tmeta = (info.get("metadata") or {}) if isinstance(info, dict) else {}
                ttext = (info.get("text") or "") if isinstance(info, dict) else ""
                name = tmeta.get("name") or tmeta.get("title") or _infer_name_from_text(ttext)
            if name:
                meta["name"] = name
            # Đảm bảo luôn có image_url nếu tồn tại ở bất kỳ collection nào.
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
    # Chat nhiều lượt: nếu có messages thì dùng tin nhắn user cuối làm query.
    query = (req.query or "").strip()
    if not query and req.messages:
        for m in reversed(req.messages):
            if (m.role or "").strip() == "user" and (m.content or "").strip():
                query = m.content.strip()
                break
    if not query:
        # “Thoáng” một chút: trả về hướng dẫn hữu ích thay vì báo lỗi.
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
        # Từ khoá thể hiện ý định về ngân sách
        if any(k in t for k in ["budget", "under", "below", "max", "maximum", "up to", "<"]):
            if re.search(r"\d", t):
                return True
        # Mẫu kiểu USD: $40, 40$, 40 usd
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
            # Tiếng Anh
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
        # size kiểu S/M/L/XL, hoặc vòng eo số 26–40
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
        # Nhẹ nhàng thôi: hỏi tối đa 2 câu mỗi lượt.
        return qs[:2]

    # Retrieval có xét ngân sách: áp min/max nếu người dùng nêu.
    budget_min = _extract_min_budget_usd(query)
    budget_max = _extract_max_budget_usd(query)
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        # Ràng buộc mâu thuẫn; ưu tiên ràng buộc “chặt” hơn (coi max không hợp lệ).
        budget_max = None

    desired_types = _desired_article_types_from_query(query)
    desired_colors = _desired_colors_from_query(query)
    desired_subcats = _desired_subcategories_from_query(query)
    desired_usages = _desired_usages_from_query(query)
    is_outfit = _looks_like_outfit_request(query)
    wants_innerwear = _wants_innerwear(query)

    # Gộp filters do caller cung cấp với ràng buộc ngân sách suy ra.
    effective_filters: Dict[str, Any] = dict(req.filters or {})
    if budget_min is not None or budget_max is not None:
        # Ưu tiên so sánh số trong Chroma nếu phiên bản hỗ trợ.
        # Đồng thời post-filter phía dưới như một “lưới an toàn”.
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

    # Lấy nhiều ứng viên hơn khi cần lọc (tránh cuối cùng < top_k).
    candidate_k = req.top_k
    if budget_min is not None or budget_max is not None or desired_types is not None or desired_colors is not None or desired_subcats is not None or desired_usages is not None:
        # Lọc theo usage dễ “gãy” nếu không truy xuất đủ ứng viên.
        candidate_k = min(max(req.top_k * 10, req.top_k), 120)

    # RAG: truy xuất document văn bản
    # Với yêu cầu outfit, mở rộng retrieval để có mix item (top/bottom/footwear).
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

    # Nếu người dùng hỏi theo dịp/usage, chạy thêm một lượt retrieval có usage hints
    # để có đủ ứng viên phù hợp trước khi lọc nghiêm ngặt.
    if desired_usages is not None:
        hints = _usage_query_hints(desired_usages)
        seed = _usage_seed_query(desired_usages)
        aug = " ".join([query, hints, seed]).strip()
        if hints:
            docs_usage = retrieve(aug, top_k=candidate_k, filters=effective_filters)
            docs = _merge_docs_keep_best(docs, docs_usage)

        # Khi “gating” độ liên quan, chỉ yêu cầu trùng token với nhãn usage.
        # Tránh false negative khi hints/seeds thêm nhiều token.
        relevance_query = " ".join(sorted(desired_usages))

        # Hỗ trợ lọc cứng: truy xuất riêng theo từng usage được yêu cầu.
        k_each = max(min(candidate_k, 60), req.top_k)
        per_usage = []
        for u in desired_usages:
            fu = _with_where_constraint(effective_filters, {"usage": u})
            per_usage.append(retrieve(aug, top_k=k_each, filters=fu))
        if per_usage:
            docs = _merge_docs_keep_best(docs, *per_usage)

    # Gating độ liên quan nghiêm ngặt: không trả về match yếu.
    relaxed_dist = None
    if desired_usages is not None:
        # Occasion/usage là ràng buộc cứng; cho phép ngưỡng distance lỏng hơn.
        relaxed_dist = max(RAG_MAX_DISTANCE, 3.0)
    docs = _relevance_filter_docs(relevance_query, docs, max_distance=relaxed_dist)

    if not docs:
        # Tránh gọi LLM khi context rỗng/không liên quan.
        return {
            "answer": (
                "I couldn’t find sufficiently relevant products in the current dataset for that request. "
                "Try using 2–6 simple keywords (category + color + gender), or loosen constraints like budget."
            ),
            "products": [],
            "sources": [],
        }

    # Nếu người dùng hỏi outfit, không gợi ý innerwear trừ khi họ yêu cầu.
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

    # Lọc theo dịp (strict): nếu người dùng yêu cầu usage, không trả về usage khác.
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

    # Lọc theo subcategory (strict): vd user hỏi bottomwear thì không trả về topwear.
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

    # Nếu là yêu cầu outfit và user chưa “ép” subcategory,
    # thử lắp một mix nhỏ (Topwear + Bottomwear + Footwear) trong ngân sách.
    if is_outfit and desired_subcats is None and not wants_innerwear:
        assembled = _assemble_outfit_docs(docs, budget_max=budget_max, top_k=req.top_k)
        if assembled:
            docs = assembled

    # Lọc theo màu (strict): nếu user yêu cầu màu, không trả về màu khác.
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

    # An toàn: đảm bảo sản phẩm trả về nằm trong khoảng ngân sách.
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

    # Lọc theo loại sản phẩm (vd: user hỏi T-shirt thì không trả về Shirts).
    if desired_types is not None:
        filtered = []
        for d in docs:
            at = _normalize_article_type((d.metadata or {}).get("articleType"))
            if at and at in desired_types:
                filtered.append(d)
        docs = filtered[: req.top_k]

        if not docs:
            # Nói thẳng: không thể thoả ràng buộc loại chính xác với các filters hiện tại.
            wanted = ", ".join(sorted(desired_types))
            msg = f"I couldn't find any items matching the requested type ({wanted}) with the current constraints."
            if budget_max is not None:
                msg += f" Budget cap: ${budget_max:g}."
            return {"answer": msg, "products": [], "sources": []}

    # An toàn cuối: không bao giờ trả về quá top_k item.
    docs = docs[: req.top_k]
    contexts = [d.text for d in docs]

    # Dựng thẻ sản phẩm có cấu trúc để UI render.
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
        # Dựng "bundle" có cấu trúc nhưng vẫn giữ `products` để tương thích ngược.
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

    # Hành vi strict tuỳ chọn: nếu người dùng có vẻ đang hỏi đúng một món cụ thể
    # mà catalog không có, nói rõ điều đó (nhưng vẫn trả về các món gần nhất).
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

    # Nếu không, sinh câu trả lời bằng LLM (hoặc fallback)
    answer = generate_answer(
        query,
        contexts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        messages=msg_history,
        products=products,
    )

    # Tuỳ chọn: hỏi thêm câu hỏi. Mặc định tắt để chat ít “ma sát”.
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
