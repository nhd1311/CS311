"""build_gold_testset.py

Sinh bộ testset có "gold IDs" để đánh giá retrieval metrics (Hit@K / MRR / nDCG).

Mục tiêu:
- Tạo test cases đa dạng, chủ yếu là truy vấn 1 loại sản phẩm rõ ràng (dễ gán nhãn).
- Tự động gợi ý 3 "gold ids" dựa trên lọc metadata + so khớp token đơn giản.

Lưu ý quan trọng:
- Gold tốt nhất nên được con người rà soát. File JSON sinh ra ở đây là "điểm khởi đầu"
  để bạn chỉnh lại (thêm/bớt id) cho đúng ý nghĩa truy vấn.

Chạy ví dụ:
  python eval/build_gold_testset.py \
    --csv datasets/archive/fashion-dataset/styles.csv \
    --out eval/testset_gold.json \
    --gold-per-case 3
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _norm_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _norm_color(s: str) -> str:
    t = _norm_str(s).lower()
    t = re.sub(r"[^a-z]+", "", t)
    if t == "grey":
        t = "gray"
    return t


def _norm_article_type(s: str) -> str:
    t = _norm_str(s).lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


@dataclass
class TestCaseSpec:
    name: str
    query: str
    expected_constraints: Optional[Dict[str, Any]] = None
    should_reject: bool = False
    top_k: int = 5


def _build_cases(top_k: int) -> List[TestCaseSpec]:
    """Danh sách testcases mở rộng.

    Quy ước constraints:
      - gender/color/usage/articleType/subcategory/category
      - min_price/max_price (USD)
    """

    # Lưu ý: cố gắng tạo case đơn giản (1-2 ràng buộc chính) để dễ gán nhãn gold.
    # Các case quá chặt (nhiều ràng buộc) vẫn hữu ích để test filter/constraint, nhưng gold có thể rỗng.

    cases: List[TestCaseSpec] = []

    # --- Dresses ---
    cases += [
        TestCaseSpec(
            name="women_red_dress_basic",
            query="women red dress",
            expected_constraints={"gender": "Women", "color": "Red", "articleType": "Dresses"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_dress_formal",
            query="women black formal dress",
            expected_constraints={"gender": "Women", "color": "Black", "articleType": "Dresses", "usage": "Formal"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_blue_dress_party",
            query="women blue party dress",
            expected_constraints={"gender": "Women", "color": "Blue", "articleType": "Dresses", "usage": "Party"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_green_dress",
            query="women green dress",
            expected_constraints={"gender": "Women", "color": "Green", "articleType": "Dresses"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_pink_dress",
            query="women pink dress",
            expected_constraints={"gender": "Women", "color": "Pink", "articleType": "Dresses"},
            top_k=top_k,
        ),
    ]

    # --- Tshirts / Tops ---
    cases += [
        TestCaseSpec(
            name="men_white_tshirt",
            query="men white t-shirt",
            expected_constraints={"gender": "Men", "color": "White", "articleType": "Tshirts"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_black_tshirt",
            query="men black t-shirt",
            expected_constraints={"gender": "Men", "color": "Black", "articleType": "Tshirts"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_tshirt",
            query="women black t-shirt",
            expected_constraints={"gender": "Women", "color": "Black", "articleType": "Tshirts"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_white_topwear",
            query="women white top",
            expected_constraints={"gender": "Women", "color": "White", "subcategory": "Topwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_blue_topwear",
            query="men blue top",
            expected_constraints={"gender": "Men", "color": "Blue", "subcategory": "Topwear"},
            top_k=top_k,
        ),
    ]

    # --- Shirts ---
    cases += [
        TestCaseSpec(
            name="men_blue_shirt_casual",
            query="men blue casual shirt",
            expected_constraints={"gender": "Men", "color": "Blue", "articleType": "Shirts", "usage": "Casual"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_white_shirt_formal",
            query="men white formal shirt",
            expected_constraints={"gender": "Men", "color": "White", "articleType": "Shirts", "usage": "Formal"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_white_shirt",
            query="women white shirt",
            expected_constraints={"gender": "Women", "color": "White", "articleType": "Shirts"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_shirt_formal",
            query="women black formal shirt",
            expected_constraints={"gender": "Women", "color": "Black", "articleType": "Shirts", "usage": "Formal"},
            top_k=top_k,
        ),
    ]

    # --- Jeans / Trousers ---
    cases += [
        TestCaseSpec(
            name="men_blue_jeans",
            query="men blue jeans",
            expected_constraints={"gender": "Men", "color": "Blue", "articleType": "Jeans"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_jeans",
            query="women black jeans",
            expected_constraints={"gender": "Women", "color": "Black", "articleType": "Jeans"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_blue_jeans",
            query="women blue jeans",
            expected_constraints={"gender": "Women", "color": "Blue", "articleType": "Jeans"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_grey_trousers_formal",
            query="men grey formal trousers",
            expected_constraints={"gender": "Men", "color": "Grey", "usage": "Formal"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_trousers_formal",
            query="women black formal trousers",
            expected_constraints={"gender": "Women", "color": "Black", "usage": "Formal"},
            top_k=top_k,
        ),
    ]

    # --- Footwear / Shoes ---
    cases += [
        TestCaseSpec(
            name="men_black_shoes_budget",
            query="men black shoes under $80",
            expected_constraints={"gender": "Men", "color": "Black", "max_price": 80, "category": "Footwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_white_shoes",
            query="men white shoes",
            expected_constraints={"gender": "Men", "color": "White", "category": "Footwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_white_shoes",
            query="women white shoes",
            expected_constraints={"gender": "Women", "color": "White", "category": "Footwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_black_heels",
            query="women black heels",
            expected_constraints={"gender": "Women", "color": "Black", "category": "Footwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_pink_shoes",
            query="women pink shoes",
            expected_constraints={"gender": "Women", "color": "Pink", "category": "Footwear"},
            top_k=top_k,
        ),
    ]

    # --- Sports / Gym ---
    cases += [
        TestCaseSpec(
            name="men_sports_tshirt",
            query="men sports t-shirt",
            expected_constraints={"gender": "Men", "usage": "Sports", "articleType": "Tshirts"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_sports_shoes",
            query="women sports shoes",
            expected_constraints={"gender": "Women", "usage": "Sports", "category": "Footwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_sports_shoes",
            query="men sports shoes",
            expected_constraints={"gender": "Men", "usage": "Sports", "category": "Footwear"},
            top_k=top_k,
        ),
    ]

    # --- Accessories ---
    cases += [
        TestCaseSpec(
            name="women_black_handbag",
            query="women black handbag",
            expected_constraints={"gender": "Women", "color": "Black", "subcategory": "Bags"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_wallet",
            query="men wallet",
            expected_constraints={"gender": "Men", "subcategory": "Wallets"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_sunglasses",
            query="women sunglasses",
            expected_constraints={"gender": "Women", "subcategory": "Sunglasses"},
            top_k=top_k,
        ),
    ]

    # --- Watches / Jewelry ---
    cases += [
        TestCaseSpec(
            name="men_watches",
            query="men watch",
            expected_constraints={"gender": "Men", "subcategory": "Watches"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_watches",
            query="women watch",
            expected_constraints={"gender": "Women", "subcategory": "Watches"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_jewellery",
            query="women jewellery",
            expected_constraints={"gender": "Women", "category": "Accessories"},
            top_k=top_k,
        ),
    ]

    # --- Outerwear ---
    cases += [
        TestCaseSpec(
            name="men_jacket_black",
            query="men black jacket",
            expected_constraints={"gender": "Men", "color": "Black", "subcategory": "Topwear"},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_jacket_black",
            query="women black jacket",
            expected_constraints={"gender": "Women", "color": "Black", "subcategory": "Topwear"},
            top_k=top_k,
        ),
    ]

    # --- Budget-centric (giữ ràng buộc nhẹ) ---
    cases += [
        TestCaseSpec(
            name="women_dress_under_50",
            query="women dress under $50",
            expected_constraints={"gender": "Women", "articleType": "Dresses", "max_price": 50},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="men_shoes_under_60",
            query="men shoes under $60",
            expected_constraints={"gender": "Men", "category": "Footwear", "max_price": 60},
            top_k=top_k,
        ),
        TestCaseSpec(
            name="women_top_under_30",
            query="women top under $30",
            expected_constraints={"gender": "Women", "subcategory": "Topwear", "max_price": 30},
            top_k=top_k,
        ),
    ]

    # --- Vietnamese rejection (English-only) ---
    cases += [
        TestCaseSpec(
            name="reject_vietnamese_1",
            query="Tôi muốn mua áo sơ mi trắng đi làm dưới 40 đô",
            should_reject=True,
            top_k=top_k,
        ),
        TestCaseSpec(
            name="reject_vietnamese_2",
            query="Gợi ý giày thể thao màu đen cho nam dưới 80 đô",
            should_reject=True,
            top_k=top_k,
        ),
    ]

    # Khử trùng lặp name (phòng khi sửa tay)
    seen = set()
    unique: List[TestCaseSpec] = []
    for c in cases:
        if c.name in seen:
            continue
        seen.add(c.name)
        unique.append(c)

    return unique


def _filter_df(df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    out = df

    # gender
    g = constraints.get("gender")
    if g:
        out = out[out["gender"].fillna("").str.lower() == str(g).lower()]

    # color
    c = constraints.get("color")
    if c:
        want = _norm_color(c)
        out = out[_norm_color_series(out["baseColour"]) == want]

    # usage
    u = constraints.get("usage")
    if u:
        out = out[out["usage"].fillna("").str.lower() == str(u).lower()]

    # articleType
    at = constraints.get("articleType")
    if at:
        want = _norm_article_type(at)
        out = out[_norm_article_type_series(out["articleType"]) == want]

    # category/subcategory
    cat = constraints.get("category")
    if cat:
        out = out[out["masterCategory"].fillna("").str.lower() == str(cat).lower()]

    sub = constraints.get("subcategory")
    if sub:
        out = out[out["subCategory"].fillna("").str.lower() == str(sub).lower()]

    # budget
    min_p = constraints.get("min_price")
    max_p = constraints.get("max_price")
    if (min_p is not None) or (max_p is not None):
        price = pd.to_numeric(out.get("price"), errors="coerce")
        if min_p is not None:
            price_mask = price >= float(min_p)
            out = out[price_mask.fillna(False)]
        if max_p is not None:
            price_mask = price <= float(max_p)
            out = out[price_mask.fillna(False)]

    return out


def _norm_color_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(_norm_color)


def _norm_article_type_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(_norm_article_type)


def _rank_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q_toks = set(_tokenize(query))
    if not q_toks:
        return df

    def score_row(r: pd.Series) -> int:
        text = " ".join(
            [
                _norm_str(r.get("productDisplayName")),
                _norm_str(r.get("articleType")),
                _norm_str(r.get("masterCategory")),
                _norm_str(r.get("subCategory")),
                _norm_str(r.get("baseColour")),
                _norm_str(r.get("usage")),
                _norm_str(r.get("gender")),
            ]
        )
        dt = set(_tokenize(text))
        return int(len(q_toks & dt))

    scores = df.apply(score_row, axis=1)
    out = df.copy()
    out["_score"] = scores
    out = out.sort_values(["_score"], ascending=False)
    return out


def suggest_gold_ids(df: pd.DataFrame, query: str, constraints: Optional[Dict[str, Any]], k: int) -> List[str]:
    constraints = constraints or {}
    # Nếu should_reject thì không có gold.
    if constraints.get("_should_reject"):
        return []

    def _try(where: Dict[str, Any]) -> pd.DataFrame:
        return _filter_df(df, where)

    cand = _try(constraints)

    # Nếu lọc quá chặt dẫn đến rỗng, nới theo nhiều bước (budget -> usage -> color -> articleType -> gender).
    if len(cand) == 0 and constraints:
        steps: List[List[str]] = [
            ["min_price", "max_price"],
            ["usage"],
            ["color"],
            ["articleType", "subcategory", "category"],
            ["gender"],
        ]
        relaxed = dict(constraints)
        for keys in steps:
            for kk in keys:
                relaxed.pop(kk, None)
            cand = _try(relaxed)
            if len(cand) > 0:
                break

    if len(cand) == 0:
        # Cuối cùng: fallback về toàn bộ df để luôn có 1 vài gold IDs (dù kém chính xác).
        cand = df

    ranked = _rank_by_query(cand, query)
    ids = []
    for v in ranked["id"].tolist():
        if v is None:
            continue
        s = str(v)
        if s not in ids:
            ids.append(s)
        if len(ids) >= k:
            break
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datasets/archive/fashion-dataset/styles.csv")
    ap.add_argument("--out", default="eval/testset_gold.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--gold-per-case", type=int, default=3)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV không tồn tại: {csv_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")

    # Chuẩn hoá các cột cần thiết (nếu thiếu thì tạo cột rỗng)
    for col in [
        "id",
        "productDisplayName",
        "articleType",
        "masterCategory",
        "subCategory",
        "baseColour",
        "season",
        "usage",
        "gender",
        "price",
    ]:
        if col not in df.columns:
            df[col] = ""

    # Ép id thành string ổn định
    df["id"] = df["id"].astype(str)

    cases = _build_cases(top_k=int(args.topk))

    out_cases: List[Dict[str, Any]] = []
    for c in cases:
        item: Dict[str, Any] = asdict(c)
        if c.should_reject:
            item["expected_ids"] = []
        else:
            item["expected_ids"] = suggest_gold_ids(
                df,
                query=c.query,
                constraints=c.expected_constraints,
                k=int(args.gold_per_case),
            )
        out_cases.append(item)

    out_obj = {
        "version": 1,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "csv": str(csv_path.as_posix()),
        "top_k": int(args.topk),
        "gold_per_case": int(args.gold_per_case),
        "n_cases": len(out_cases),
        "cases": out_cases,
        "notes": "Gold IDs được gợi ý tự động; khuyến nghị rà soát thủ công để tăng độ chính xác.",
    }

    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved testset: {out_path} (cases={len(out_cases)})")


if __name__ == "__main__":
    main()
