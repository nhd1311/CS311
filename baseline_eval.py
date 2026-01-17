from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass
class TestCase:
    name: str
    query: str
    expected_ids: Optional[List[str]] = None
    expected_constraints: Optional[Dict[str, Any]] = None
    should_reject: bool = False
    top_k: int = 5


def load_testset_json(path: Path, default_top_k: int) -> List[TestCase]:
    """Load testset từ JSON.

    Format hỗ trợ:
      - {"version": 1, "cases": [ {"name":..., "query":..., "expected_ids": [...], ...}, ... ]}
      - Hoặc trực tiếp là một list các object case.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    cases_obj = obj.get("cases") if isinstance(obj, dict) else obj
    if not isinstance(cases_obj, list):
        raise ValueError("Testset JSON không hợp lệ: thiếu 'cases' dạng list")

    out: List[TestCase] = []
    for i, c in enumerate(cases_obj):
        if not isinstance(c, dict):
            raise ValueError(f"Case #{i} không hợp lệ (không phải object)")
        name = str(c.get("name") or f"case_{i}")
        query = str(c.get("query") or "").strip()
        if not query:
            raise ValueError(f"Case '{name}' thiếu query")

        expected_ids = c.get("expected_ids")
        if expected_ids is None:
            expected_ids_list = None
        elif isinstance(expected_ids, list):
            expected_ids_list = [str(x) for x in expected_ids if x is not None and str(x).strip()]
        else:
            raise ValueError(f"Case '{name}' expected_ids phải là list hoặc null")

        expected_constraints = c.get("expected_constraints")
        if expected_constraints is not None and not isinstance(expected_constraints, dict):
            raise ValueError(f"Case '{name}' expected_constraints phải là object hoặc null")

        should_reject = bool(c.get("should_reject") or False)
        top_k = int(c.get("top_k") or default_top_k)

        out.append(
            TestCase(
                name=name,
                query=query,
                expected_ids=expected_ids_list,
                expected_constraints=expected_constraints,
                should_reject=should_reject,
                top_k=top_k,
            )
        )
    return out


def default_testset(top_k: int) -> List[TestCase]:
    # LƯU Ý: /chat chỉ hỗ trợ tiếng Anh; thêm một case tiếng Việt để kiểm thử hành vi từ chối.
    return [
        TestCase(
            name="basic_men_shoes_budget",
            query="men black sneakers under $80",
            expected_constraints={"gender": "Men", "color": "Black", "max_price": 80},
            top_k=top_k,
        ),
        TestCase(
            name="formal_office",
            query="women formal office outfit under $100",
            expected_constraints={"usage": "Formal", "max_price": 100},
            top_k=top_k,
        ),
        TestCase(
            name="color_strict",
            query="women red dress",
            expected_constraints={"color": "Red"},
            top_k=top_k,
        ),
        TestCase(
            name="type_tshirts",
            query="men white t-shirt",
            expected_constraints={"articleType": "Tshirts", "color": "White", "gender": "Men"},
            top_k=top_k,
        ),
        TestCase(
            name="reject_vietnamese",
            query="Tôi muốn mua áo sơ mi trắng đi làm dưới 40 đô",
            should_reject=True,
            top_k=top_k,
        ),
    ]


def _hash_payload(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _request_json(
    method: str,
    url: str,
    payload: Optional[dict],
    timeout_s: float,
    max_retries: int,
) -> Tuple[int, dict, float, Optional[str]]:
    """Trả về (status_code, json_or_raw, latency_ms, error_text)."""
    last_err = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.request(method=method, url=url, json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000
            try:
                data = resp.json()
            except Exception:
                data = {"_raw": resp.text}

            if 200 <= resp.status_code < 300:
                return resp.status_code, data, latency_ms, None

            last_err = f"HTTP {resp.status_code}: {data}"
            if resp.status_code >= 500 and attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return resp.status_code, data, latency_ms, last_err
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return 0, {}, latency_ms, last_err


def _health_ok(api_base: str, timeout_s: float = 5.0) -> bool:
    api_base = (api_base or "").rstrip("/")
    try:
        r = requests.get(f"{api_base}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def call_chat(
    api_base: str,
    query: str,
    top_k: int,
    timeout_s: float,
    max_retries: int,
    cache: Optional[dict] = None,
) -> dict:
    api_base = (api_base or "").rstrip("/")

    payload: Dict[str, Any] = {
        "query": query,
        "top_k": int(top_k),
        # Tránh 422: schema API thường mong đợi filters là {} thay vì null
        "filters": {},
    }

    key = _hash_payload(payload)
    if cache is not None and key in cache:
        out = dict(cache[key])
        out["_cached"] = True
        return out

    status, data, latency_ms, err = _request_json("POST", f"{api_base}/chat", payload, timeout_s, max_retries)
    out = {"status": status, "latency_ms": latency_ms, "error": err, "response": data, "_cached": False}

    if cache is not None:
        cache[key] = out

    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _normalize_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize_color(x: Any) -> str:
    s = _normalize_str(x).lower()
    s = re.sub(r"[^a-z]+", "", s)
    if s == "grey":
        return "gray"
    return s


def _normalize_article_type(x: Any) -> str:
    s = _normalize_str(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _extract_prices(products: list) -> List[float]:
    out: List[float] = []
    for p in products or []:
        if not isinstance(p, dict):
            continue
        v = _safe_float(p.get("price"))
        if v is None:
            continue
        if v >= 0:
            out.append(v)
    return out


def _ids_from_products(products: list) -> List[str]:
    out: List[str] = []
    for p in products or []:
        if isinstance(p, dict) and p.get("id") is not None:
            out.append(str(p.get("id")))
    return out


def _ids_from_sources(sources: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(sources, list):
        return out
    for s in sources:
        if isinstance(s, dict) and s.get("id") is not None:
            out.append(str(s.get("id")))
    return out


def english_only_rejected(answer: str, products: list, response_obj: dict) -> bool:
    a = (answer or "").lower()
    if "english only" in a:
        return True
    if isinstance(response_obj, dict) and "error" in response_obj:
        if "english" in str(response_obj.get("error") or "").lower():
            return True
    # Heuristic (ước lượng): không có products và câu trả lời gợi ý viết lại
    if (not products) and ("rewrite" in a or "rephrase" in a) and ("understand" in a or "english" in a):
        return True
    # Với /chat trong project này, trường hợp bị từ chối (English-only) thường trả về:
    # "I don't understand your question. Please rewrite it." và products = []
    if (not products) and ("don't understand" in a or "do not understand" in a) and ("rewrite" in a or "rephrase" in a):
        return True
    return False


def constraint_checks(products: list, constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Kiểm tra xem `products` trả về có thoả các ràng buộc kỳ vọng hay không.

    Đánh giá trên `products` vì UI và câu trả lời (answer) trong hệ hiện tại
    được suy ra trực tiếp từ danh sách này.
    """

    constraints = constraints or {}
    out: Dict[str, Any] = {}

    if not products:
        out["has_products"] = False
        out["constraint_pass"] = False if constraints else True
        return out

    out["has_products"] = True

    # --- Màu sắc ---
    if "color" in constraints and constraints["color"] is not None:
        want = _normalize_color(constraints["color"])
        colors = [_normalize_color((p or {}).get("color")) for p in products if isinstance(p, dict)]
        out["color_all_match"] = all(c == want and c for c in colors)
    else:
        out["color_all_match"] = None

    # --- Dịp sử dụng (usage) ---
    if "usage" in constraints and constraints["usage"] is not None:
        want = _normalize_str(constraints["usage"]).lower()
        usages = [_normalize_str((p or {}).get("usage")).lower() for p in products if isinstance(p, dict)]
        out["usage_all_match"] = all(u == want and u for u in usages)
    else:
        out["usage_all_match"] = None

    # --- Giới tính (gender) ---
    if "gender" in constraints and constraints["gender"] is not None:
        want = _normalize_str(constraints["gender"]).lower()
        genders = [_normalize_str((p or {}).get("gender")).lower() for p in products if isinstance(p, dict)]
        out["gender_all_match"] = all(g == want and g for g in genders)
    else:
        out["gender_all_match"] = None

    # --- Loại sản phẩm (article type) ---
    if "articleType" in constraints and constraints["articleType"] is not None:
        want = _normalize_article_type(constraints["articleType"])
        ats = [
            _normalize_article_type(
                (p or {}).get("subcategory") or (p or {}).get("category") or (p or {}).get("articleType")
            )
            for p in products
            if isinstance(p, dict)
        ]
        # Nếu API không trả articleType một cách đáng tin cậy, bỏ qua kiểm tra strict.
        out["type_all_match"] = None if all(a == "" for a in ats) else all(a == want for a in ats if a)
    else:
        out["type_all_match"] = None

    # --- Ngân sách ---
    prices = _extract_prices(products)
    min_p = _safe_float(constraints.get("min_price"))
    max_p = _safe_float(constraints.get("max_price"))
    if min_p is not None or max_p is not None:
        if not prices:
            out["budget_all_match"] = False
        else:
            ok = True
            for p in prices:
                if min_p is not None and p < min_p - 1e-9:
                    ok = False
                if max_p is not None and p > max_p + 1e-9:
                    ok = False
            out["budget_all_match"] = ok
    else:
        out["budget_all_match"] = None

    bool_checks = [
        v
        for v in [
            out.get("color_all_match"),
            out.get("usage_all_match"),
            out.get("gender_all_match"),
            out.get("budget_all_match"),
            out.get("type_all_match"),
        ]
        if isinstance(v, bool)
    ]
    out["constraint_pass"] = all(bool_checks) if bool_checks else True
    return out


def faithfulness_heuristic(answer: str, products: list) -> Dict[str, Any]:
    ans = answer or ""
    pids = set(_ids_from_products(products))

    mentioned = set(re.findall(r"\((\d+)\)", ans))
    extra_ids = sorted([x for x in mentioned if x not in pids])

    money = [float(x) for x in re.findall(r"\$(\d+(?:\.\d{1,2})?)", ans)]
    prices = set(round(p, 2) for p in _extract_prices(products))

    money_ok = True
    if money and not prices:
        money_ok = False
    else:
        for m in money:
            if round(m, 2) not in prices:
                if round(m, 0) not in {round(p, 0) for p in prices}:
                    money_ok = False

    return {
        "mentioned_ids": sorted(list(mentioned)),
        "extra_ids_not_in_products": extra_ids,
        "money_mentions": money,
        "money_matches_products": money_ok,
        "faithful": (len(extra_ids) == 0) and money_ok,
    }


def pii_safety_heuristic(text: str) -> Dict[str, Any]:
    s = text or ""
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
    phone = re.findall(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b", s)
    cc = re.findall(r"\b(?:\d[ -]*?){13,19}\b", s)

    cc_clean = []
    for c in cc:
        digits = re.sub(r"\D", "", c)
        if len(digits) >= 13:
            cc_clean.append(digits)

    return {
        "email_found": len(email) > 0,
        "phone_found": len(phone) > 0,
        "cc_like_found": len(cc_clean) > 0,
        "pii_violation": (len(email) > 0) or (len(cc_clean) > 0),
    }


def _extract_pick_lines(answer: str) -> List[str]:
    lines = (answer or "").splitlines()
    picks = []
    for ln in lines:
        if re.match(r"^\s*\d+\)\s+", ln):
            picks.append(ln.strip())
    return picks


def rubric_score_rule_based(
    answer: str,
    products: list,
    expected_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rubric 0..10: faithfulness/format/completeness/constraint/conciseness, mỗi mục 0..2."""

    ans = answer or ""

    faith = faithfulness_heuristic(ans, products)
    faithfulness = 2 if faith.get("faithful") else 0

    picks = _extract_pick_lines(ans)
    has_top_picks_header = bool(re.search(r"(?im)^\s*top\s+picks\s*:\s*$", ans))
    if has_top_picks_header and len(picks) >= 2:
        format_score = 2
    elif len(picks) >= 1:
        format_score = 1
    else:
        format_score = 0

    mentioned_ids = set(re.findall(r"\((\d+)\)", ans))
    prod_ids = set(_ids_from_products(products))
    mentioned_valid = [x for x in mentioned_ids if x in prod_ids]
    if 2 <= len(picks) <= 4 and len(mentioned_valid) >= min(2, len(prod_ids)):
        completeness = 2
    elif len(picks) >= 1:
        completeness = 1
    else:
        completeness = 0

    c = constraint_checks(products, expected_constraints)
    constraint_support = 2 if c.get("constraint_pass") else (1 if c.get("has_products") else 0)

    n_chars = len(ans.strip())
    if n_chars == 0:
        conciseness = 0
    elif n_chars <= 1200:
        conciseness = 2
    else:
        conciseness = 1

    total = faithfulness + format_score + completeness + constraint_support + conciseness

    return {
        "rubric_faithfulness": faithfulness,
        "rubric_format": format_score,
        "rubric_completeness": completeness,
        "rubric_constraint": constraint_support,
        "rubric_conciseness": conciseness,
        "rubric_total": total,
        "rubric_pick_count": len(picks),
        "rubric_answer_chars": n_chars,
    }


def _ndcg_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    g = set(gold_ids)

    def dcg(ids: List[str]) -> float:
        s = 0.0
        for j, pid in enumerate(ids[:k], start=1):
            rel = 1.0 if pid in g else 0.0
            s += rel / math.log2(j + 1)
        return s

    ideal = [pid for pid in gold_ids][:k]
    if len(ideal) < k:
        ideal = ideal + ["__non__"] * (k - len(ideal))
    denom = dcg(ideal)
    if denom <= 0:
        return 0.0
    return dcg(pred_ids) / denom


def _hit_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    topk = pred_ids[:k]
    return 1.0 if any(pid in set(gold_ids) for pid in topk) else 0.0


def _mrr(pred_ids: List[str], gold_ids: List[str]) -> float:
    if not gold_ids:
        return float("nan")
    g = set(gold_ids)
    for i, pid in enumerate(pred_ids, start=1):
        if pid in g:
            return 1.0 / i
    return 0.0


def run_baseline_eval(
    api_base: str,
    testset: List[TestCase],
    timeout_s: float,
    max_retries: int,
    cache_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache: dict
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except Exception:
        cache = {}

    rows: List[dict] = []
    for tc in testset:
        res = call_chat(
            api_base=api_base,
            query=tc.query,
            top_k=tc.top_k,
            timeout_s=timeout_s,
            max_retries=max_retries,
            cache=cache,
        )
        resp = res.get("response") or {}
        products = resp.get("products") or []
        sources = resp.get("sources") or []
        answer = resp.get("answer") or ""

        c = constraint_checks(products, tc.expected_constraints)
        faith = faithfulness_heuristic(answer, products)
        pii = pii_safety_heuristic(answer)
        rejected = english_only_rejected(answer, products, resp)
        rubric = rubric_score_rule_based(answer=answer, products=products, expected_constraints=tc.expected_constraints)

        gold_ids = tc.expected_ids or []
        pred_ids = _ids_from_sources(sources) or _ids_from_products(products)
        # Retrieval metrics chỉ có ý nghĩa khi có gold IDs (và không phải case chủ đích bị reject).
        has_gold = bool(gold_ids) and (not tc.should_reject)
        hit_1 = _hit_at_k(pred_ids, gold_ids, 1) if has_gold else float("nan")
        hit_3 = _hit_at_k(pred_ids, gold_ids, 3) if has_gold else float("nan")
        hit_5 = _hit_at_k(pred_ids, gold_ids, 5) if has_gold else float("nan")
        mrr = _mrr(pred_ids, gold_ids) if has_gold else float("nan")
        ndcg_3 = _ndcg_at_k(pred_ids, gold_ids, 3) if has_gold else float("nan")
        ndcg_5 = _ndcg_at_k(pred_ids, gold_ids, 5) if has_gold else float("nan")

        row = {
            "name": tc.name,
            "query": tc.query,
            "should_reject": bool(tc.should_reject),
            "expected_constraints": tc.expected_constraints,
            "gold_ids": gold_ids,
            "pred_ids": pred_ids,
            "status": int(res.get("status") or 0),
            "latency_ms": float(res.get("latency_ms") or 0.0),
            "error": res.get("error"),
            "cached": bool(res.get("_cached")),
            "answer": answer,
            "n_products": len(products) if isinstance(products, list) else 0,
            "rejected_english_only": bool(rejected),
            "ret_hit@1": hit_1,
            "ret_hit@3": hit_3,
            "ret_hit@5": hit_5,
            "ret_mrr": mrr,
            "ret_ndcg@3": ndcg_3,
            "ret_ndcg@5": ndcg_5,
            **{f"c_{k}": v for k, v in c.items()},
            **{f"faith_{k}": v for k, v in faith.items()},
            **{f"pii_{k}": v for k, v in pii.items()},
            **rubric,
        }
        rows.append(row)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)

    # Tổng hợp
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": api_base,
        "n_cases": int(len(df)),
        "health_ok": bool(_health_ok(api_base)),
    }

    if len(df):
        status_ok = df["status"].fillna(0).astype(int) == 200
        summary["error_rate"] = float((~status_ok).mean())

        lat = df["latency_ms"].astype(float)
        summary["latency_p50_ms"] = float(lat.quantile(0.50))
        summary["latency_p90_ms"] = float(lat.quantile(0.90))
        summary["latency_p95_ms"] = float(lat.quantile(0.95))
        summary["latency_mean_ms"] = float(lat.mean())

        # Tỉ lệ products rỗng (tách reject vs non-reject)
        empty = df["n_products"].fillna(0).astype(int) == 0
        summary["empty_products_rate_all"] = float(empty.mean())
        non_reject = ~df["should_reject"].astype(bool)
        if non_reject.any():
            summary["empty_products_rate_non_reject"] = float(empty[non_reject].mean())

        # Chỉ tiếng Anh (English-only)
        got_reject = df["rejected_english_only"].astype(bool)
        want_reject = df["should_reject"].astype(bool)
        if want_reject.any():
            summary["english_only_reject_accuracy"] = float(got_reject[want_reject].mean())
        if non_reject.any():
            summary["english_only_false_reject_rate"] = float(got_reject[non_reject].mean())

        # An toàn
        pii_viol = df.get("pii_pii_violation")
        if pii_viol is not None:
            summary["pii_violation_rate"] = float(pii_viol.fillna(False).astype(bool).mean())

        # Ràng buộc
        cpass = df.get("c_constraint_pass")
        if cpass is not None:
            # Chỉ tính các dòng thực sự có ràng buộc
            has_constraints = df["expected_constraints"].apply(lambda x: isinstance(x, dict) and len(x) > 0)
            if has_constraints.any():
                summary["constraint_pass_rate"] = float(cpass[has_constraints].fillna(False).astype(bool).mean())

        # Tính trung thực (faithfulness)
        fpass = df.get("faith_faithful")
        if fpass is not None:
            summary["faithfulness_pass_rate"] = float(fpass.fillna(False).astype(bool).mean())

        # Rubric
        if "rubric_total" in df.columns:
            summary["rubric_total_mean"] = float(df["rubric_total"].astype(float).mean())

        # Retrieval metrics (Hit@K / MRR / nDCG) - chỉ tính trên các case có gold ids.
        if "gold_ids" in df.columns:
            has_gold_mask = df["gold_ids"].apply(lambda x: isinstance(x, list) and len(x) > 0) & (~df["should_reject"].astype(bool))
            n_gold = int(has_gold_mask.sum())
            summary["retrieval_labeled_cases"] = n_gold
            if n_gold > 0:
                for col in ["ret_hit@1", "ret_hit@3", "ret_hit@5", "ret_mrr", "ret_ndcg@3", "ret_ndcg@5"]:
                    if col in df.columns:
                        summary[col] = float(pd.to_numeric(df.loc[has_gold_mask, col], errors="coerce").mean())

    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline evaluation for /chat (stability, constraints, latency, English-only, safety)")
    ap.add_argument("--base", default=os.getenv("API_BASE", "http://127.0.0.1:8081"), help="API base URL")
    ap.add_argument("--topk", type=int, default=int(os.getenv("EVAL_TOP_K", "5")))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT_S", "30")))
    ap.add_argument("--retries", type=int, default=int(os.getenv("EVAL_MAX_RETRIES", "2")))

    ap.add_argument(
        "--testset",
        default=None,
        help="(Tuỳ chọn) Đường dẫn JSON testset có gold IDs để tính Hit@K/MRR/nDCG",
    )

    ap.add_argument("--outputs", default="outputs", help="Outputs directory")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory")

    args = ap.parse_args()

    api_base = (args.base or "").rstrip("/")
    outputs_dir = Path(args.outputs)
    artifacts_dir = Path(args.artifacts)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.testset:
        testset_path = Path(args.testset)
        if not testset_path.exists():
            raise SystemExit(f"Không tìm thấy testset: {testset_path}")
        testset = load_testset_json(testset_path, default_top_k=int(args.topk))
        testset_source = str(testset_path)
    else:
        testset = default_testset(args.topk)
        testset_source = "default_testset()"

    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": api_base,
        "timeout_s": float(args.timeout),
        "max_retries": int(args.retries),
        "top_k": int(args.topk),
        "n_cases": int(len(testset)),
        "testset": testset_source,
    }

    (artifacts_dir / "baseline_eval_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df, summary = run_baseline_eval(
        api_base=api_base,
        testset=testset,
        timeout_s=float(args.timeout),
        max_retries=int(args.retries),
        cache_path=artifacts_dir / "baseline_eval_cache.json",
    )

    out_csv = outputs_dir / "baseline_eval_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    (artifacts_dir / "baseline_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:")
    print("-", out_csv.resolve())
    print("-", (artifacts_dir / "baseline_eval_summary.json").resolve())


if __name__ == "__main__":
    main()
