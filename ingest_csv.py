"""ingest_csv.py

CLI để ingest dữ liệu CSV vào API RAG.

Gợi ý: dùng batch nhỏ (32–64) để tránh API bị quá tải/đứt kết nối khi embed.
Ví dụ:
    python ingest_csv.py --csv datasets/archive/fashion-dataset/styles.csv --api http://localhost:8080/ingest --batch 32
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
import requests

DEFAULT_CSV = "datasets/archive/fashion-dataset/styles.csv"
DEFAULT_API = os.getenv("INGEST_API", "http://localhost:8080/ingest")

# Resolve paths relative to the repo root (this file lives at repo root).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _to_posix_relpath(path: str) -> str:
    rel = os.path.relpath(path, start=_REPO_ROOT)
    return rel.replace("\\", "/")


def _guess_image_path(csv_path: str, item_id: str) -> str | None:
    if not item_id:
        return None
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    candidates = [
        os.path.join(csv_dir, "images", f"{item_id}.jpg"),
        os.path.join(os.path.dirname(csv_dir), "images", f"{item_id}.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("productDisplayName") or "",
        row.get("articleType") or "",
        row.get("masterCategory") or "",
        row.get("subCategory") or "",
        row.get("baseColour") or "",
        row.get("season") or "",
        str(row.get("year")) if not pd.isna(row.get("year")) else "",
        row.get("usage") or "",
        row.get("gender") or "",
    ]
    return " ".join(map(str, parts))


def to_str_safe(val: Any, fallback: str) -> str:
    if val is None:
        return fallback
    if isinstance(val, float) and math.isnan(val):
        return fallback
    return str(val)


def clean_val(val: Any):
    """Convert NaN/None to None to make JSON-compliant."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def build_metadata(r: Dict[str, Any], csv_path: str) -> Dict[str, Any]:
    item_id = to_str_safe(r.get("id"), "")
    img_path = _guess_image_path(csv_path, item_id)
    image_url = _to_posix_relpath(img_path) if img_path else None
    
    meta = {
        "name": clean_val(r.get("productDisplayName")),
        "category": clean_val(r.get("masterCategory")),
        "subcategory": clean_val(r.get("subCategory")),
        "articleType": clean_val(r.get("articleType")),
        "gender": clean_val(r.get("gender")),
        "color": clean_val(r.get("baseColour")),
        "season": clean_val(r.get("season")),
        "usage": clean_val(r.get("usage")),
        "image_url": image_url,
        # Optional pricing fields if present in the CSV
        "price_range_usd": clean_val(r.get("price_range_usd")),
        "price": clean_val(r.get("price")),
    }
    # Remove None values because Chroma metadata does not accept None
    meta = {k: v for k, v in meta.items() if v is not None}
    return meta


def ingest_batch(api: str, batch, *, timeout_s: int = 600, retries: int = 3):
    """Post one ingest batch to the API.

    Embedding can be slow (CPU/first model load), so we allow a long timeout and
    a few retries to avoid client-side ReadTimeout aborting a valid long request.
    """

    payload = {"items": batch}
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            resp = requests.post(api, json=payload, timeout=int(timeout_s))
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
            # Exponential-ish backoff.
            sleep_s = min(30, 2 ** attempt)
            print(f"[WARN] Timeout when calling {api} (attempt {attempt + 1}/{retries}). Sleeping {sleep_s}s then retry...")
            time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    # New default dataset location.
    parser.add_argument("--csv", default=os.getenv("DATASET_CSV", "datasets/archive/fashion-dataset/styles.csv"))
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("INGEST_TIMEOUT", "600")),
        help="Requests timeout in seconds for each batch (default: 600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("INGEST_RETRIES", "3")),
        help="Number of retries on timeout per batch (default: 3).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Row offset to start ingesting from (0-based). Useful for resume.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of rows to ingest from --start. 0 = no limit.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV không tồn tại: {args.csv}")
        sys.exit(1)

    if args.start < 0:
        print("--start must be >= 0")
        sys.exit(1)
    if args.limit < 0:
        print("--limit must be >= 0")
        sys.exit(1)

    df = pd.read_csv(args.csv, on_bad_lines='skip', encoding='utf-8')
    all_rows = df.to_dict(orient="records")
    all_total = len(all_rows)

    if args.start > all_total:
        print(f"--start ({args.start}) is beyond total rows ({all_total})")
        sys.exit(1)

    end = all_total if args.limit == 0 else min(all_total, args.start + args.limit)
    rows = all_rows[args.start:end]
    total = len(rows)
    print(f"[INFO] Rows selected: {total} (from {args.start} to {end - 1}, total={all_total})")

    for i in range(0, total, args.batch):
        chunk = rows[i : i + args.batch]
        payload_items = []
        for j, r in enumerate(chunk):
            global_idx = args.start + i + j
            item_id = to_str_safe(r.get("id"), f"row-{global_idx}")
            payload_items.append(
                {
                    "id": item_id,
                    "text": build_text(r),
                    "metadata": build_metadata(r, args.csv),
                }
            )

        ingest_batch(args.api, payload_items, timeout_s=args.timeout, retries=args.retries)
        print(f"[INFO] Ingested {min(i + len(chunk), total)} / {total}")

    print("[DONE] Ingest completed.")


if __name__ == "__main__":
    main()
