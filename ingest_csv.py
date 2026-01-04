"""ingest_csv.py

CLI để ingest dữ liệu CSV vào API RAG.

Gợi ý: dùng batch nhỏ (32–64) để tránh API bị quá tải/đứt kết nối khi embed.
Ví dụ:
    python ingest_csv.py --csv datasets/fashion_dataset_normalized.csv --api http://localhost:8080/ingest --batch 32
"""

import argparse
import math
import os
import sys
from typing import Any, Dict

import pandas as pd
import requests

DEFAULT_CSV = "datasets/fashion_dataset_normalized.csv"
DEFAULT_API = os.getenv("INGEST_API", "http://localhost:8080/ingest")


def build_text(row: Dict[str, Any]) -> str:
    parts = [
        row.get("name") or "",
        row.get("description") or "",
        row.get("category") or "",
        row.get("subcategory") or "",
        row.get("color") or "",
        str(row.get("price")) if not pd.isna(row.get("price")) else "",
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


def build_metadata(r: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "name": clean_val(r.get("name")),
        "category": clean_val(r.get("category")),
        "subcategory": clean_val(r.get("subcategory")),
        "gender": clean_val(r.get("gender")),
        "color": clean_val(r.get("color")),
        "price": None if pd.isna(r.get("price")) else r.get("price"),
        "image_url": clean_val(r.get("image_url")),
    }
    # Remove None values because Chroma metadata does not accept None
    meta = {k: v for k, v in meta.items() if v is not None}
    return meta


def ingest_batch(api: str, batch):
    payload = {"items": batch}
    resp = requests.post(api, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV không tồn tại: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    rows = df.to_dict(orient="records")
    total = len(rows)
    print(f"[INFO] Rows: {total}")

    for i in range(0, total, args.batch):
        chunk = rows[i : i + args.batch]
        payload_items = []
        for j, r in enumerate(chunk):
            item_id = to_str_safe(r.get("id"), f"row-{i+j}")
            payload_items.append(
                {
                    "id": item_id,
                    "text": build_text(r),
                    "metadata": build_metadata(r),
                }
            )

        ingest_batch(args.api, payload_items)
        print(f"[INFO] Ingested {i + len(chunk)} / {total}")

    print("[DONE] Ingest completed.")


if __name__ == "__main__":
    main()
