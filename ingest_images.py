"""
Ingest ảnh vào collection products_image.
Yêu cầu: open-clip-torch, torch, Pillow, requests.
Chạy ví dụ:
python ingest_images.py --csv datasets/fashion_dataset_normalized.csv --api http://localhost:8080/ingest_image --batch 128 --image-column image_url
"""

import argparse
import math
import os
import sys
from typing import Any, Dict

import pandas as pd
import requests

DEFAULT_CSV = "datasets/fashion_dataset_normalized.csv"
DEFAULT_API = os.getenv("INGEST_IMAGE_API", "http://localhost:8080/ingest_image")


def to_str_safe(val: Any, fallback: str) -> str:
    if val is None:
        return fallback
    if isinstance(val, float) and math.isnan(val):
        return fallback
    return str(val)


def clean_val(val: Any):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def build_metadata(r: Dict[str, Any], img_url: str) -> Dict[str, Any]:
    meta = {
        "name": clean_val(r.get("name")),
        "category": clean_val(r.get("category")),
        "subcategory": clean_val(r.get("subcategory")),
        "gender": clean_val(r.get("gender")),
        "color": clean_val(r.get("color")),
        "price": None if pd.isna(r.get("price")) else r.get("price"),
        "image_url": img_url,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    return meta


def ingest_batch(api: str, batch):
    resp = requests.post(api, json={"items": batch}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--image-column", default="image_url")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV không tồn tại: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.image_column not in df.columns:
        print(f"Không thấy cột ảnh: {args.image_column}")
        sys.exit(1)

    rows = df.to_dict(orient="records")
    total = len(rows)
    print(f"[INFO] Rows: {total}")

    for i in range(0, total, args.batch):
        chunk = rows[i : i + args.batch]
        payload_items = []
        for j, r in enumerate(chunk):
            img_url = r.get(args.image_column)
            if not isinstance(img_url, str) or not img_url:
                continue
            item_id = to_str_safe(r.get("id"), f"row-{i+j}")
            payload_items.append(
                {
                    "id": item_id,
                    "image_url": img_url,
                    "metadata": build_metadata(r, img_url),
                }
            )

        if not payload_items:
            print(f"[WARN] Batch {i} trống ảnh, bỏ qua")
            continue

        ingest_batch(args.api, payload_items)
        print(f"[INFO] Ingested images {i + len(chunk)} / {total}")

    print("[DONE] Ingest images completed.")


if __name__ == "__main__":
    main()
