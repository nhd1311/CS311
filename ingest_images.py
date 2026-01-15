"""
Ingest ảnh vào collection products_image.
Yêu cầu: open-clip-torch, torch, Pillow, requests.
Chạy ví dụ:
python ingest_images.py --csv datasets/archive/fashion-dataset/styles.csv --api http://localhost:8080/ingest_image --batch 128
"""

import argparse
import base64
import math
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
import requests

DEFAULT_CSV = "datasets/archive/fashion-dataset/styles.csv"
DEFAULT_API = os.getenv("INGEST_IMAGE_API", "http://localhost:8080/ingest_image")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _to_posix_relpath(path: str) -> str:
    rel = os.path.relpath(path, start=_REPO_ROOT)
    return rel.replace("\\", "/")


def _guess_image_dir(csv_path: str) -> str:
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    candidates = [
        os.path.join(csv_dir, "images"),
        os.path.join(os.path.dirname(csv_dir), "images"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    # Mặc định dùng csv_dir/images kể cả khi chưa tồn tại (giữ hành vi dễ đoán).
    return candidates[0]


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
        "name": clean_val(r.get("productDisplayName")),
        "category": clean_val(r.get("masterCategory")),
        "subcategory": clean_val(r.get("subCategory")),
        "articleType": clean_val(r.get("articleType")),
        "gender": clean_val(r.get("gender")),
        "color": clean_val(r.get("baseColour")),
        "season": clean_val(r.get("season")),
        "year": None if pd.isna(r.get("year")) else r.get("year"),
        "usage": clean_val(r.get("usage")),
        "image_url": img_url,
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    return meta


def ingest_batch(api: str, batch, *, timeout_s: int = 900, retries: int = 3):
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            resp = requests.post(api, json={"items": batch}, timeout=int(timeout_s))
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            print(f"[ERROR] HTTP {resp.status_code}")
            print(f"[ERROR] Response: {resp.text}")
            print(f"[ERROR] First item in batch: {batch[0] if batch else 'empty'}")
            raise
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
            sleep_s = min(30, 2 ** attempt)
            print(f"[WARN] Timeout when calling {api} (attempt {attempt + 1}/{retries}). Sleeping {sleep_s}s then retry...")
            time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    # Vị trí dataset mặc định mới.
    parser.add_argument("--csv", default=os.getenv("DATASET_CSV", "datasets/archive/fashion-dataset/styles.csv"))
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--image-column", default="image_url")
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("INGEST_IMAGE_TIMEOUT", "900")),
        help="Requests timeout in seconds for each batch (default: 900).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("INGEST_IMAGE_RETRIES", "3")),
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

    img_dir = _guess_image_dir(args.csv)

    ingested = 0

    for i in range(0, total, args.batch):
        chunk = rows[i : i + args.batch]
        payload_items = []
        for j, r in enumerate(chunk):
            global_idx = args.start + i + j
            item_id = to_str_safe(r.get("id"), f"row-{global_idx}")
            # Tạo đường dẫn ảnh tuyệt đối từ id
            img_path = os.path.join(img_dir, f"{item_id}.jpg")
            
            # Kiểm tra xem file ảnh có tồn tại không
            if not os.path.exists(img_path):
                continue
            
            # Đọc ảnh và mã hóa thành base64
            try:
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print(f"[WARN] Không thể đọc ảnh {img_path}: {e}")
                continue
                
            payload_items.append(
                {
                    "id": item_id,
                    "image_url": f"data:image/jpeg;base64,{img_base64}",
                    # Lưu đường dẫn tương đối theo repo để hiển thị/debug.
                    "metadata": build_metadata(r, _to_posix_relpath(img_path)),
                }
            )

        if not payload_items:
            print(f"[WARN] Batch {i} trống ảnh, bỏ qua")
            continue

        ingest_batch(args.api, payload_items, timeout_s=args.timeout, retries=args.retries)
        ingested += len(payload_items)
        print(f"[INFO] Ingested images: {ingested} (from {len(payload_items)} in this batch)")

    print("[DONE] Ingest images completed.")


if __name__ == "__main__":
    main()
