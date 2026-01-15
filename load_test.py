import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


def _percentiles_ms(samples_ms: List[float], ps: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {f"p{int(p)}_ms": float("nan") for p in ps}
    arr = np.array(samples_ms, dtype=float)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{int(p)}_ms"] = float(np.percentile(arr, p))
    return out


def _worker_chat(base_url: str, query: str, top_k: int, timeout_s: float) -> Tuple[bool, float, Optional[int], Optional[str]]:
    url = base_url.rstrip("/") + "/chat"
    payload = {"query": query, "top_k": top_k}
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        dt = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200
        # Cố gắng bắt nhanh thông báo lỗi phía server
        err = None
        if not ok:
            err = r.text[:500]
        return ok, dt, r.status_code, err
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return False, dt, None, repr(e)


def _worker_image_upload(base_url: str, img_bytes: bytes, filename: str, top_k: int, timeout_s: float) -> Tuple[bool, float, Optional[int], Optional[str]]:
    url = base_url.rstrip("/") + "/search/image/upload"
    files = {"file": (filename, img_bytes, "image/jpeg")}
    params = {"top_k": top_k}
    t0 = time.perf_counter()
    try:
        r = requests.post(url, files=files, params=params, timeout=timeout_s)
        dt = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200
        err = None
        if not ok:
            err = r.text[:500]
        return ok, dt, r.status_code, err
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return False, dt, None, repr(e)


def run_load_test(
    kind: str,
    base_url: str,
    n: int,
    concurrency: int,
    timeout_s: float,
    chat_query: str,
    chat_top_k: int,
    image_path: Optional[str],
    image_top_k: int,
) -> Dict[str, Any]:
    ps = [50, 90, 95]

    # Kiểm tra health trước khi chạy
    health_url = base_url.rstrip("/") + "/health"
    try:
        h = requests.get(health_url, timeout=5)
        health_ok = h.status_code == 200
    except Exception:
        health_ok = False

    errors: List[Dict[str, Any]] = []
    samples_ms: List[float] = []

    t_start = time.perf_counter()

    if kind == "chat":
        worker = lambda: _worker_chat(base_url, chat_query, chat_top_k, timeout_s)
    elif kind == "image":
        if not image_path:
            raise SystemExit("--image-path is required for kind=image")
        p = Path(image_path)
        if not p.exists():
            raise SystemExit(f"image not found: {p}")
        img_bytes = p.read_bytes()
        worker = lambda: _worker_image_upload(base_url, img_bytes, p.name, image_top_k, timeout_s)
    else:
        raise SystemExit("kind must be chat or image")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker) for _ in range(n)]
        for fut in as_completed(futures):
            ok, dt, status, err = fut.result()
            samples_ms.append(float(dt))
            if not ok:
                errors.append({"status": status, "error": err, "latency_ms": float(dt)})

    t_total = time.perf_counter() - t_start

    out: Dict[str, Any] = {
        "kind": kind,
        "base_url": base_url,
        "n": int(n),
        "concurrency": int(concurrency),
        "timeout_s": float(timeout_s),
        "health_ok": bool(health_ok),
        "error_rate": float(len(errors) / n) if n else 0.0,
        "n_errors": int(len(errors)),
        "elapsed_s": float(t_total),
        "rps": float(n / t_total) if t_total > 0 else float("inf"),
        "latency_ms": {
            "min_ms": float(min(samples_ms)) if samples_ms else float("nan"),
            "mean_ms": float(statistics.mean(samples_ms)) if samples_ms else float("nan"),
            "max_ms": float(max(samples_ms)) if samples_ms else float("nan"),
            **_percentiles_ms(samples_ms, ps),
        },
    }

    # Chỉ kèm một mẫu nhỏ lỗi (tránh spam)
    if errors:
        out["errors_sample"] = errors[: min(10, len(errors))]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight load test for FastAPI endpoints")
    ap.add_argument("--base", default="http://127.0.0.1:8081", help="API base URL")
    ap.add_argument("--kind", choices=["chat", "image"], default="chat")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--c", type=int, default=10, help="concurrency")
    ap.add_argument("--timeout", type=float, default=60.0)

    ap.add_argument("--chat-query", default="men black sneakers under $80")
    ap.add_argument("--chat-topk", type=int, default=5)

    ap.add_argument("--image-path", default=None)
    ap.add_argument("--image-topk", type=int, default=3)

    args = ap.parse_args()

    result = run_load_test(
        kind=args.kind,
        base_url=args.base,
        n=args.n,
        concurrency=args.c,
        timeout_s=args.timeout,
        chat_query=args.chat_query,
        chat_top_k=args.chat_topk,
        image_path=args.image_path,
        image_top_k=args.image_topk,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
