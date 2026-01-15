import base64
import io
import os
import traceback
from typing import List, Optional, Dict, Any

import requests
import torch
from PIL import Image

from .models import IngestImageItem, QueryResult
from .deps import get_image_model, get_chroma_collection_image

_img_model = None
_img_preprocess = None
_img_device = None
_img_collection = None


def _normalize_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Chuẩn hoá dạng bộ lọc 'where' của Chroma (xem app.rag._normalize_where)."""
    if not filters:
        return None
    cleaned: Dict[str, Any] = {
        k: v for k, v in (filters or {}).items() if v is not None and v != {} and v != ""
    }
    if not cleaned:
        return None
    if any(str(k).startswith("$") for k in cleaned.keys()):
        return cleaned
    if len(cleaned) == 1:
        return cleaned
    return {"$and": [{k: v} for k, v in cleaned.items()]}


def _get_resources():
    global _img_model, _img_preprocess, _img_device, _img_collection
    if _img_model is None or _img_preprocess is None or _img_device is None:
        _img_model, _img_preprocess, _img_device = get_image_model()
    if _img_collection is None:
        _img_collection = get_chroma_collection_image()
    return _img_model, _img_preprocess, _img_device, _img_collection


def _load_image_from_url(url: str) -> Image.Image:
    # Hỗ trợ data URLs (base64), đường dẫn HTTP, và file local
    if url.startswith('data:image/'):
        # Data URL với mã hoá base64
        header, data = url.split(',', 1)
        img_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url.startswith(('http://', 'https://')):
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        # Đường dẫn file local
        if not os.path.exists(url):
            raise FileNotFoundError(f"Image file not found: {url}")
        return Image.open(url).convert("RGB")


def _load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _encode_image(img: Image.Image):
    img_model, img_preprocess, img_device, _ = _get_resources()
    with torch.no_grad():
        tensor = img_preprocess(img).unsqueeze(0).to(img_device)
        feats = img_model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]


def ingest_images(items: List[IngestImageItem]) -> int:
    _, _, _, img_collection = _get_resources()
    ids, embeddings, metadatas = [], [], []
    for it in items:
        try:
            img = _load_image_from_url(it.image_url)
            emb = _encode_image(img)
            ids.append(it.id)
            embeddings.append(emb)
            metadatas.append(it.metadata)
        except Exception as e:  # bỏ qua ảnh lỗi nhưng vẫn tiếp tục
            print(f"[WARN] Skip {it.id}: {e}")
            traceback.print_exc()
    if not ids:
        raise RuntimeError("No valid images ingested")
    # Chroma yêu cầu documents là một list chuỗi. Dùng chuỗi rỗng giúp phản hồi truy vấn nhất quán.
    img_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=[""] * len(ids))
    return len(ids)


def _safe_doc_text(res: dict, idx: int) -> str:
    """Trả về chuỗi an toàn cho document được truy xuất.

    Với collection ảnh, ta thường lưu document rỗng; một số phiên bản Chroma có thể trả về None.
    """
    docs = res.get("documents")
    if not docs:
        return ""
    try:
        v = docs[0][idx]
    except Exception:
        return ""
    if v is None:
        return ""
    return str(v)


def retrieve_by_image(image_url: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
    _, _, _, img_collection = _get_resources()
    where = _normalize_where(filters)
    img = _load_image_from_url(image_url)
    q_emb = _encode_image(img)
    res = img_collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    out = []
    for idx, doc_id in enumerate(res.get("ids", [[]])[0]):
        out.append(
            QueryResult(
                id=doc_id,
                text=_safe_doc_text(res, idx),
                score=res["distances"][0][idx],
                metadata=res["metadatas"][0][idx],
            )
        )
    return out


def retrieve_by_image_file(image_bytes: bytes, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
    _, _, _, img_collection = _get_resources()
    where = _normalize_where(filters)
    img = _load_image_from_bytes(image_bytes)
    q_emb = _encode_image(img)
    res = img_collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    out = []
    for idx, doc_id in enumerate(res.get("ids", [[]])[0]):
        out.append(
            QueryResult(
                id=doc_id,
                text=_safe_doc_text(res, idx),
                score=res["distances"][0][idx],
                metadata=res["metadatas"][0][idx],
            )
        )
    return out
