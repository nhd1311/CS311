from typing import List, Optional, Dict, Any

from .models import IngestItem, QueryResult
from .deps import get_model, get_chroma_collection


_model = None
_collection = None


def _normalize_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Chuẩn hoá dạng bộ lọc 'where' của Chroma.

    Một số phiên bản Chroma yêu cầu `where` ở mức top-level chỉ được chứa đúng
    một toán tử (ví dụ: "$and") HOẶC một ràng buộc của một trường. Khi caller truyền
    nhiều ràng buộc như {"gender": "Women", "price": {"$lte": 70}},
    hãy bọc chúng vào một biểu thức $and tường minh.
    """
    if not filters:
        return None

    # Loại bỏ các giá trị rỗng để tránh tạo where clause không hợp lệ.
    cleaned: Dict[str, Any] = {
        k: v for k, v in (filters or {}).items() if v is not None and v != {} and v != ""
    }
    if not cleaned:
        return None

    # Nếu người dùng đã cung cấp where clause dạng toán tử, giữ nguyên.
    if any(str(k).startswith("$") for k in cleaned.keys()):
        return cleaned

    # Chỉ một ràng buộc trường: hợp lệ.
    if len(cleaned) == 1:
        return cleaned

    # Nhiều ràng buộc trường: bọc bằng $and.
    return {"$and": [{k: v} for k, v in cleaned.items()]}


def _get_resources():
    global _model, _collection
    if _model is None:
        _model = get_model()
    if _collection is None:
        _collection = get_chroma_collection()
    return _model, _collection


def ingest(items: List[IngestItem]) -> int:
    model, collection = _get_resources()
    texts = [it.text for it in items]
    ids = [it.id for it in items]
    metadatas = [it.metadata for it in items]
    # Giữ tham số bảo thủ để giảm đỉnh bộ nhớ trong lúc ingest.
    embs = model.encode(texts, batch_size=32, show_progress_bar=False)
    upsert = getattr(collection, "upsert", None)
    if callable(upsert):
        upsert(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)
    else:
        collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)
    return len(items)


def retrieve(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
    model, collection = _get_resources()
    filters = filters or {}
    q_emb = model.encode([query])[0]
    where = _normalize_where(filters)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    out = []
    for idx, doc_id in enumerate(res.get("ids", [[]])[0]):
        out.append(
            QueryResult(
                id=doc_id,
                text=res["documents"][0][idx],
                score=res["distances"][0][idx],
                metadata=res["metadatas"][0][idx],
            )
        )
    return out


def get_items_by_ids(ids: List[str]):
    """Lấy item theo id từ text collection.

    Trả về dict: id -> {"text": str, "metadata": dict}
    """
    if not ids:
        return {}
    _, collection = _get_resources()

    getter = getattr(collection, "get", None)
    if not callable(getter):
        return {}

    res = getter(ids=ids, include=["documents", "metadatas"])  # type: ignore[arg-type]
    out = {}
    res_ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for i, doc_id in enumerate(res_ids):
        out[str(doc_id)] = {
            "text": "" if i >= len(docs) or docs[i] is None else str(docs[i]),
            "metadata": {} if i >= len(metas) or metas[i] is None else metas[i],
        }
    return out
