from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    items: List[IngestItem]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    filters: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


# Nạp (ingest) và truy vấn ảnh
class IngestImageItem(BaseModel):
    id: str
    image_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestImageRequest(BaseModel):
    items: List[IngestImageItem]

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    # Tương thích ngược: client cũ chỉ gửi `query`.
    # Client mới có thể gửi `messages` cho hội thoại nhiều lượt.
    query: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
    top_k: int = 3
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.2
