import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import open_clip
import time


EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "ViT-B-32")
IMAGE_MODEL_PRETRAINED = os.getenv("IMAGE_MODEL_PRETRAINED", "openai")


def get_model():
    # cache singleton đơn giản
    global _model
    try:
        return _model  # type: ignore[name-defined]
    except NameError:
        _model = SentenceTransformer(EMBED_MODEL)
        return _model


def get_image_model():
    global _img_model, _img_preprocess, _img_device
    try:
        return _img_model, _img_preprocess, _img_device  # type: ignore[name-defined]
    except NameError:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            IMAGE_MODEL_NAME, pretrained=IMAGE_MODEL_PRETRAINED
        )
        model.to(device)
        _img_model, _img_preprocess, _img_device = model, preprocess, device
        return _img_model, _img_preprocess, _img_device


def get_chroma_collection():
    host = os.getenv("CHROMA_HOST")
    port = os.getenv("CHROMA_PORT")

    if host and port:
        last_err = None
        for _ in range(10):
            try:
                client = chromadb.HttpClient(host=host, port=int(port))
                # Gửi một request nhẹ để kiểm tra kết nối.
                try:
                    client.heartbeat()
                except Exception:
                    # Một số phiên bản client (cũ/mới) có thể không có heartbeat; bỏ qua.
                    pass
                return client.get_or_create_collection("products")
            except Exception as e:
                last_err = e
                time.sleep(1)
        # Phương án cuối cùng: dùng bộ nhớ (in-memory) để API vẫn chạy.
        # (UI vẫn hoạt động, nhưng kết quả có thể trống cho tới khi Chroma truy cập được.)
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        return client.get_or_create_collection("products")
    else:
        # dùng in-memory cho local/dev để khởi chạy nhanh
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    return client.get_or_create_collection("products")


def get_chroma_collection_image():
    host = os.getenv("CHROMA_HOST")
    port = os.getenv("CHROMA_PORT")

    if host and port:
        last_err = None
        for _ in range(10):
            try:
                client = chromadb.HttpClient(host=host, port=int(port))
                try:
                    client.heartbeat()
                except Exception:
                    pass
                return client.get_or_create_collection("products_image")
            except Exception as e:
                last_err = e
                time.sleep(1)

        # Nếu không truy cập được Chroma qua HTTP thì chuyển sang in-memory.
        client = chromadb.Client(Settings(anonymized_telemetry=False))
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    return client.get_or_create_collection("products_image")
