# Fashion RAG (Text + Image) – Hướng dẫn sử dụng

Dự án này là một **Fashion Chatbot** dùng **RAG (Retrieval-Augmented Generation)** để tìm kiếm sản phẩm thời trang theo:

- **Văn bản** (mô tả/nhu cầu)
- **Hình ảnh** (tìm sản phẩm tương tự từ ảnh)

Backend viết bằng **FastAPI**, lưu vector trong **ChromaDB**, embed văn bản bằng **Sentence-Transformers** và embed ảnh bằng **OpenCLIP**. Có sẵn UI demo trong `index.html`.

---

## Kiến trúc nhanh

Khi chạy bằng **Docker Compose** (khuyến nghị):

- API (FastAPI): `http://127.0.0.1:8081`
- ChromaDB (vector database): `http://127.0.0.1:8001`

> Ghi chú: nếu máy bạn đã có service khác dùng cổng 8080/8000, bạn có thể đổi host port trong `docker-compose.yml` (ví dụ 8081/8001). Hãy dùng đúng URL theo port mapping hiện tại.

- 2 collections:
  - `products` (text)
  - `products_image` (image)

---

## Yêu cầu

### Cách 1 (khuyến nghị): Docker Compose

- Docker Desktop (Windows)

### Cách 2: Chạy local bằng Python

- Python 3.11+
- Cài dependencies trong `requirements.txt`

---

## Cấu hình môi trường (.env)

Trong repo đã có `.env` và `.env.example`.

- Nếu bạn chưa có `.env` đúng, hãy copy từ mẫu:
  - Copy `.env.example` → `.env` và chỉnh lại các biến.

Các biến quan trọng:

- **Chroma** (dùng khi chạy qua docker-compose):

  - `CHROMA_HOST` (ví dụ: `chroma` trong docker-compose)
  - `CHROMA_PORT` (mặc định `8000`)

- **Embedding text / image**:

  - `EMBED_MODEL` (mặc định: `sentence-transformers/all-MiniLM-L6-v2`)
  - `IMAGE_MODEL` (mặc định: `ViT-B-32`)
  - `IMAGE_MODEL_PRETRAINED` (mặc định: `openai`)

- **LLM (tùy chọn)**: dùng để tạo câu trả lời dựa trên ngữ cảnh RAG.
  - `LLM_BASE_URL`
  - `LLM_API_KEY`
  - `LLM_MODEL`

Ghi chú:

- `app/llm_client.py` hỗ trợ **2 chế độ**:
  1. **Hugging Face (legacy)** nếu `LLM_BASE_URL` chứa `api-inference.huggingface.co` (endpoint này hiện đã bị HF deprecate)
  2. **OpenAI-compatible Chat Completions** (khuyến nghị): dùng OpenAI / Ollama / LM Studio / hoặc Hugging Face Router `https://router.huggingface.co/v1`
- Nếu LLM bị lỗi/không cấu hình, hệ thống sẽ **fallback** sang trả lời dựa trên retrieval.

Lưu ý quan trọng khi dùng Docker:

- `env_file: .env` chỉ được đọc khi container được **tạo**. Nếu bạn thay đổi `.env`, hãy **recreate** container `api` (ví dụ: `docker compose down` rồi `docker compose up -d --build`).

---

## Chạy dự án

### Cách 1: Docker Compose (API + ChromaDB)

1. Khởi động dịch vụ:

```bash
docker compose up --build
```

- ChromaDB sẽ chạy ở cổng `8001`
- API sẽ chạy ở cổng `8081`

2. Mở tài liệu API (Swagger):

- `http://127.0.0.1:8081/docs`

> Lưu ý: `docker-compose.yml` dùng `env_file: .env` cho service `api`. Hãy đảm bảo `.env` tồn tại và hợp lệ.

3. Dừng dịch vụ:

```bash
docker compose down
```

### Cách 2: Chạy local (nhanh để dev)

1. Tạo môi trường và cài thư viện:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:

```bat
.\.venv\Scripts\activate.bat
```

```bash
pip install -r requirements.txt
```

2. Chạy API bằng uvicorn:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Ghi chú:

- Nếu bạn **không set `CHROMA_HOST/CHROMA_PORT`**, code sẽ dùng Chroma **in-memory** (không lưu persist) theo `app/deps.py`.

---

## Nạp dữ liệu (Ingest)

Dataset mặc định hiện tại:

- `datasets/archive/fashion-dataset/styles.csv`

Ghi chú:

- Nếu bạn có dataset khác, chỉ cần trỏ tham số `--csv` tới file CSV của bạn.

### 1) Ingest văn bản (text)

Script: `ingest_csv.py`

- Script sẽ đọc CSV và gọi API endpoint `POST /ingest` theo batch.
- Mặc định API ingest: `http://localhost:8080/ingest` (có thể đổi qua biến môi trường `INGEST_API`).

Ví dụ chạy:

```bash
python ingest_csv.py --csv datasets/archive/fashion-dataset/styles.csv --api http://127.0.0.1:8081/ingest --batch 64
```

Nếu bạn chạy **local** (API ở port 8080), đổi `--api` thành `http://127.0.0.1:8080/ingest`.

### 2) Ingest hình ảnh (image)

Script: `ingest_images.py`

- Script sẽ đọc cột ảnh (mặc định: `image_url`) và gọi API endpoint `POST /ingest_image`.
- Mặc định API ingest ảnh: `http://localhost:8080/ingest_image` (có thể đổi qua biến môi trường `INGEST_IMAGE_API`).

Ví dụ chạy:

```bash
python ingest_images.py --csv datasets/archive/fashion-dataset/styles.csv --api http://127.0.0.1:8081/ingest_image --batch 128
```

Nếu bạn chạy **local** (API ở port 8080), đổi `--api` thành `http://127.0.0.1:8080/ingest_image`.

Gợi ý:

- Nếu mạng chậm hoặc ảnh lỗi, script sẽ bỏ qua ảnh lỗi và tiếp tục.

---

## UI Demo

File UI: `index.html`

- Mở trực tiếp file bằng trình duyệt (hoặc dùng Live Server trong VS Code).
- Demo mặc định gọi API:
  - `const API_BASE = 'http://127.0.0.1:8081';`

Tính năng demo:

- Chat text: gọi `POST /chat`
- Search ảnh:
  - Upload ảnh: `POST /search/image/upload`

---

## API endpoints (tóm tắt)

- `GET /health` – healthcheck
- `POST /ingest` – ingest text items vào collection `products`
- `POST /query` – search text (semantic) trong `products`
- `POST /ingest_image` – ingest image URLs vào collection `products_image`
- `POST /search/image/upload` – search theo ảnh upload
- `POST /chat` – chat nhiều lượt + RAG (trả về `answer`, `products`, `sources`)

Bạn xem schema chi tiết tại Swagger: `http://127.0.0.1:8081/docs`.

Ví dụ gọi nhanh `POST /chat`:

macOS/Linux (bash):

```bash
curl -X POST http://127.0.0.1:8081/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"I need a white men's dress shirt for work, budget under $40\",\"top_k\":5}"
```

Windows (PowerShell):

```powershell
$body = @{ query = "I need a white men's dress shirt for work, budget under `$40"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8081/chat" -ContentType "application/json" -Body $body
```

> Nếu bạn chạy local (API ở port 8080), đổi URL từ `8081` → `8080`.

---

## Troubleshooting

### Tăng tốc hệ thống (Performance)

Nếu bạn thấy hệ thống phản hồi chậm, thường là do 3 nguyên nhân chính:

1. **Gọi LLM qua mạng** (chậm nhất)
2. **Embedding chạy CPU** (nhất là trên máy yếu / Docker bị giới hạn CPU)
3. **Search ảnh phải tải ảnh từ URL** (mạng chậm)

Các gợi ý nhanh:

- **LLM qua mạng thường là chậm nhất**: nếu không cần câu trả lời “tự nhiên”, bạn có thể để trống cấu hình LLM; hệ thống sẽ tự **fallback** sang retrieval-only.
- **Embedding chạy CPU**: nếu máy yếu, lần ingest đầu tiên sẽ tốn thời gian; sau khi ingest xong, query thường nhanh hơn.
- **Search ảnh**: sẽ tốn thời gian encode ảnh; tránh ingest ảnh nếu bạn chỉ cần search text.
- Nếu dùng Docker Desktop trên Windows, hãy đảm bảo Docker có đủ CPU/RAM trong Docker Desktop Settings.

Mẹo hữu ích (PyTorch):

- Bạn có thể set `TORCH_NUM_THREADS` để giới hạn số luồng CPU (tuỳ máy). Ví dụ: `TORCH_NUM_THREADS=4`.

Gợi ý thêm:

- Nếu bạn không cần search ảnh, hãy tạm thời không ingest ảnh để tiết kiệm thời gian/CPU.

- **Fatal error in launcher / uvicorn.exe trỏ sai đường dẫn .venv (Windows)**:

  Lỗi dạng:
  `Fatal error in launcher: Unable to create process using ... DoAn\\.venv\\Scripts\\python.exe ... CS311\\.venv\\Scripts\\uvicorn.exe ...`

  Nguyên nhân thường gặp: bạn đã **copy/đổi tên/move thư mục `.venv`** từ project khác.

  Cách xử lý nhanh:

  - Chạy bằng module để bỏ qua launcher `.exe`:
    `python -m uvicorn app.main:app --host 0.0.0.0 --port 8080`

  Cách xử lý dứt điểm (khuyến nghị):

  - Xoá `.venv` và tạo lại đúng trong repo hiện tại, rồi `pip install -r requirements.txt`.

- **Demo không gọi được API**:

  - Đảm bảo API chạy ở `127.0.0.1:8081` (hoặc đúng host port bạn đang map trong `docker-compose.yml`).
  - Trên Windows, dùng `127.0.0.1` giúp tránh lỗi IPv6 `localhost -> ::1`.

- **Kết quả rỗng**:

  - Bạn chưa ingest dữ liệu (hãy chạy `ingest_csv.py`, và nếu dùng search ảnh thì chạy thêm `ingest_images.py`).
  - Nếu chạy local không có Chroma server, bạn đang dùng Chroma in-memory (mất dữ liệu khi restart).

- **LLM không trả lời đúng / fallback**:
  - Nếu bạn thấy dòng `(Fallback mode: answering from retrieved results without the LLM.)` trong câu trả lời, nghĩa là hệ thống **không gọi được LLM** và đang trả lời bằng retrieval-only.
  - Kiểm tra `LLM_BASE_URL`, `LLM_MODEL` và `LLM_API_KEY` trong `.env`.
  - Bật log chẩn đoán để biết chính xác lỗi gì:
    - Set `LLM_DEBUG=1` trong `.env`
    - Recreate container `api` (ví dụ: `docker compose up --build -d`)
    - Xem log: `docker compose logs --tail 200 api`
  - Nếu không muốn dùng LLM, vẫn có thể dùng retrieval-only (hệ thống tự fallback).

---

## Thư mục quan trọng

- `app/main.py` – FastAPI endpoints
- `app/rag.py` – RAG text (ingest/retrieve)
- `app/image_rag.py` – RAG image (OpenCLIP + Chroma)
- `app/llm_client.py` – gọi LLM (HF Inference / OpenAI-compatible)
- `ingest_csv.py` – ingest dữ liệu text từ CSV
- `ingest_images.py` – ingest dữ liệu ảnh từ CSV
- `index.html` – UI demo

---

## License / Ghi chú
