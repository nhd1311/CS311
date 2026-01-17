# HỆ THỐNG TÌM KIẾM SẢN PHẨM THỜI TRANG

Hệ thống **Fashion Search/Chatbot** theo mô hình **RAG (Retrieval‑Augmented Generation)** cho **truy vấn văn bản** và **tìm kiếm bằng hình ảnh**, triển khai bằng **FastAPI** và **ChromaDB**.


---

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Kiến trúc & thành phần](#2-kiến-trúc--thành-phần)
- [3. Dữ liệu, collections & embedding](#3-dữ-liệu-collections--embedding)
- [4. Cấu hình (Environment Variables)](#4-cấu-hình-environment-variables)
- [5. Chạy hệ thống](#5-chạy-hệ-thống)
- [6. Ingest dữ liệu](#6-ingest-dữ-liệu)
- [7. Hợp đồng API (HTTP)](#7-hợp-đồng-api-http)
- [8. Logic RAG & các cơ chế lọc](#8-logic-rag--các-cơ-chế-lọc)
- [9. Tích hợp LLM & chính sách trả lời](#9-tích-hợp-llm--chính-sách-trả-lời)
- [10. Đánh giá & kết quả](#10-đánh-giá--kết-quả)
- [11. Troubleshooting / vận hành](#11-troubleshooting--vận-hành)
- [12. Cấu trúc repo](#12-cấu-trúc-repo)

---

## 1. Tổng quan

### 1.1 Mục tiêu

- **Text RAG**: người dùng nhập nhu cầu (tiếng Anh) → hệ thống truy xuất (retrieve) sản phẩm liên quan → trả về danh sách sản phẩm + câu trả lời.
- **Image RAG**: người dùng upload ảnh → hệ thống truy xuất sản phẩm tương tự theo embedding hình ảnh → trả về danh sách kết quả.

### 1.2 Ràng buộc quan trọng

1. **English-only**: API có kiểm tra “looks like English”. Nếu không đạt, endpoint `/chat` trả:
   - `answer: "I can't understand your question. Please rewrite it."`
   - `products: []`

   Endpoint `/query` hiện trả `error: "English only: please rephrase your request in English."`.

2. **Giảm hallucination**: mặc định `answer` được sinh **deterministic** từ chính danh sách `products` (để đồng bộ với UI cards). Có thể “ép” dùng LLM bằng `USE_LLM_ANSWER=1`.

---

## 2. Kiến trúc & thành phần

### 2.1 Sơ đồ kiến trúc (logical)

```text
Browser (index.html)
   |
   | HTTP (JSON / multipart)
   v
FastAPI (app/main.py)
   |
   | retrieve/query
   v
ChromaDB (collections)
   |  - products        (text embeddings)
   |  - products_image  (image embeddings)
   v
Answer generation
   - default: deterministic from products
   - optional: LLM via OpenAI-compatible API (HF Router)
```

### 2.2 Thành phần runtime

- **API**: FastAPI + Uvicorn.
- **Vector store**: ChromaDB.
- **Text embedding**: `sentence-transformers` (mặc định `sentence-transformers/all-MiniLM-L6-v2`).
- **Image embedding**: `open-clip-torch` (mặc định `ViT-B-32`, pretrained `openai`).
- **LLM** (tuỳ chọn): OpenAI-compatible API, khuyến nghị dùng **Hugging Face Router** (`https://router.huggingface.co/v1`).

### 2.3 Ports (Docker Compose mặc định)

- API: `8081 -> 8080` trong container
- Chroma: `8001 -> 8000` trong container

---

## 3. Dữ liệu, collections & embedding

### 3.1 Dataset

- CSV metadata: `datasets/archive/fashion-dataset/styles.csv`
- Ảnh: `datasets/archive/fashion-dataset/images/<id>.jpg`

### 3.2 Collections

- `products` (text): lưu `documents=text`, `metadatas=metadata`, `embeddings=text_embedding`.
- `products_image` (image): lưu `embeddings=image_embedding`, `metadatas=metadata`, `documents` thường rỗng (để ổn định phản hồi giữa các phiên bản Chroma).

### 3.3 Tính nhất quán embedding

Nếu bạn thay đổi `EMBED_MODEL` hoặc `IMAGE_MODEL*`, bạn **nên re-ingest** vào Chroma (hoặc tách storage/collection) để tránh “mismatch” giữa embedding cũ và mới.

---

## 4. Cấu hình (Environment Variables)

File mẫu: `.env.example` → copy thành `.env` (không commit `.env`).

### 4.1 Kết nối Chroma

| Biến          | Mặc định | Ý nghĩa                                                                |
| ------------- | -------: | ---------------------------------------------------------------------- |
| `CHROMA_HOST` | `chroma` | Host của Chroma HTTP (trong Docker Compose dùng service name `chroma`) |
| `CHROMA_PORT` |   `8000` | Port Chroma trong network Docker                                       |

Nếu **không** set `CHROMA_HOST/CHROMA_PORT`, code sẽ fallback Chroma **in-memory** (mất dữ liệu khi restart).

### 4.2 Models

| Biến                     |                                 Mặc định | Ý nghĩa                 |
| ------------------------ | ---------------------------------------: | ----------------------- |
| `EMBED_MODEL`            | `sentence-transformers/all-MiniLM-L6-v2` | Model embedding văn bản |
| `IMAGE_MODEL`            |                               `ViT-B-32` | Tên model OpenCLIP      |
| `IMAGE_MODEL_PRETRAINED` |                                 `openai` | Weights của OpenCLIP    |

### 4.3 LLM (tuỳ chọn)

| Biến             | Mặc định | Ý nghĩa                                                                      |
| ---------------- | -------: | ---------------------------------------------------------------------------- |
| `LLM_BASE_URL`   | _(rỗng)_ | Base URL OpenAI-compatible (khuyến nghị: `https://router.huggingface.co/v1`) |
| `LLM_API_KEY`    | _(rỗng)_ | API key (HF token nếu dùng HF Router)                                        |
| `LLM_MODEL`      | _(rỗng)_ | Tên model (ví dụ: `meta-llama/Llama-3.1-8B-Instruct`)                        |
| `LLM_DEBUG`      |      `0` | Log chi tiết lỗi gọi LLM                                                     |
| `USE_LLM_ANSWER` |      `0` | `1` = ép dùng LLM tạo `answer` ngay cả khi đã có `products`                  |

Ghi chú:

- Nếu `USE_LLM_ANSWER=1` nhưng thiếu `LLM_API_KEY` hoặc `LLM_MODEL`, hệ sẽ **fallback** về deterministic để đảm bảo UI nhất quán.

### 4.4 Chat/RAG behavior toggles

| Biến                          | Mặc định | Ý nghĩa                                                                             |
| ----------------------------- | -------: | ----------------------------------------------------------------------------------- |
| `ASK_FOLLOWUPS`               |      `0` | `1` = hỏi thêm 1–2 câu để làm rõ budget/occasion/size                               |
| `STRICT_EXACT_LOOKUP`         |      `0` | `1` = với câu hỏi “tìm đúng mẫu”, nếu match yếu thì nói rõ “không có exact product” |
| `RAG_MAX_DISTANCE`            |    `1.0` | Ngưỡng distance của Chroma (nhỏ hơn = tốt hơn)                                      |
| `RAG_MIN_TOKEN_OVERLAP`       |   `0.12` | Ngưỡng overlap token (lọc bổ sung)                                                  |
| `RAG_MIN_DISTINCTIVE_MATCHES` |      `1` | Số lượng token “đặc trưng” trùng tối thiểu                                          |

---

## 5. Chạy hệ thống

### 5.1 Chạy bằng Docker Compose (khuyến nghị)

Yêu cầu:

- Cài **Docker Desktop** và đảm bảo Docker Engine đang chạy.
- Lần chạy đầu có thể mất thời gian vì image cài `torch`, `sentence-transformers`, `open-clip`.

1. Tạo `.env`:

- Copy `.env.example` → `.env`.
- Nếu bạn **chưa có API key LLM**: để `USE_LLM_ANSWER=0` (mặc định). Khi đó hệ vẫn chạy bình thường và `answer` sẽ được tạo deterministic từ `products`.

2. Chạy dịch vụ:

```bash
docker compose up --build
```

3. Kiểm tra healthcheck:

- `http://127.0.0.1:8081/health` → mong đợi `{ "status": "ok" }`

4. Swagger UI:

- `http://127.0.0.1:8081/docs`

5. UI demo:

- Mở `index.html` (khuyến nghị mở qua VS Code Live Server để tránh một số hạn chế của trình duyệt khi mở file trực tiếp).
- UI gọi API theo mặc định tại `http://127.0.0.1:8081`.

### 5.2 Chạy local (không Docker)

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Nếu muốn dùng Chroma persistent khi chạy local: chạy Chroma bằng Docker và set `CHROMA_HOST=127.0.0.1`, `CHROMA_PORT=8001` (port host).

---

## 6. Ingest dữ liệu

### 6.1 Ingest text (bắt buộc trước khi chat/query)

Script: `ingest_csv.py` → gọi `POST /ingest` theo batch.

Ví dụ:

```bash
python ingest_csv.py --csv datasets/archive/fashion-dataset/styles.csv --api http://127.0.0.1:8081/ingest --batch 64
```

### 6.2 Ingest image (tuỳ chọn, để dùng image search)

Script: `ingest_images.py` → đọc ảnh theo `<id>.jpg` → gọi `POST /ingest_image`.

```bash
python ingest_images.py --csv datasets/archive/fashion-dataset/styles.csv --api http://127.0.0.1:8081/ingest_image --batch 128
```

Ghi chú vận hành:

- Batch quá lớn có thể gây peak RAM/VRAM (đặc biệt khi encode ảnh). Nếu gặp lỗi, giảm `--batch`.

---

## 7. Hợp đồng API (HTTP)

Tài liệu OpenAPI đầy đủ có tại `/docs`.

### 7.1 `GET /health`

Response:

```json
{ "status": "ok" }
```

### 7.2 `POST /ingest`

Request body (theo `app/models.py`):

```json
{
  "items": [{ "id": "123", "text": "...", "metadata": { "color": "Black" } }]
}
```

Response:

```json
{ "ingested": 64 }
```

### 7.3 `POST /query` (retrieval-only)

Request body:

```json
{
  "query": "men black sneakers under $80",
  "top_k": 3,
  "filters": { "gender": "Men" }
}
```

Response:

```json
{
  "results": [
    {
      "id": "...",
      "text": "...",
      "score": 0.78,
      "metadata": { "color": "Black" }
    }
  ]
}
```

Behavior notes:

- Nếu query fail English-only → trả `{ "results": [], "error": "English only: ..." }`.
- `filters` là **object**. Nếu client gửi `null` thay vì `{}`, FastAPI/Pydantic có thể trả `422` (schema mismatch).

### 7.4 `POST /chat` (RAG + answer)

Request body (hỗ trợ chat 1-turn hoặc multi-turn):

```json
{
  "query": "women black dress for party under $120",
  "messages": [{ "role": "user", "content": "..." }],
  "top_k": 3,
  "filters": {},
  "max_tokens": 512,
  "temperature": 0.2
}
```

Response (khung chính):

```json
{
  "answer": "...",
  "products": [
    {
      "id": "...",
      "name": "...",
      "image_url": "...",
      "price": "...",
      "color": "...",
      "gender": "...",
      "category": "...",
      "subcategory": "...",
      "usage": "...",
      "snippet": "..."
    }
  ],
  "sources": [
    { "id": "...", "text": "...", "score": 0.81, "metadata": { "...": "..." } }
  ],
  "outfit": {
    "topwear": { "...": "..." },
    "bottomwear": { "...": "..." },
    "footwear": { "...": "..." },
    "accessories": [],
    "estimated_total_price": 123.0
  }
}
```

Behavior notes:

- Empty query → trả câu hướng dẫn và `products: []`.
- Fail English-only → trả `answer: "I don't understand your question. Please rewrite it."` và `products: []`.

### 7.5 `POST /ingest_image`

Request body:

```json
{
  "items": [
    {
      "id": "123",
      "image_url": "datasets/.../images/123.jpg",
      "metadata": { "image_url": "..." }
    }
  ]
}
```

Response:

```json
{ "ingested": 128 }
```

### 7.6 `POST /search/image/upload`

Request: `multipart/form-data` upload file (field name `file`) + query param `top_k`.

Response:

```json
{
  "results": [
    {
      "id": "...",
      "text": "",
      "score": 0.42,
      "metadata": { "name": "...", "image_url": "..." }
    }
  ]
}
```

Ghi chú:

- Endpoint image search sẽ “enrich” metadata bằng cách lookup `products` để bổ sung `name`/`image_url` nếu thiếu.

---

## 8. Logic RAG & các cơ chế lọc

Các cơ chế chính nằm trong `app/main.py`:

1. **Relevance gating** (giảm trả kết quả “rác”):

- Gate theo `distance` (Chroma: **lower is better**) với `RAG_MAX_DISTANCE`.
- Gate theo lexical overlap (`RAG_MIN_TOKEN_OVERLAP`).
- Gate theo token “đặc trưng” (`RAG_MIN_DISTINCTIVE_MATCHES`).

2. **Hard filters** (khi user biểu đạt rõ ràng):

- Budget: parse `under/below/...` và lọc theo `price` (chưa chắc dataset có price đầy đủ).
- Color: nếu phát hiện màu, lọc cứng chỉ giữ items matching color.
- Usage/occasion: nếu phát hiện (formal/party/sports/...), có thêm retrieval pass + lọc cứng theo `metadata.usage`.
- Subcategory: có thể lọc theo `Topwear/Bottomwear/Bags`.

3. **Outfit mode**:

- Nếu query “outfit/complete look/...”, hệ cố gắng assemble mix (Topwear + Bottomwear + Footwear) và tránh innerwear trừ khi user yêu cầu.

---

## 9. Tích hợp LLM & chính sách trả lời

File: `app/llm_client.py`.

### 9.1 Chính sách mặc định (an toàn)

- Nếu đã có `products` và `USE_LLM_ANSWER=0` → `answer` được render deterministic từ `products`.
- Mục tiêu: **không mâu thuẫn** với UI product cards và hạn chế hallucination.

### 9.2 Bật LLM (tuỳ chọn)

- Set `USE_LLM_ANSWER=1` và cấu hình `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- Prompt đã có “HARD RULES” để buộc LLM chỉ dựa trên `PRODUCT LIST`/`PRODUCT CONTEXT`.

### 9.3 Fallback

- Nếu LLM lỗi/misconfigured/unreachable → fallback `_fallback_answer(...)` (retrieval-only) để API không bị “down”.

---

## 10. Đánh giá & kết quả

### 10.0 Baseline evaluation (khuyến nghị)

Mục tiêu trả lời 3 câu hỏi:

1. **Hệ thống hiện tại có chạy ổn không?** (healthcheck, error rate)
2. **Chất lượng trả lời / tuân thủ ràng buộc / latency đang ở mức nào?** (constraint pass rate, faithfulness heuristic, rubric 0–10, p50/p90/p95)
3. **Có lỗi English-only / safety / empty products không?** (reject accuracy, false reject rate, PII heuristic, empty-products rate)

Chạy nhanh (single-system):

```bash
python baseline_eval.py --base http://127.0.0.1:8081
```

Chạy với **gold testset** để tính **Hit@K / MRR / nDCG** (retrieval metrics):

```bash
python baseline_eval.py --base http://127.0.0.1:8081 --testset eval/testset_gold.json
```

Khuyến nghị (để tránh bị “dính cache cũ” khi lần trước API down / lỗi mạng): chạy vào thư mục output/artifacts riêng:

```bash
python baseline_eval.py --base http://127.0.0.1:8081 --testset eval/testset_gold.json --outputs outputs/gold_eval --artifacts artifacts/gold_eval
```

File xuất ra:

- `outputs/baseline_eval_results.csv`: kết quả theo từng test case (status, latency, n_products, constraint/faithfulness/safety, rubric)
- `artifacts/baseline_eval_summary.json`: tổng hợp cấp hệ thống (error_rate, latency p50/p95, empty_products_rate, english-only, safety)
- `artifacts/baseline_eval_config.json`: cấu hình chạy
- `artifacts/baseline_eval_cache.json`: cache kết quả (giúp chạy lại nhanh và ổn định)

Ghi chú:

- Khi dùng `--testset ...`, file CSV sẽ có thêm các cột retrieval:
  - `ret_hit@1`, `ret_hit@3`, `ret_hit@5`, `ret_mrr`, `ret_ndcg@3`, `ret_ndcg@5`
- Summary JSON cũng sẽ có các key tương ứng, cùng `retrieval_labeled_cases`.
- Các retrieval metrics chỉ được tính trên các case có `expected_ids` (gold) và `should_reject=false`.
- Nếu API **không chạy**, kết quả thường ra `error_rate=1.0`, `empty_products_rate=1.0` và các retrieval metrics gần như 0 — đây là dấu hiệu “server down”, không phải chất lượng hệ thống.

Bạn có thể chỉnh testset mặc định ngay trong `baseline_eval.py` (hàm `default_testset`).

### 10.0.1 Sinh gold testset (khởi tạo nhanh)

Repo có sẵn script sinh **testset có gold IDs** (dùng làm baseline, sau đó bạn chỉnh thủ công để “chốt”):

```bash
python eval/build_gold_testset.py --csv datasets/archive/fashion-dataset/styles.csv --out eval/testset_gold.json
```

Gold trong `eval/testset_gold.json` được **gợi ý tự động** dựa trên lọc metadata + token overlap đơn giản, vì vậy bạn nên rà soát lại `expected_ids` để phù hợp ý nghĩa truy vấn.

Hiện tại `eval/testset_gold.json` đã được mở rộng để có **~40 testcases** (bao phủ nhiều nhóm sản phẩm + budget + 2 case tiếng Việt để test English-only). Bạn có thể tiếp tục tăng số case bằng cách chỉnh trong `eval/build_gold_testset.py` và chạy lại script.

### 10.1 Công cụ

- `evaluate.ipynb`: đánh giá end-to-end (text + image), có thể xuất artifacts vào `artifacts/` và `outputs/`.
- `load_test.py`: load test nhẹ cho `/chat` và `/search/image/upload`.
- `baseline_eval.py`: baseline evaluation cho `/chat` (không cần notebook).

### 10.2 Metrics

- Reliability: error rate.
- Performance: p50/p90/p95 latency, throughput (RPS).
- Behavioral correctness: English-only handling, empty query handling.

### 10.3 Kết quả đo thực tế (2026-01-14)

Các số dưới đây được đo bằng `load_test.py` trên máy Windows của bạn, sau warm-up.

**Text `/chat`** (200 requests, concurrency=10):

- error_rate: **0.0** (0/200)
- latency (ms): p50 **257.93**, p90 **303.99**, p95 **334.01**
- throughput: ~**37.39 rps**

**Image `/search/image/upload`** (50 requests, concurrency=5, ảnh `10000.jpg`):

- error_rate: **0.0** (0/50)
- latency (ms): p50 **485.30**, p90 **527.27**, p95 **579.00**
- throughput: ~**10.15 rps**

---

## 11. Troubleshooting / vận hành

### 11.1 Cold start chậm

Lần đầu gọi `/chat` hoặc `/search/image/upload` sau restart/rebuild có thể chậm do load model (SentenceTransformer/OpenCLIP). Khuyến nghị warm-up vài request trước khi đo.

### 11.2 Lỗi "Connection refused" khi chạy eval

Dấu hiệu:

- `GET http://127.0.0.1:8081/health` không vào được
- `baseline_eval.py` báo `health_ok=false` hoặc `error_rate=1.0`

Cách xử lý:

1. Nếu bạn chạy bằng Docker Compose: đảm bảo stack đang chạy (`docker compose up --build`).
2. Kiểm tra containers đang chạy: `docker ps` (phải thấy API và chroma).
3. Sau khi API up, chạy lại eval và nên dùng thư mục `--artifacts` mới để tránh cache lỗi cũ.

### 11.3 Không có kết quả / kết quả rỗng

Kiểm tra theo thứ tự:

1. Đã ingest text vào `products` chưa?
2. Query có bị reject English-only không?
3. Các ngưỡng `RAG_*` có quá chặt không? (thử tăng `RAG_MAX_DISTANCE` hoặc giảm overlap)

### 11.4 Lỗi 422 (Unprocessable Entity)

Nguyên nhân phổ biến: gửi sai schema.

- `filters` trong `POST /query` và `POST /chat` nên là `{}` thay vì `null`.

### 11.5 LLM không chạy / không khác biệt

- Nếu `USE_LLM_ANSWER=0`, câu trả lời là deterministic từ `products` → thay model không làm thay đổi `answer`.
- Bật `USE_LLM_ANSWER=1` và đảm bảo `LLM_API_KEY` + `LLM_MODEL` hợp lệ.

---

## 12. Cấu trúc repo

### Backend (`app/`)

- `app/main.py`: endpoints + logic filter/relevance/English-only.
- `app/deps.py`: khởi tạo embedding models và Chroma collections.
- `app/rag.py`: ingest/retrieve text (`products`).
- `app/image_rag.py`: ingest/retrieve image (`products_image`).
- `app/llm_client.py`: LLM client + policy deterministic/LLM + fallback.
- `app/models.py`: Pydantic request/response models.

### Scripts

- `ingest_csv.py`: ingest text từ CSV.
- `ingest_images.py`: ingest ảnh.
- `load_test.py`: load test.
- `baseline_eval.py`: baseline evaluation cho `/chat` (stability/constraints/latency + retrieval metrics nếu có gold).

### Evaluation (`eval/`)

- `eval/build_gold_testset.py`: sinh testset gold IDs từ `styles.csv` (để bạn chỉnh và chốt).
- `eval/testset_gold.json`: testset mẫu (gold IDs gợi ý tự động).

### UI

- `index.html`: UI demo.

### Hạ tầng

- `docker-compose.yml`: chạy 1 API + Chroma.

---

## Ghi chú bảo mật

- Không commit `.env` / `.env.b`.
- Không paste `LLM_API_KEY` vào issue/public log.



