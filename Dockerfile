# syntax=docker/dockerfile:1.7

# Cố định phiên bản base image để cache layer của Docker ổn định theo thời gian.
FROM python:3.11.8-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .

# Dùng cache của BuildKit cho tải xuống/wheel của pip để tăng tốc rebuild.
# Cache này không được đóng gói vào image cuối cùng.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
