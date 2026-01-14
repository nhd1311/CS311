# syntax=docker/dockerfile:1.7

# Pin the base image version to keep Docker layer caching stable over time.
FROM python:3.11.8-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .

# Use BuildKit cache for pip downloads/wheels to speed up rebuilds.
# The cache is not baked into the final image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
