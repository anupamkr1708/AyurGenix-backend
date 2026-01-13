FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python deps
# -----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn "uvicorn[standard]"

# -----------------------------
# Copy project
# -----------------------------
COPY . .

EXPOSE 10000

CMD gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers 1 \
  --threads 2 \
  --timeout 600
