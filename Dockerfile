FROM python:3.10-slim

# -----------------------------
# Environment optimizations
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV LANGCHAIN_TRACING_V2=false

WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install gunicorn "uvicorn[standard]"

# -----------------------------
# Copy project
# -----------------------------
COPY . .

# -----------------------------
# Expose port (Render uses $PORT)
# -----------------------------
EXPOSE 10000

# -----------------------------
# Production server
# -----------------------------
CMD gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers 1 \
  --threads 2 \
  --timeout 600
