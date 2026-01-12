FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (for layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install gunicorn "uvicorn[standard]"

# Copy project
COPY . .

# Render injects PORT, don't hardcode
EXPOSE 10000

# IMPORTANT: bind to $PORT and single worker (low RAM)
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
