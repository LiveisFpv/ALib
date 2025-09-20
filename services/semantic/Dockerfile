FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

# System deps (optional, keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY src ./src
COPY cmd ./cmd
COPY proto ./proto
COPY conf ./conf
COPY scripts ./scripts

# Create cache dir for HF models
RUN mkdir -p /cache/huggingface

EXPOSE 5104

CMD ["python", "cmd/main.py"]

