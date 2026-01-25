FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

# Системные зависимости (минимум)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip \
 && python -m pip install --prefer-binary -r requirements.txt

COPY src ./src
COPY cmd ./cmd
COPY proto ./proto
COPY conf ./conf
COPY scripts ./scripts

# Кеш для моделей HF (на время работы контейнера)
RUN mkdir -p /cache/huggingface

EXPOSE 5104
CMD ["python", "cmd/main.py"]
