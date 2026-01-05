"""Application configuration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", "5044"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
SEMANTIC_PORT = os.getenv("SEMANTIC_PORT", "5104")


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "semantic_db")
DB_SSL_MODE = os.getenv("SSLMode") or os.getenv("DB_SSL_MODE", "disable")


def build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    return (
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}?sslmode={DB_SSL_MODE}"
    )


DATABASE_URL = build_database_url()


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index/faiss.index")
FAISS_DOC_IDS_PATH = os.getenv("FAISS_DOC_IDS_PATH", "data/index/doc_ids.npy")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
EMBEDDING_LORA_PATH = os.getenv("EMBEDDING_LORA_PATH")


@dataclass(slots=True)
class DatabaseSettings:
    host: str = DB_HOST
    port: int = DB_PORT
    user: str = DB_USER
    password: str = DB_PASSWORD
    name: str = DB_NAME
    ssl_mode: str = DB_SSL_MODE
    url: str = DATABASE_URL

    def psycopg_dsn(self, *, driver: Optional[str] = None) -> str:
        base_driver = driver or "postgresql"
        dsn = (
            f"{base_driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )
        if self.ssl_mode:
            dsn = f"{dsn}?sslmode={self.ssl_mode}"
        return dsn


DATABASE_SETTINGS = DatabaseSettings()
