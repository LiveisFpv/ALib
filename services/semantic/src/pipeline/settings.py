"""Settings for the ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class PipelineSettings:
    data_root: str = os.getenv("DATA_ROOT", "data")
    raw_dir: str = os.getenv("PIPELINE_RAW_DIR", "data/raw")
    processed_dir: str = os.getenv("PIPELINE_PROCESSED_DIR", "data/processed")
    index_dir: str = os.getenv("PIPELINE_INDEX_DIR", "data/index")
    openalex_email: str | None = os.getenv("OPENALEX_EMAIL")
    openalex_en: int = int(os.getenv("PIPELINE_OPENALEX_EN", "1000000"))
    openalex_ru: int = int(os.getenv("PIPELINE_OPENALEX_RU", "550000"))
    openalex_chunk_size: int = int(os.getenv("PIPELINE_OPENALEX_CHUNK", "50000"))
    openalex_gzip: bool = os.getenv("PIPELINE_OPENALEX_GZIP", "true").lower() in {"1", "true", "yes"}
    openalex_resume: bool = os.getenv("PIPELINE_OPENALEX_RESUME", "true").lower() in {"1", "true", "yes"}


DEFAULT_SETTINGS = PipelineSettings()


__all__ = ["PipelineSettings", "DEFAULT_SETTINGS"]
