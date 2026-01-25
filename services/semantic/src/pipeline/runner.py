"""Orchestrates the end-to-end ingestion pipeline."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.pipeline.settings import PipelineSettings


LOGGER = logging.getLogger("pipeline.runner")


def _run_command(cmd: Sequence[str], *, cwd: str | None = None) -> None:
    display = " ".join(cmd)
    LOGGER.info("Running command: %s", display)
    subprocess.run(cmd, check=True, cwd=cwd)


@dataclass(slots=True)
class PipelineRunner:
    settings: PipelineSettings = field(default_factory=PipelineSettings)

    def ensure_directories(self) -> None:
        for path in (self.settings.raw_dir, self.settings.processed_dir, self.settings.index_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def fetch_openalex(self) -> None:
        if not self.settings.openalex_email:
            raise RuntimeError("OPENALEX_EMAIL must be set for fetching data")
        cmd = [
            "python",
            "src/parser/openalex_csv_parser.py",
            "--email",
            self.settings.openalex_email,
            "--outdir",
            self.settings.raw_dir,
            "--en",
            str(self.settings.openalex_en),
            "--ru",
            str(self.settings.openalex_ru),
            "--chunk-size",
            str(self.settings.openalex_chunk_size),
        ]
        if self.settings.openalex_gzip:
            cmd.append("--gzip")
        if self.settings.openalex_resume:
            cmd.append("--resume")
        _run_command(cmd)

    def clean_openalex(self) -> None:
        cmd = [
            "python",
            "src/parser/clean_openalex.py",
            "--indir",
            self.settings.raw_dir,
            "--outdir",
            self.settings.processed_dir,
        ]
        _run_command(cmd)

    def load_database(self) -> None:
        cmd = [
            "python",
            "src/parser/load_openalex_to_db.py",
            "--indir",
            self.settings.processed_dir,
        ]
        _run_command(cmd)

    def build_embeddings(self) -> None:
        cmd = [
            "python",
            "src/parser/e5_embed_corpus.py",
            "--outdir",
            self.settings.index_dir,
        ]
        _run_command(cmd)

    def build_faiss(self) -> None:
        cmd = [
            "python",
            "src/parser/e5_build_faiss.py",
            "--emb-dir",
            self.settings.index_dir,
        ]
        _run_command(cmd)

    def run_full(self) -> None:
        LOGGER.info("Starting full pipeline")
        self.ensure_directories()
        # self.fetch_openalex()
        # self.clean_openalex()
        self.load_database()
        # self.build_embeddings()
        # self.build_faiss()
        LOGGER.info("Pipeline completed")


__all__ = ["PipelineRunner"]
