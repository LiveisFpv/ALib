#!/usr/bin/env python3
"""CLI entrypoint for the ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.config import dictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.runner import PipelineRunner
from src.pipeline.worker import PipelineWorker


def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    level = os.getenv("PIPELINE_LOG_LEVEL", "INFO").upper()
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": level,
                }
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run or serve the ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["once", "serve"],
        default="once",
        help="Run pipeline once or stay alive awaiting signal",
    )
    parser.add_argument(
        "--signal",
        type=int,
        default=None,
        help="POSIX signal number to trigger pipeline (serve mode)",
    )
    args = parser.parse_args()

    if args.mode == "once":
        PipelineRunner().run_full()
    else:
        worker = PipelineWorker(trigger_signal=args.signal or 10)  # default SIGUSR1=10
        worker.serve()


if __name__ == "__main__":
    main()
