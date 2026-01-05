"""Signal-based pipeline worker."""

from __future__ import annotations

import logging
import signal
import threading
import time

from src.pipeline.runner import PipelineRunner


LOGGER = logging.getLogger("pipeline.worker")


class PipelineWorker:
    """Waits for POSIX signal and triggers the pipeline."""

    def __init__(self, runner: PipelineRunner | None = None, *, trigger_signal: int = signal.SIGUSR1) -> None:
        self.runner = runner or PipelineRunner()
        self.trigger_signal = trigger_signal
        self._event = threading.Event()
        self._running = False
        signal.signal(self.trigger_signal, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # type: ignore[override]
        LOGGER.info("Received signal %s", signum)
        self._event.set()

    def run_once(self) -> None:
        LOGGER.info("Running pipeline once")
        self.runner.run_full()

    def serve(self, poll_interval: float = 1.0) -> None:
        LOGGER.info("Pipeline worker started; waiting for signal %s", self.trigger_signal)
        self._running = True
        try:
            while self._running:
                if self._event.wait(poll_interval):
                    self._event.clear()
                    try:
                        self.runner.run_full()
                    except Exception as exc:  # pragma: no cover - log and continue
                        LOGGER.exception("Pipeline run failed: %s", exc)
        finally:
            LOGGER.info("Pipeline worker stopped")

    def stop(self) -> None:
        self._running = False
        self._event.set()


__all__ = ["PipelineWorker"]
