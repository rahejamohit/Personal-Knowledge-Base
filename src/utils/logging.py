"""Centralized logging configuration.

Architectural notes
-------------------
* We configure logging in one place (`configure_logging`) so every entry point
  — CLI, scripts, tests — gets the same format and level. This avoids the
  classic Python footgun of multiple `logging.basicConfig` calls silently
  no-op'ing each other.
* JSON output is opt-in via `PKA_LOG_FORMAT=json`. Phase 3 (production)
  switches to JSON for ingestion into Loki/Cloud Logging; Phase 1 defaults to
  human-readable text for laptop dev.
* `get_logger(__name__)` is the only public accessor — modules should not
  call `logging.getLogger` directly so that any future cross-cutting concern
  (correlation IDs, OpenTelemetry trace context) has one chokepoint.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

from pythonjsonlogger import jsonlogger

_CONFIGURED: bool = False

_DEFAULT_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(level: str | int | None = None, *, json_output: bool | None = None) -> None:
    """Idempotently configure the root logger.

    Parameters
    ----------
    level:
        Log level name ("DEBUG", "INFO", ...) or numeric level. If `None`,
        reads `PKA_LOG_LEVEL` (default INFO).
    json_output:
        If `True`, emit JSON lines (good for log aggregators). If `None`,
        reads `PKA_LOG_FORMAT` ("json" toggles on).

    Calling more than once is safe — subsequent calls re-apply the level but
    do not duplicate handlers (which would cause double-logging).
    """
    global _CONFIGURED

    resolved_level = level if level is not None else os.getenv("PKA_LOG_LEVEL", "INFO")
    resolved_json = (
        json_output
        if json_output is not None
        else os.getenv("PKA_LOG_FORMAT", "").lower() == "json"
    )

    root = logging.getLogger()
    root.setLevel(resolved_level)

    if _CONFIGURED:
        # Already attached a handler — just update the level and bail.
        return

    handler = logging.StreamHandler(sys.stderr)
    if resolved_json:
        # Includes std fields plus anything passed via `extra={...}`.
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"levelname": "level", "asctime": "ts"},
            )
        )
    else:
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root.addHandler(handler)

    # Tame noisy third-party loggers. CrewAI/LiteLLM emit a lot at DEBUG;
    # users can re-enable explicitly via `PKA_LOG_LEVEL=DEBUG`.
    for noisy in ("httpx", "httpcore", "openai", "google", "LiteLLM", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Always use this instead of `logging.getLogger`."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
