"""
logger.py — Centralised Logging Configuration
===============================================
Single source of truth for logging across the entire framework.

Usage (in any module):
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Running CUPED adjustment", extra={"n_control": 500})

Environment Variables
---------------------
LOG_LEVEL   : DEBUG | INFO | WARNING | ERROR | CRITICAL  (default: INFO)
LOG_FORMAT  : json | text                                 (default: text)
LOG_FILE    : path to log file                            (default: None, stdout only)

Examples
--------
# Development (human-readable)
LOG_LEVEL=DEBUG python -m src.experiment --data data/sample_experiment.csv ...

# Production (structured JSON, file output)
LOG_LEVEL=INFO LOG_FORMAT=json LOG_FILE=logs/experiment.log python -m src.experiment ...
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each log line is a valid JSON object with consistent fields, making it
    easy to ingest into log aggregators (Datadog, Splunk, CloudWatch, etc.).

    Fields always present
    ---------------------
    timestamp   : ISO-8601 UTC timestamp
    level       : log level name (INFO, WARNING, etc.)
    logger      : dotted logger name (e.g. src.stats.cuped)
    message     : the log message
    module      : source file module
    line        : source line number

    Fields present when applicable
    --------------------------------
    exception   : formatted traceback (on ERROR/CRITICAL)
    ...extra    : any extra= kwargs passed to the log call
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Attach any extra fields passed via extra={...}
        skip = {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                payload[key] = value

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Text Formatter (development)
# ---------------------------------------------------------------------------

TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger for the framework.

    Call once at application startup (e.g. in experiment.py main()).
    Subsequent calls to get_logger() will inherit this configuration.

    Parameters
    ----------
    level : str, optional
        Log level. Reads LOG_LEVEL env var if not provided.
    fmt : str, optional
        'json' or 'text'. Reads LOG_FORMAT env var if not provided.
    log_file : str, optional
        Path to log file. Reads LOG_FILE env var if not provided.
    """
    level    = (level    or os.getenv("LOG_LEVEL",  "INFO")).upper()
    fmt      = (fmt      or os.getenv("LOG_FORMAT", "text")).lower()
    log_file = log_file  or os.getenv("LOG_FILE",   None)

    numeric_level = getattr(logging, level, logging.INFO)

    # Root logger for the framework namespace only
    root = logging.getLogger("src")
    root.setLevel(numeric_level)
    root.handlers.clear()

    formatter: logging.Formatter
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(TEXT_FORMAT, datefmt=DATE_FORMAT)

    # Always write to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # Optionally write to a rotating file
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False  # don't bubble up to the global root logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped to *name*.

    Parameters
    ----------
    name : str
        Typically __name__ from the calling module.
        e.g. 'src.stats.cuped' → logger name 'src.stats.cuped'

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
