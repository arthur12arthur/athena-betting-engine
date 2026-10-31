"""Logging uniforme (console + fichier optionnel)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(level: str | None = None, logfile: str | None = None) -> logging.Logger:
    lvl = (level or os.getenv("ATHENA_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger("athena")
    if root.handlers:
        return root
    root.setLevel(getattr(logging, lvl, logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"athena.{name}")
