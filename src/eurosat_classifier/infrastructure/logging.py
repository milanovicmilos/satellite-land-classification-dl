"""Logging helpers for entrypoints and infrastructure services."""

import logging


def configure_logging() -> None:
    """Configures a conservative default logger for local development."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
