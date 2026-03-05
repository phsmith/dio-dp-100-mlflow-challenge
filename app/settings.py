import logging
import os

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOGGER_NAME = "ice_cream_sales"
_CONFIGURED = False
logger = logging.getLogger(LOGGER_NAME)


def _resolve_log_level(level: str | None) -> int:
    level_name = (level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)).upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure root logging once for all project scripts."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=_resolve_log_level(level),
        format=fmt or os.getenv("LOG_FORMAT", DEFAULT_LOG_FORMAT),
    )
    _CONFIGURED = True


# Configure logging once when this module is imported.
configure_logging()
