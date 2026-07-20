from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
THIRD_PARTY_LOGGERS = ("httpcore", "httpx", "neo4j")


def configure_logging(
    log_level: str | int = logging.INFO,
    *,
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure application logging without replacing host-managed handlers."""
    resolved_level = _resolve_log_level(log_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(stream_handler)

    if log_file is not None:
        resolved_log_file = log_file.resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(root_logger, resolved_log_file):
            file_handler = RotatingFileHandler(
                resolved_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
            root_logger.addHandler(file_handler)

    for logger_name in THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def close_log_file(log_file: Path) -> None:
    """Close and detach a configured file handler."""
    root_logger = logging.getLogger()
    resolved_log_file = log_file.resolve()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_log_file:
            root_logger.removeHandler(handler)
            handler.close()


def _resolve_log_level(log_level: str | int) -> int:
    if isinstance(log_level, int):
        return log_level
    normalized_level = log_level.strip().upper()
    resolved_level = logging.getLevelNamesMapping().get(normalized_level)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unknown log level: {log_level}")
    return resolved_level


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    return any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_file
        for handler in logger.handlers
    )
