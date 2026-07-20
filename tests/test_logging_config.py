from __future__ import annotations

import logging

import pytest

from bioevidence.utils.logging_config import (
    DEFAULT_LOG_FORMAT,
    THIRD_PARTY_LOGGERS,
    close_log_file,
    configure_logging,
)


@pytest.fixture
def restore_logging_state():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_root_level = root_logger.level
    original_third_party_levels = {
        logger_name: logging.getLogger(logger_name).level for logger_name in THIRD_PARTY_LOGGERS
    }
    root_logger.handlers.clear()
    try:
        yield root_logger
    finally:
        for handler in root_logger.handlers:
            if handler not in original_handlers:
                handler.close()
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_root_level)
        for logger_name, level in original_third_party_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def test_configure_logging_sets_root_and_quiets_third_party_loggers(restore_logging_state) -> None:
    root_logger = restore_logging_state
    root_logger.handlers.clear()

    configure_logging("DEBUG")

    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0].formatter is not None
    assert root_logger.handlers[0].formatter._fmt == DEFAULT_LOG_FORMAT
    assert all(logging.getLogger(name).level == logging.WARNING for name in THIRD_PARTY_LOGGERS)


def test_configure_logging_is_idempotent(restore_logging_state) -> None:
    root_logger = restore_logging_state
    root_logger.handlers.clear()

    configure_logging("INFO")
    configure_logging(logging.ERROR)

    assert root_logger.level == logging.ERROR
    assert len(root_logger.handlers) == 1


def test_configure_logging_rejects_unknown_level(restore_logging_state) -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("verbose")


def test_configure_logging_writes_and_closes_rotating_file(tmp_path, restore_logging_state) -> None:
    root_logger = restore_logging_state
    root_logger.handlers.clear()
    log_file = tmp_path / "run.log"

    configure_logging("INFO", log_file=log_file, max_bytes=1024, backup_count=1)
    logging.getLogger("bioevidence.test").info("event_count=%d", 3)
    close_log_file(log_file)

    assert "event_count=3" in log_file.read_text(encoding="utf-8")
    assert all(
        not isinstance(handler, logging.FileHandler) or handler.baseFilename != str(log_file.resolve())
        for handler in root_logger.handlers
    )
