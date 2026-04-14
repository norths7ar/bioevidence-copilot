import logging
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str,
    *,
    log_file: Path | None = None,
    log_dir: Path | None = None,
    log_level: int = logging.INFO,
    with_date: bool = False,
) -> logging.Logger:
    """Return a logger with optional explicit file logging.

    The helper does not infer any log directory from the caller. Pass
    `log_file` or `log_dir` explicitly when file logging is desired.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    resolved_log_file = log_file
    if resolved_log_file is None and log_dir is not None:
        filename = name
        if with_date:
            today = datetime.today().strftime("%Y-%m-%d")
            filename = f"{filename}_{today}"
        resolved_log_file = Path(log_dir) / f"{filename}.log"

    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
