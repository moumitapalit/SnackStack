import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler("app.log")
        formatter = logging.Formatter("%(asctime)s | %(name)18s | %(levelname)7s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger