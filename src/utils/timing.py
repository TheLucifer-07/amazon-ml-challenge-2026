"""Execution timing context manager and decorator."""

import time
from collections.abc import Generator
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger("timing")


@contextmanager
def timer(description: str) -> Generator[None, None, None]:
    """Log elapsed wall-clock time for a block of code."""
    start_time = time.perf_counter()
    logger.info(f"Started: {description}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"Finished: {description} in {elapsed:.2f}s ({elapsed / 60:.2f}m)")
