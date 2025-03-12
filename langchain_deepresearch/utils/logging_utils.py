"""
Logging utilities for DeepResearch.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any


def setup_research_logger(
        config: Dict[str, Any],
        log_dir: Optional[str] = None,
        log_prefix: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger for research sessions.

    Args:
        config: Configuration dictionary
        log_dir: Directory for log files (default: ./logs)
        log_prefix: Prefix for log files (default: research)

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("deep_research")

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Get log level from config
    log_level_name = config["logging"]["level"]
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(log_level)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_dir is specified
    if log_dir:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Define log file
        prefix = log_prefix or "research"
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def log_research_start(
        logger: logging.Logger, query: str, breadth: int, depth: int, time_limit: Optional[int] = None
) -> None:
    """
    Log the start of a research session.

    Args:
        logger: Logger instance
        query: Research query
        breadth: Number of parallel searches
        depth: Depth of recursive research
        time_limit: Maximum research time in seconds
    """
    logger.info("=" * 80)
    logger.info(f"Starting new research session")
    logger.info(f"Query: {query}")
    logger.info(f"Settings: Breadth={breadth}, Depth={depth}, Time limit={time_limit or 'default'}")
    logger.info("=" * 80)


def log_research_end(
        logger: logging.Logger,
        success: bool,
        elapsed_time: float,
        num_sources: int,
        num_learnings: int,
        early_completion: bool = False,
        error: Optional[str] = None
) -> None:
    """
    Log the end of a research session.

    Args:
        logger: Logger instance
        success: Whether the research was successful
        elapsed_time: Time elapsed in seconds
        num_sources: Number of sources consulted
        num_learnings: Number of learnings gathered
        early_completion: Whether research completed early
        error: Error message if research failed
    """
    logger.info("=" * 80)
    if success:
        logger.info(f"Research completed in {elapsed_time:.1f} seconds")
        logger.info(f"Consulted {num_sources} sources")
        logger.info(f"Gathered {num_learnings} key insights")
        if early_completion:
            logger.info("Research completed early with sufficient findings")
    else:
        logger.error(f"Research failed: {error}")
    logger.info("=" * 80)
