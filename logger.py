"""
Logger configuration for Multilingual HTML Translator
Original site: https://hitdarderut-haaretz.org
Translated versions: https://degeneration-of-nation.org

Handles logging setup and configuration while maintaining original functionality
"""

import logging
import sys
from config import LOGGING_CONFIG

def setup_logger(language_code: str = None) -> logging.Logger:
    """
    Creates and configures the logger with both file and console handlers
    Maintains the exact logging behavior of the original implementation
    """
    # Create logger
    logger = logging.getLogger('website_translator')
    logger.setLevel(LOGGING_CONFIG['level'])
    logger.propagate = False
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        LOGGING_CONFIG['format'],
        LOGGING_CONFIG['datefmt']
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if language code is provided)
    if language_code:
        file_handler = logging.FileHandler(f"{LOGGING_CONFIG['log_file_suffix']}_{language_code}.txt")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger

def set_verbose_mode(logger: logging.Logger, verbose: bool) -> None:
    """Sets console handler level based on verbose mode"""
    level = logging.INFO if verbose else logging.WARNING
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)