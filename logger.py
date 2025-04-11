"""
Logging configuration for the PDF RAG application.
"""

import logging
import os
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger with the given name.
    
    Args:
        name: The name of the logger, typically __name__ from the calling module
        
    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level from config
    level = getattr(logging, LOG_LEVEL)
    logger.setLevel(level)
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
