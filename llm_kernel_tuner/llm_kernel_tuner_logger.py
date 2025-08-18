import logging

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    logger = logging.getLogger(name)
    
    # Only add NullHandler if no handlers are present
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger