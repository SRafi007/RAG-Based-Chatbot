#app/utils/logger.py
"""Logger configuration for the application."""
import logging
import os
from app.config.settings import settings

# Ensure logs directory exists
log_dir = os.path.dirname(settings.log_file)
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
logger = logging.getLogger(settings.app_name)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

# File handler - logs to file
file_handler = logging.FileHandler(settings.log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Console handler - optional for local dev
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Log format
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attach handlers (avoid duplicates)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Usage helper
def get_logger(name: str = None):
    """Return a child logger with the same config."""
    return logger.getChild(name) if name else logger

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Server started successfully")
    logger.error("Redis connection failed")
