import os
import sys
from loguru import logger as loguru_logger


class LoggingService:
    _instance = None
    _initialized = False

    def __new__(cls, log_level="INFO"):
        """Ensure only one instance of LoggingService exists (singleton pattern)"""
        if cls._instance is None:
            cls._instance = super(LoggingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_level="INFO"):
        """Initialize the logger with the specified log level (only once)"""
        # Only initialize once, even if __init__ is called multiple times
        if LoggingService._initialized:
            return
        
        # Create logs directory if it doesn't exist
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up logger
        self.logger = self._setup_logger(log_level)
        
        # Mark as initialized
        LoggingService._initialized = True

    def _setup_logger(self, log_level):
        """Configure and return a logger with file and console handlers"""
        # Remove default handler
        loguru_logger.remove()

        # Console handler with auto-flush (perfect for notebooks)
        loguru_logger.add(
            sys.stdout,
            level=log_level,
            colorize=True,
            enqueue=False,  # Synchronous for immediate output
        )

        # File handler with date-based rotation
        loguru_logger.add(
            os.path.join(self.log_dir, "{time:YYYY-MM-DD}.log"),
            level=log_level,
            rotation="00:00",  # Rotate at midnight
            encoding="utf-8",
            enqueue=False,  # Synchronous for immediate flush
        )

        return loguru_logger

    def info(self, message):
        """Log an informational message"""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)

    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        """Log an error message, optionally with exception info"""
        if exc_info:
            self.logger.exception(message)
        else:
            self.logger.error(message)

    def critical(self, message, exc_info=False):
        """Log a critical message, optionally with exception info"""
        if exc_info:
            self.logger.exception(message)
        else:
            self.logger.critical(message)


# Create a singleton instance
logger = LoggingService()
