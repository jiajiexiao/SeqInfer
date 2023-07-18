import logging
from enum import Enum


class LogLevel(int, Enum):
    """Int Enumeration for log levels.

    The log levels are represented as integers. They correspond to the standard
    logging module levels:
    - NOTSET: 0
    - DEBUG: 10
    - INFO: 20
    - WARNING: 30
    - ERROR: 40
    - CRITICAL: 50
    """

    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def get_logger(name: str, logger_level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """Get a logger with the given name and level.

    Args:
        name (str): The name of the logger.
        logger_level (LogLevel, optional): The log level for the logger. Defaults to LogLevel.INFO.

    Returns:
        logging.Logger: The logger object.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    # create formatter
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s: %(message)s")

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger_level)

    # add formatter to handler
    console_handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(console_handler)

    return logger


def log_msg(
    name: str,
    msg: str,
    log_level: LogLevel = LogLevel.INFO,
    logger_level: LogLevel = LogLevel.INFO,
) -> None:
    """Log a message with the specified log level to the logger.

    Args:
        name (str): The name of the logger.
        msg (str): The message to be logged.
        log_level (LogLevel, optional): The log level for the message. Defaults to LogLevel.INFO.
        logger_level (LogLevel, optional): The log level for the logger. Defaults to LogLevel.INFO.
    """
    logger = get_logger(name, logger_level)
    logger.log(log_level, msg)
