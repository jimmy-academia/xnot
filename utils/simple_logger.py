"""Simple colored console logger using Rich."""

from datetime import datetime
from enum import Enum
from typing import Union

from rich.console import Console
from rich.theme import Theme


# Custom theme for log levels
LOG_THEME = Theme({
    "debug": "dim",
    "llm_input": "blue",
    "llm_output": "cyan",
    "info": "green",
    "warning": "yellow",
    "error": "red",
    "critical": "bold red on yellow",
})

console = Console(theme=LOG_THEME)


class LogLevel(Enum):
    """Log levels with priority values"""
    DEBUG = 10
    LLM_INPUT = 15
    LLM_OUTPUT = 16
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class SimpleLogger:
    """Simple logger class that supports colored terminal output via rich"""

    def __init__(
        self,
        name: str = "xnot",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        console_output: bool = True
    ):
        self.name = name
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value
        else:
            self.log_level = log_level
        self.console_output = console_output

    def _log(self, level: LogLevel, message: str) -> None:
        """Internal method to log messages at specified level"""
        if level.value < self.log_level:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        level_name = level.name
        style = level_name.lower()

        if self.console_output:
            console.print(f"[{style}][{timestamp}] {level_name}: {message}[/{style}]")

    def debug(self, message: str) -> None:
        self._log(LogLevel.DEBUG, message)

    def llm_input(self, message: str) -> None:
        self._log(LogLevel.LLM_INPUT, message)

    def llm_output(self, message: str) -> None:
        self._log(LogLevel.LLM_OUTPUT, message)

    def info(self, message: str) -> None:
        self._log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self._log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self._log(LogLevel.ERROR, message)

    def critical(self, message: str) -> None:
        self._log(LogLevel.CRITICAL, message)


# Module-level logger instance
logger = SimpleLogger()


def setup_logger_level(verbose: bool):
    """
    Configures the global logger based on the verbose flag.

    Args:
        verbose (bool): If True, set level to DEBUG. Otherwise, INFO.

    Returns:
        SimpleLogger: The configured logger instance
    """
    if verbose:
        logger.log_level = LogLevel.DEBUG.value
        logger.debug("Verbose mode enabled.")
    else:
        logger.log_level = LogLevel.INFO.value
    return logger
