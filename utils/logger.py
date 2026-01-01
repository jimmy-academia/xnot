"""
Logging utilities for xnot evaluation framework.

This module re-exports from simple_logger and debug_logger for backwards compatibility.
"""

# Re-export from simple_logger
from .simple_logger import (
    LOG_THEME,
    console,
    LogLevel,
    SimpleLogger,
    logger,
    setup_logger_level,
)

# Re-export from debug_logger
from .debug_logger import (
    DebugLogger,
    consolidate_logs,
)

__all__ = [
    'LOG_THEME',
    'console',
    'LogLevel',
    'SimpleLogger',
    'logger',
    'setup_logger_level',
    'DebugLogger',
    'consolidate_logs',
]
