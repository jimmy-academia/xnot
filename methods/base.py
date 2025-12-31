#!/usr/bin/env python3
"""Abstract base class for all evaluation methods."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMethod(ABC):
    """Abstract base class for evaluation methods."""

    name: str = "base"

    def __init__(self, run_dir: str = None, defense: bool = False, **kwargs):
        self.run_dir = run_dir
        self.defense = defense

    @abstractmethod
    def evaluate(self, query: Any, context: str) -> int:
        """Evaluate item against request.

        Args:
            query: Restaurant data (str or dict)
            context: User request text

        Returns:
            1 (recommend), 0 (neutral), -1 (not recommend)
        """
        pass

    def __call__(self, query: Any, context: str) -> int:
        """Allow method to be called as function."""
        return self.evaluate(query, context)

    def __repr__(self) -> str:
        """Show method info in magenta color."""
        info = f"{self.__class__.__name__}(defense={self.defense}"
        if self.run_dir:
            info += f", run_dir={self.run_dir}"
        info += ")"
        return f"\033[35m{info}\033[0m"  # Magenta
