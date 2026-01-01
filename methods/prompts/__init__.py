"""Shared prompts for evaluation methods."""

from .common import (
    DEFENSE_PREAMBLE,
    DEFENSE_PREAMBLE_SHORT,
    with_defense,
    get_defense_system_prompt,
)

__all__ = [
    'DEFENSE_PREAMBLE',
    'DEFENSE_PREAMBLE_SHORT',
    'with_defense',
    'get_defense_system_prompt',
]
