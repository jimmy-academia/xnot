#!/usr/bin/env python3
"""Shared defense configuration for methods."""

# =============================================================================
# Defense Configuration (shared across cot.py, listwise.py)
# =============================================================================

_defense = None
_use_defense_prompt = False


def set_defense_mode(enabled: bool):
    """Toggle between normal and defense prompts."""
    global _use_defense_prompt
    _use_defense_prompt = enabled


def set_defense(defense_concept: str):
    """Enable defense prompt."""
    global _defense
    _defense = defense_concept
