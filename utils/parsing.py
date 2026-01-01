#!/usr/bin/env python3
"""Parsing utilities for script, answer, and index extraction."""

import re
import ast
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def substitute_variables(instruction: str, query, context: str, cache: dict) -> str:
    """Substitute {(var)}[key][index] patterns with actual values."""
    pattern = r'\{\((\w+)\)\}((?:\[[^\]]+\])*)'

    def _sub(match):
        var = match.group(1)
        accessors = match.group(2) or ''

        # Get base value
        if var == 'input':
            val = query
        elif var == 'context':
            val = context
        else:
            val = cache.get(var, '')

        # Try to parse string as literal if needed
        if isinstance(val, str) and accessors:
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (dict, list, tuple)):
                    val = parsed
            except:
                pass

        # Apply accessors [key] or [index]
        for acc in re.findall(r'\[([^\]]+)\]', accessors):
            try:
                if isinstance(val, dict):
                    val = val.get(acc, val.get(int(acc)) if acc.isdigit() else '')
                elif isinstance(val, (list, tuple)) and acc.isdigit():
                    idx = int(acc)
                    val = val[idx] if 0 <= idx < len(val) else ''
                else:
                    val = ''
            except:
                val = ''

        # Return as string
        if isinstance(val, (dict, list, tuple)):
            return json.dumps(val)
        return str(val)

    return re.sub(pattern, _sub, instruction)


def parse_script(script: str) -> list:
    """Parse script into [(index, instruction), ...]."""
    steps = []
    for line in script.split('\n'):
        if '=LLM(' not in line:
            continue
        idx_match = re.search(r'\((\d+)\)\s*=\s*LLM', line)
        instr_match = re.search(r'LLM\(["\'](.+?)["\']\)', line, re.DOTALL)
        if idx_match and instr_match:
            steps.append((idx_match.group(1), instr_match.group(1)))
    return steps


def parse_final_answer(output: str) -> int:
    """Parse output to -1, 0, or 1.

    Returns 0 (neutral) if parsing fails, with a warning logged.
    """
    output = output.strip()
    if output in ["-1", "0", "1"]:
        return int(output)

    # Pattern: ANSWER: X format (from cot.py)
    match = re.search(r'(?:ANSWER|Answer|FINAL ANSWER|Final Answer):\s*(-?[01])', output, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Pattern: Standalone number
    match = re.search(r'(?:^|[:\s])(-1|0|1)(?:\s|$|\.)', output)
    if match:
        return int(match.group(1))

    lower = output.lower()
    # Handle POSITIVE/NEGATIVE/NEUTRAL from script intermediate steps
    if "negative" in lower:
        return -1
    if "positive" in lower:
        return 1
    if "neutral" in lower:
        return 0
    if "not recommend" in lower:
        return -1
    if "recommend" in lower and "not" not in lower:
        return 1

    # Could not parse - log warning and return neutral
    logger.warning(f"Could not parse answer from response: {output[:100]}...")
    return 0


def normalize_pred(raw: Any) -> int:
    """Normalize prediction to {-1, 0, 1}.

    Handles int, bool, float, and str inputs.
    For strings, delegates to parse_final_answer().
    """
    if raw is None:
        raise ValueError("Prediction is None")
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw in {-1, 0, 1}:
            return raw
        raise ValueError(f"Invalid int: {raw}")
    if isinstance(raw, bool):
        return 1 if raw else -1
    if isinstance(raw, float):
        return -1 if raw <= -0.5 else (1 if raw >= 0.5 else 0)
    if isinstance(raw, str):
        return parse_final_answer(raw)
    raise ValueError(f"Cannot normalize: {repr(raw)}")


def parse_index(response: str, max_index: int = 20) -> int:
    """Parse LLM response to extract item index (1 to max_index).

    Args:
        response: LLM response text
        max_index: Maximum valid index

    Returns:
        Index (1-based) or 0 if parsing fails
    """
    if response is None:
        return 0

    # Try to find a number 1-max_index in the response
    # First try: exact match at start
    match = re.match(r'^\s*(\d+)', str(response))
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    # Second try: find any number in brackets like [5] or (5)
    match = re.search(r'[\[\(](\d+)[\]\)]', str(response))
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    # Third try: find standalone number
    for match in re.finditer(r'\b(\d+)\b', str(response)):
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    return 0  # Failed to parse


def parse_indices(response: str, max_index: int = 20, k: int = 5) -> list[int]:
    """Parse LLM response to extract up to k item indices.

    Args:
        response: LLM response text
        max_index: Maximum valid index
        k: Maximum number of indices to extract

    Returns:
        List of indices (1-based), up to k unique items
    """
    if response is None:
        return []

    indices = []
    for match in re.finditer(r'\b(\d+)\b', str(response)):
        idx = int(match.group(1))
        if 1 <= idx <= max_index and idx not in indices:
            indices.append(idx)
            if len(indices) >= k:
                break
    return indices
