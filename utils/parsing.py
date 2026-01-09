#!/usr/bin/env python3
"""Parsing utilities for script, answer, and index extraction."""

import re
import ast
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


MAX_SUBSTITUTION_CHARS = 8000  # ~2k tokens max per substitution


def substitute_variables(instruction: str, items, user_query: str, cache: dict) -> str:
    """Substitute {(var)}[key][index] patterns with actual values.

    Supports hierarchical step IDs like {(2.rev.0)} and {(final)}.

    Args:
        instruction: Template string with {(var)} patterns
        items: Item data dict (for {(input)} or {(items)} access)
        user_query: User's request text (for {(query)} substitution)
        cache: Cache of previous step outputs
    """
    pattern = r'\{\(([a-zA-Z0-9_.]+)\)\}((?:\[[^\]]+\])*)'

    def _sub(match):
        var = match.group(1)
        accessors = match.group(2) or ''

        # Get base value
        if var == 'query':
            val = user_query
        elif var in ('input', 'items'):
            val = items
        elif var == 'context':
            # context maps to items (restaurant data)
            val = items
        else:
            val = cache.get(var, '')

        # Try to parse string as literal if needed
        if isinstance(val, str) and accessors:
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (dict, list, tuple)):
                    val = parsed
            except Exception:
                pass

        # Apply accessors [key], [index], or [start:end] slices
        for acc in re.findall(r'\[([^\]]+)\]', accessors):
            try:
                # Check for slice notation (contains ':')
                if ':' in acc:
                    parts = acc.split(':')
                    if len(parts) == 2:
                        start = int(parts[0]) if parts[0] else None
                        end = int(parts[1]) if parts[1] else None
                        if isinstance(val, (str, list, tuple)):
                            val = val[start:end]
                        else:
                            val = ''
                    else:
                        val = ''
                elif isinstance(val, dict):
                    val = val.get(acc, val.get(int(acc)) if acc.isdigit() else '')
                elif isinstance(val, (list, tuple)) and acc.isdigit():
                    idx = int(acc)
                    val = val[idx] if 0 <= idx < len(val) else ''
                elif isinstance(val, str) and acc.isdigit():
                    # Support string indexing like [text][0] - though unusual
                    idx = int(acc)
                    val = val[idx] if 0 <= idx < len(val) else ''
                else:
                    val = ''
            except Exception:
                val = ''

        # Return as string with size limit
        if isinstance(val, (dict, list, tuple)):
            result = json.dumps(val)
        else:
            result = str(val)

        # Truncate if too large (prevents prompt explosion)
        if len(result) > MAX_SUBSTITUTION_CHARS:
            # Keep start and end for context
            keep = MAX_SUBSTITUTION_CHARS // 2
            result = result[:keep] + "\n...[TRUNCATED]...\n" + result[-keep//2:]

        return result

    return re.sub(pattern, _sub, instruction)


def parse_script(script: str) -> list:
    """Parse script into [(index, instruction), ...].

    Handles various formats:
    - (0)=LLM("instruction")
    - (0) = LLM("instruction")
    - (0)=LLM('instruction')
    - (2.rev.0)=LLM("instruction")  # Hierarchical step IDs
    - (final)=LLM("instruction")    # Named step IDs
    - Multi-line instructions
    """
    steps = []
    for line in script.split('\n'):
        # Skip lines without LLM pattern (allow spaces around = and after LLM)
        if not re.search(r'=\s*LLM\s*\(', line):
            continue

        # Match step index: (0), (final), (2.rev.0), etc. with optional spaces
        # Supports: numeric, alphanumeric, and dot-separated hierarchical IDs
        idx_match = re.search(r'\(([a-zA-Z0-9_.]+)\)\s*=\s*LLM', line)
        if not idx_match:
            continue

        # Match instruction in quotes (single or double)
        # Handle both LLM("...") and LLM('...')
        instr_match = re.search(r'LLM\s*\(\s*["\'](.+?)["\']\s*\)', line, re.DOTALL)
        if instr_match:
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


def parse_limit_spec(spec: str) -> list[int]:
    """Parse limit specification into list of request indices.

    User input is 1-indexed (matching request IDs R01, R11, etc.).
    Returns 0-indexed for internal use.

    Args:
        spec: Limit specification string (1-indexed)

    Returns:
        Sorted list of request indices (0-based internally)

    Examples:
        '5' -> [0,1,2,3,4]  (first 5 requests)
        '1-10' -> [0,1,2,3,4,5,6,7,8,9]  (R01-R10)
        '1,5,10' -> [0,4,9]  (R01, R05, R10)
        '11,12,13' -> [10,11,12]  (R11, R12, R13)
    """
    if not spec or not spec.strip():
        return []

    spec = spec.strip()
    indices = set()

    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            # Range: "11-15" means R11-R15 -> indices 10-14
            try:
                start, end = part.split('-', 1)
                # Convert 1-indexed to 0-indexed
                indices.update(range(int(start) - 1, int(end)))
            except ValueError:
                logger.warning(f"Invalid range in limit spec: {part}")
        else:
            # Single number: could be count or index
            try:
                n = int(part)
                # If it's a single number with no comma/range, treat as count (first N)
                if ',' not in spec and '-' not in spec:
                    indices.update(range(n))
                else:
                    # Specific index: 1-indexed input -> 0-indexed internal
                    indices.add(n - 1)
            except ValueError:
                logger.warning(f"Invalid number in limit spec: {part}")

    return sorted(indices)


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


def parse_numbered_steps(response: str, max_steps: int = 3) -> list:
    """Parse numbered steps from LLM response.

    Extracts step content from lines starting with digits (e.g., "1. Step", "2) Step").
    Strips the number prefix and returns the step text.

    Args:
        response: LLM response text containing numbered steps
        max_steps: Maximum number of steps to return (default: 3)

    Returns:
        List of step strings (up to max_steps)
    """
    lines = response.strip().split('\n')
    steps = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Strip number prefix like "1.", "2)", "3:"
            for i, ch in enumerate(line):
                if ch in '.):' and i < 3:
                    line = line[i+1:].strip()
                    break
        if line and len(line) > 5:
            steps.append(line)
    return steps[:max_steps]
