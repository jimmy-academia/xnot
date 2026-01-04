#!/usr/bin/env python3
"""Phase 2 tools for LWT manipulation and data access."""

import json
import re
from typing import List


def tool_read(path: str, data: dict) -> str:
    """Read full value at path."""
    def resolve_path(p: str, d):
        if not p:
            return d
        # Parse path like items[2].item_data[0].review
        parts = re.split(r'\.|\[|\]', p)
        parts = [x for x in parts if x]
        val = d
        for part in parts:
            try:
                if isinstance(val, list) and part.isdigit():
                    val = val[int(part)]
                elif isinstance(val, dict):
                    val = val.get(part)
                else:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
        return val

    result = resolve_path(path, data)
    if result is None:
        return f"Error: path '{path}' not found"
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def tool_lwt_list(lwt_steps: List[str]) -> str:
    """Show current LWT steps with indices."""
    if not lwt_steps:
        return "(empty)"
    lines = []
    for i, step in enumerate(lwt_steps):
        lines.append(f"{i}: {step}")
    return "\n".join(lines)


def tool_lwt_get(idx: int, lwt_steps: List[str]) -> str:
    """Get step at index."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    return lwt_steps[idx]


def tool_lwt_set(idx: int, step: str, lwt_steps: List[str]) -> str:
    """Replace step at index. Returns status."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    lwt_steps[idx] = step
    return f"Replaced step at index {idx}"


def tool_lwt_delete(idx: int, lwt_steps: List[str]) -> str:
    """Delete step at index. Returns status."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    lwt_steps.pop(idx)
    return f"Deleted step at index {idx}"


def tool_lwt_insert(idx: int, step: str, lwt_steps: List[str]) -> str:
    """Insert step at index (shifts others down). Returns status."""
    if idx < 0 or idx > len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)})"
    lwt_steps.insert(idx, step)
    return f"Inserted at index {idx}"
