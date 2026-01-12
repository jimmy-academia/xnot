"""
Utility functions for General ANoT.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict


def compact_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Pretty JSON with compact arrays/short values on single lines.

    - Objects get indented normally
    - Arrays of primitives stay on one line: ["a", "b", "c"]
    - Short key-value pairs stay readable
    """
    raw = json.dumps(obj, indent=indent, ensure_ascii=False, default=str)

    # Collapse simple arrays onto single lines
    # Match: [\n  "item1",\n  "item2"\n]
    raw = re.sub(r'("|\d+),\s+', r'\1, ', raw)
    raw = re.sub(r'\[\n\s*("|\d+)', r'[\1', raw)
    raw = re.sub(r'("|\d+)\n\s*\]', r'\1]', raw)

    return raw


def get_next_run_number(results_dir: Path) -> int:
    """Get next run number by scanning existing directories."""
    if not results_dir.exists():
        return 1

    max_num = 0
    for path in results_dir.iterdir():
        if path.is_dir():
            # Match patterns like "001_..." or "42_..."
            match = re.match(r'^(\d+)_', path.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

    return max_num + 1
