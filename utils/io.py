"""File I/O utilities for reading and writing various formats."""

import json
import re
import argparse
from pathlib import Path


class NamespaceEncoder(json.JSONEncoder):
    """JSON encoder that handles argparse.Namespace objects."""
    def default(self, obj):
        if isinstance(obj, argparse.Namespace):
            return obj.__dict__
        return super().default(obj)


def good_json_dump(dictionary):
    """Dump dictionary to compact, readable JSON string."""
    obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
    obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
    obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
    obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
    return obj


def dumpj(filepath, dictionary):
    """Write dictionary to JSON file with nice formatting."""
    with open(filepath, "w") as f:
        obj = good_json_dump(dictionary)
        f.write(obj)


def loadj(filepath):
    """Load JSON file and return contents."""
    with open(filepath) as f:
        return json.load(f)


def loadjl(filepath):
    """Load JSON Lines file and return list of items."""
    filepath = Path(filepath)
    items = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):  # Skip empty lines and comments
                continue
            items.append(json.loads(line))
    return items
