"""File I/O utilities for reading and writing various formats."""

import json
import pickle
import re
import argparse
from pathlib import Path


def readf(path, encoding="utf-8"):
    """Read text file and return contents."""
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def writef(path, content, encoding="utf-8"):
    """Write content to text file."""
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


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


def dumpp(filepath, obj):
    """Pickle object to file."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def loadp(filepath):
    """Load pickled object from file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
