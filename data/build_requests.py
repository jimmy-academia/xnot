#!/usr/bin/env python3
"""Convert requests_{n}.json to requests_{n}.jsonl.

Usage:
    python data/build_requests.py 1        # converts requests_1.json → requests_1.jsonl
    python data/build_requests.py 1 2 3    # converts multiple
    python data/build_requests.py          # converts all found requests_*.json files
"""

import json
import sys
from pathlib import Path

YELP_DIR = Path(__file__).parent / "yelp"


def convert_json_to_jsonl(n: str):
    """Convert requests_{n}.json to requests_{n}.jsonl."""
    json_path = YELP_DIR / f"requests_{n}.json"
    jsonl_path = YELP_DIR / f"requests_{n}.jsonl"

    if not json_path.exists():
        print(f"  [skip] {json_path.name} not found")
        return False

    # Read and parse concatenated JSON objects
    content = json_path.read_text()
    requests = []

    # Parse multiple JSON objects (not a JSON array), skipping // comments
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # Skip whitespace and comment lines
        while idx < len(content):
            # Skip whitespace
            while idx < len(content) and content[idx] in ' \t\n\r':
                idx += 1
            if idx >= len(content):
                break
            # Skip // comment lines (can be multiple consecutive)
            if content[idx:idx+2] == '//':
                end_of_line = content.find('\n', idx)
                idx = end_of_line + 1 if end_of_line != -1 else len(content)
            else:
                break  # Found non-whitespace, non-comment content

        if idx >= len(content):
            break

        # Decode next object
        obj, end_idx = decoder.raw_decode(content, idx)
        requests.append(obj)
        idx = end_idx

    # Write JSONL
    with open(jsonl_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req, separators=(',', ':')) + '\n')

    print(f"  {json_path.name} → {jsonl_path.name} ({len(requests)} requests)")
    return True


def main():
    if len(sys.argv) > 1:
        # Convert specified selection numbers
        nums = sys.argv[1:]
    else:
        # Find all requests_*.json files
        nums = []
        for p in YELP_DIR.glob("requests_*.json"):
            n = p.stem.replace("requests_", "")
            nums.append(n)
        nums.sort()

    if not nums:
        print("No requests_*.json files found")
        return

    print(f"Converting {len(nums)} file(s)...")
    for n in nums:
        convert_json_to_jsonl(n)


if __name__ == "__main__":
    main()
