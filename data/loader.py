#!/usr/bin/env python3
"""Data loading and formatting utilities."""

import json
from pathlib import Path
from typing import Any, Union

ATTACKED_DIR = Path("data/attacked")
SELECTIONS_PATH = Path("data/selections.jsonl")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DEFAULT_SELECTION = "v1_basic"


# --- Data Generation from Selections ---

def load_selections() -> dict:
    """Load all selections from selections.jsonl, keyed by id."""
    selections = {}
    if SELECTIONS_PATH.exists():
        with open(SELECTIONS_PATH) as f:
            for line in f:
                if line.strip():
                    sel = json.loads(line)
                    selections[sel["id"]] = sel
    return selections


def generate_from_selection(selection_id: str, output_path: str) -> None:
    """Generate processed data from a hardcoded selection.

    Args:
        selection_id: ID of selection in selections.jsonl
        output_path: Where to write the output JSONL
    """
    selections = load_selections()
    if selection_id not in selections:
        available = list(selections.keys())
        raise ValueError(f"Selection '{selection_id}' not found. Available: {available}")

    selection = selections[selection_id]
    print(f"Generating data from selection: {selection_id}")
    print(f"  Description: {selection.get('description', 'N/A')}")

    restaurant_ids = selection.get("restaurant_ids", [])
    review_ids = selection.get("review_ids_per_restaurant", {})

    if not restaurant_ids:
        print("  Warning: No restaurant_ids in selection - cannot generate")
        raise ValueError(f"Selection '{selection_id}' has no restaurant_ids")

    # TODO: Load from raw Yelp data based on selection IDs
    items = []
    # ... implementation in Stage 2

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')

    print(f"  Generated {len(items)} items to {output_path}")


def _ensure_data_exists(path: str, selection_id: str = None) -> None:
    """Check if data exists, generate from selection if not."""
    if Path(path).exists():
        return

    selection_id = selection_id or DEFAULT_SELECTION
    print(f"Data not found at {path}")
    print(f"Generating from selection: {selection_id}")
    generate_from_selection(selection_id, path)


DEFAULT_REQUESTS = [
    {"id": "R0", "context": "I'm in a hurry and need quick service. Is the wait time reasonable?"},
    {"id": "R1", "context": "I've heard mixed things. Is this place consistent in quality?"},
    {"id": "R2", "context": "Planning a special dinner date. Good for romantic occasions?"},
    {"id": "R3", "context": "Is it worth the price? Looking for good value, not necessarily cheap."},
    {"id": "R4", "context": "I care more about food quality than service. How's the food?"},
]


def load_requests(path: str = "requests.json") -> list[dict]:
    """Load user requests from JSON file.

    Supports both simple format (id, context) and complex format (id, text, structure).
    For complex format, 'text' is used as 'context'.
    """
    try:
        with open(path) as f:
            requests = json.load(f)
            # Normalize complex format to have 'context' field
            for req in requests:
                if "text" in req and "context" not in req:
                    req["context"] = req["text"]
            return requests
    except FileNotFoundError:
        return DEFAULT_REQUESTS


def load_data(path: str, limit: int = None, attack: str = "none") -> Union[dict, list]:
    """Load data, optionally with attack variant(s).

    Args:
        path: Path to clean data file
        limit: Max items to load
        attack: "none"/"clean" for original, specific name, or "all" for all attacks

    Returns:
        - dict {attack_name: items} when attack="all"
        - list[dict] for single attack/clean
    """
    # Ensure data exists (generate from selection if not)
    _ensure_data_exists(path)

    def _load_jsonl(filepath: str, limit: int = None) -> list[dict]:
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
                    if limit and len(items) >= limit:
                        break
        return items

    # Load all attacks
    if attack == "all":
        result = {"clean": _load_jsonl(path, limit)}
        for attack_file in ATTACKED_DIR.glob("*.jsonl"):
            result[attack_file.stem] = _load_jsonl(str(attack_file), limit)
        return result

    # Load clean data
    if attack in ("none", "clean", None):
        return _load_jsonl(path, limit)

    # Load specific attack
    attack_path = ATTACKED_DIR / f"{attack}.jsonl"
    if not attack_path.exists():
        raise FileNotFoundError(f"Attacked data not found: {attack_path}")
    return _load_jsonl(str(attack_path), limit)


def format_query(item: dict, mode: str = "string"):
    """Format restaurant item as query. Excludes ground-truth labels.

    Returns: (query, review_count)
    """
    reviews = item.get("item_data", [])

    if mode == "dict":
        # Return clean dict for structured access
        return {
            "item_name": item.get("item_name", "Unknown"),
            "item_id": item.get("item_id", "unknown"),
            "city": item.get("city", "Unknown"),
            "neighborhood": item.get("neighborhood", "Unknown"),
            "price_range": item.get("price_range", "Unknown"),
            "cuisine": item.get("cuisine", []),
            "item_data": [{"review_id": r.get("review_id", ""), "review": r.get("review", ""),
                          "stars": r.get("stars", 0)}
                          for r in reviews]
        }, len(reviews)

    # String mode (default)
    parts = [
        "Restaurant:",
        f"Name: {item.get('item_name', 'Unknown')}",
        f"City: {item.get('city', 'Unknown')}",
        f"Neighborhood: {item.get('neighborhood', 'Unknown')}",
        f"Price: {item.get('price_range', 'Unknown')}",
        f"Cuisine: {', '.join(item.get('cuisine', [])) or 'Unknown'}",
        "",
        "Reviews:",
    ]
    for r in reviews:
        parts.append(f"[{r.get('review_id', 'unknown')}] {r.get('review', '')}")
    return "\n".join(parts), len(reviews)


def normalize_pred(raw: Any) -> int:
    """Normalize prediction to {-1, 0, 1}."""
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
        s = raw.strip().lower()
        if s in {"-1", "0", "1"}:
            return int(s)
        if any(p in s for p in ["not recommend", "don't recommend", "avoid", "reject"]):
            return -1
        if any(p in s for p in ["recommend", "yes", "suitable", "good", "great"]):
            return 1
        if any(p in s for p in ["neutral", "uncertain", "maybe", "mixed"]):
            return 0
        for tok in s.replace(",", " ").replace(":", " ").split():
            tok = tok.strip("()[]{}.,;:")
            if tok in {"-1", "0", "1"}:
                return int(tok)
    raise ValueError(f"Cannot normalize: {repr(raw)}")


def prepare_data(data_path: str, requests_path: str, limit: int = None) -> tuple[list, list]:
    """Load and prepare items and requests.

    Returns: (items, requests)
    """
    items = load_data(data_path, limit)
    requests = load_requests(requests_path)
    return items, requests


