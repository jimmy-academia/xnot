#!/usr/bin/env python3
"""Data loading and formatting utilities."""

import json
import sys
from pathlib import Path
from typing import Any, Union

DATA_DIR = Path(__file__).parent
ATTACKED_DIR = DATA_DIR / "attacked"
SELECTIONS_PATH = DATA_DIR / "selections.jsonl"
YELP_DIR = DATA_DIR / "yelp"


class Dataset:
    """Wrapper for loaded dataset with schema/stats."""

    def __init__(self, name: str, items: list, requests: list):
        self.name = name
        self.items = items
        self.requests = requests

    def _get_schema(self) -> str:
        """Extract nested schema from first item."""
        if not self.items:
            return "  (no items)"

        item = self.items[0]
        lines = ["  Item Schema:"]

        # Top-level keys (excluding item_data)
        top_keys = [k for k in item.keys() if k != "item_data"]
        lines.append(f"    {', '.join(top_keys)}")

        # item_data structure
        if "item_data" in item and item["item_data"]:
            review = item["item_data"][0]
            review_keys = [k for k in review.keys() if k != "user"]
            lines.append(f"    item_data[]:")
            lines.append(f"      {', '.join(review_keys)}")

            # user structure
            if "user" in review and review["user"]:
                user_keys = list(review["user"].keys())
                lines.append(f"      user: {', '.join(user_keys)}")

        return "\n".join(lines)

    def __repr__(self):
        total_reviews = sum(len(item.get("item_data", [])) for item in self.items)
        categories = set()
        for item in self.items:
            categories.update(item.get("categories", []))

        return (
            f"Dataset({self.name})\n"
            f"  Items: {len(self.items)}\n"
            f"  Requests: {len(self.requests)}\n"
            f"  Reviews: {total_reviews}\n"
            f"  Categories: {len(categories)}\n\n"
            f"{self._get_schema()}"
        )

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def resolve_dataset(name_or_path: str) -> tuple[Path, Path]:
    """Resolve dataset name to (data_path, requests_path).

    If name_or_path is a path (contains / or ends with .json/.jsonl), use as-is.
    Otherwise treat as dataset name and resolve to standard locations.
    """
    # Explicit path - use directly
    if "/" in name_or_path or name_or_path.endswith((".json", ".jsonl")):
        data_path = Path(name_or_path)
        # Derive requests from data path name
        requests_path = REQUESTS_DIR / f"{data_path.stem}.json"
    else:
        # Dataset name - resolve to standard locations
        data_path = PROCESSED_DIR / f"{name_or_path}.jsonl"
        requests_path = REQUESTS_DIR / f"{name_or_path}.json"

    return data_path, requests_path


def check_dataset_exists(data_path: Path, requests_path: Path, dataset_name: str) -> None:
    """Warn user if dataset files don't exist and exit."""
    missing = []
    if not data_path.exists():
        missing.append(f"  - Data: {data_path}")
    if not requests_path.exists():
        missing.append(f"  - Requests: {requests_path}")

    if missing:
        print(f"\n\u26a0\ufe0f  Dataset '{dataset_name}' not found:")
        print("\n".join(missing))
        print(f"\nTo create this dataset, run:")
        print(f"  1. python data/scripts/{dataset_name}_curation.py")
        print(f"  2. python data/scripts/{dataset_name}_review_sampler.py <selection_name>")
        print()
        sys.exit(1)

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

def load_dataset(data: str, selection_name: str = None, limit: int = None, attack: str = "none") -> Dataset:
    """Unified dataset loading - handles both selection-based and legacy datasets.

    Args:
        data: Dataset name (e.g., 'yelp') or explicit path to JSONL file
        selection_name: Selection name (e.g., 'selection_1') or None for legacy
        limit: Max items to load
        attack: Attack type for legacy datasets

    Returns: Dataset object
    """
    if data == 'yelp':
        items, requests = load_yelp_dataset(selection_name, limit)
        return Dataset(f"yelp/{selection_name}", items, requests)

    # Legacy dataset loading (fallback)
    raise ValueError(f"Unknown dataset: {data}")


# --- Yelp Dataset Loading ---

def load_yelp_dataset(selection_name: str, limit: int = None) -> tuple[list[dict], list[dict]]:
    """Load Yelp dataset from cached files (no raw file access).

    Reads:
    - selection_n.jsonl (restaurant IDs + LLM scores)
    - rev_selection_n.jsonl (sampled review_ids per restaurant)
    - reviews_cache_n.jsonl (reviews + user metadata)
    - restaurants_cache_n.jsonl (restaurant metadata)
    - requests_n.json (user requests for this selection)

    Args:
        selection_name: e.g., "selection_1"
        limit: Max restaurants to return (uses top by llm_percent)

    Returns: (items, requests) tuple where items is list of dicts with full data:
    {
        "item_id": "...",
        "item_name": "...",
        "city": "...",
        "state": "...",
        "categories": [...],
        "stars": 4.5,
        "attributes": {...},
        "llm_percent": 95,
        "item_data": [
            {
                "review_id": "...",
                "review": "...",
                "stars": 4,
                "user": {"name": "...", "friends": [...], ...}
            }
        ]
    }
    """
    # Derive file paths
    n = selection_name.replace("selection_", "")
    selection_path = YELP_DIR / f"{selection_name}.jsonl"
    rev_selection_path = YELP_DIR / f"rev_{selection_name}.jsonl"
    reviews_cache_path = YELP_DIR / f"reviews_cache_{n}.jsonl"
    restaurants_cache_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"
    requests_path = YELP_DIR / f"requests_{n}.json"

    # Check files exist with smart error messages
    selection_exists = selection_path.exists()
    cache_files = [rev_selection_path, reviews_cache_path, restaurants_cache_path]
    cache_missing = [p for p in cache_files if not p.exists()]

    if not selection_exists:
        # Selection file missing - need to run both scripts
        msg = f"Selection file not found: {selection_path}\n\n"
        msg += f"To create, run:\n"
        msg += f"  1. python data/scripts/yelp_curation.py\n"
        msg += f"  2. python data/scripts/yelp_review_sampler.py {selection_name}"
        raise FileNotFoundError(msg)
    elif cache_missing:
        # Selection exists but cache files missing - need to run sampler
        msg = f"Cache files missing for '{selection_name}':\n"
        msg += "\n".join(f"  - {p.name}" for p in cache_missing)
        msg += f"\n\nTo create, run:\n"
        msg += f"  python data/scripts/yelp_review_sampler.py {selection_name}"
        raise FileNotFoundError(msg)

    # Load selection (for llm_percent ordering)
    selection = {}
    with open(selection_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                selection[item["item_id"]] = item

    # Load rev_selection (review_ids per restaurant)
    rev_selection = {}
    with open(rev_selection_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rev_selection[item["item_id"]] = item["review_ids"]

    # Load reviews cache into dict by review_id
    reviews_cache = {}
    with open(reviews_cache_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                reviews_cache[r["review_id"]] = r

    # Load restaurants cache into dict by business_id
    restaurants_cache = {}
    with open(restaurants_cache_path) as f:
        for line in f:
            if line.strip():
                biz = json.loads(line)
                restaurants_cache[biz["business_id"]] = biz

    # Sort by llm_percent descending
    sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))
    if limit:
        sorted_ids = sorted_ids[:limit]

    # Assemble final items
    items = []
    for biz_id in sorted_ids:
        biz = restaurants_cache.get(biz_id, {})
        sel = selection.get(biz_id, {})
        review_ids = rev_selection.get(biz_id, [])

        # Parse categories string to list
        cats_str = biz.get("categories", "")
        categories = [c.strip() for c in cats_str.split(",") if c.strip()] if cats_str else []

        # Build reviews list in order
        item_data = []
        for rid in review_ids:
            r = reviews_cache.get(rid)
            if r:
                item_data.append({
                    "review_id": r["review_id"],
                    "review": r["text"],
                    "stars": r["stars"],
                    "date": r.get("date", ""),
                    "useful": r.get("useful", 0),
                    "funny": r.get("funny", 0),
                    "cool": r.get("cool", 0),
                    "user": r.get("user", {})
                })

        items.append({
            "item_id": biz_id,
            "item_name": biz.get("name", ""),
            "address": biz.get("address", ""),
            "city": biz.get("city", ""),
            "state": biz.get("state", ""),
            "postal_code": biz.get("postal_code", ""),
            "latitude": biz.get("latitude"),
            "longitude": biz.get("longitude"),
            "stars": biz.get("stars"),
            "review_count": biz.get("review_count"),
            "is_open": biz.get("is_open"),
            "attributes": biz.get("attributes", {}),
            "categories": categories,
            "hours": biz.get("hours"),
            "llm_percent": sel.get("llm_percent", 0),
            "llm_reasoning": sel.get("llm_reasoning", ""),
            "item_data": item_data
        })

    # Load requests (use defaults if file doesn't exist)
    if requests_path.exists():
        requests = load_requests(str(requests_path))
    else:
        requests = DEFAULT_REQUESTS

    return items, requests


