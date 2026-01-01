#!/usr/bin/env python3
"""Data loading and formatting utilities."""

import json
from pathlib import Path

from utils.io import loadjl
from utils.parsing import parse_final_answer, normalize_pred

DATA_DIR = Path("data")
ATTACKED_DIR = DATA_DIR / "attacked"
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


def load_requests(path: str = "requests.jsonl") -> list[dict]:
    """Load user requests from JSONL file.

    Supports both simple format (id, context) and complex format (id, text, structure).
    For complex format, 'text' is used as 'context'.
    """
    try:
        requests = loadjl(path)
        # Normalize complex format to have 'context' field
        for req in requests:
            if "text" in req and "context" not in req:
                req["context"] = req["text"]
        return requests
    except FileNotFoundError:
        raise FileNotFoundError(f"Requests file not found: {path}")


def format_query(item: dict, mode: str = "string"):
    """Format restaurant item as query. Excludes ground-truth labels.

    Returns: (query, review_count)
    """
    reviews = item.get("item_data", [])

    if mode == "dict":
        # Return clean dict for structured access (includes attributes, hours, user for ANoT)
        return {
            "item_id": item.get("item_id", "unknown"),
            "item_name": item.get("item_name", "Unknown"),
            "city": item.get("city", "Unknown"),
            "neighborhood": item.get("neighborhood", "Unknown"),
            "attributes": item.get("attributes", {}),
            "hours": item.get("hours"),
            "categories": item.get("categories", []),
            "item_data": [
                {
                    "review_id": r.get("review_id", ""),
                    "review": r.get("review", ""),
                    "stars": r.get("stars", 0),
                    "date": r.get("date", ""),
                    "user": r.get("user", {})
                }
                for r in reviews
            ]
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


def format_ranking_query(items: list[dict], mode: str = "string") -> tuple:
    """Format all items with indices for ranking task.

    Args:
        items: List of item dicts (all restaurants to rank)
        mode: "string" for text format, "dict" for structured format

    Returns:
        (query, item_count) where query has items indexed 1 to N
    """
    if mode == "dict":
        return {
            "items": [
                {
                    "index": i + 1,
                    "item_id": item.get("item_id"),
                    "item_name": item.get("item_name", "Unknown"),
                    "city": item.get("city", "Unknown"),
                    "attributes": item.get("attributes", {}),
                    "categories": item.get("categories", []),
                    "hours": item.get("hours"),
                    "item_data": [
                        {
                            "review_id": r.get("review_id", ""),
                            "review": r.get("review", ""),
                            "stars": r.get("stars", 0),
                            "date": r.get("date", ""),
                            "user": r.get("user", {})
                        }
                        for r in item.get("item_data", [])
                    ]
                }
                for i, item in enumerate(items)
            ]
        }, len(items)

    # String mode
    parts = ["Restaurants:\n"]
    for i, item in enumerate(items, 1):
        reviews = item.get("item_data", [])
        parts.append(f"[{i}] {item.get('item_name', 'Unknown')} ({item.get('city', 'Unknown')})")
        parts.append(f"    Reviews: {len(reviews)}")
        # Include brief review excerpts (first 2 reviews, truncated)
        for r in reviews[:2]:
            excerpt = r.get("review", "")[:150]
            if len(r.get("review", "")) > 150:
                excerpt += "..."
            parts.append(f"      - {excerpt}")
        parts.append("")

    return "\n".join(parts), len(items)


def load_dataset(data: str, selection_name: str = None, limit: int = None, review_limit: int = None) -> Dataset:
    """Unified dataset loading - returns clean data (attacks applied separately).

    Args:
        data: Dataset name (e.g., 'yelp') or explicit path to JSONL file
        selection_name: Selection name (e.g., 'selection_1') or None for legacy
        limit: Max items to load
        review_limit: Max reviews per restaurant (for testing)

    Returns: Dataset object with clean (unattacked) data.
             Attacks are applied in run_evaluation_loop() via apply_attacks().
    """
    if data == 'yelp':
        items, requests = load_yelp_dataset(selection_name, limit, review_limit)
        return Dataset(f"yelp/{selection_name}", items, requests)

    # Legacy dataset loading (fallback)
    raise ValueError(f"Unknown dataset: {data}")


# --- Yelp Dataset Loading ---

def load_yelp_dataset(selection_name: str, limit: int = None, review_limit: int = None) -> tuple[list[dict], list[dict]]:
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
        review_limit: Max reviews per restaurant (for testing)

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
    requests_path = YELP_DIR / f"requests_{n}.jsonl"
    groundtruth_path = YELP_DIR / f"groundtruth_{n}.jsonl"

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

    # Check groundtruth file exists
    if not groundtruth_path.exists():
        msg = f"Groundtruth file not found: {groundtruth_path}\n\n"
        msg += f"To create, run:\n"
        msg += f"  python data/scripts/yelp_precompute_groundtruth.py {selection_name}"
        raise FileNotFoundError(msg)

    # Load groundtruth and build lookup: {item_id: {request_id: gold_label}}
    groundtruth_lookup = {}
    for gt in loadjl(groundtruth_path):
        item_id = gt["item_id"]
        if item_id not in groundtruth_lookup:
            groundtruth_lookup[item_id] = {}
        groundtruth_lookup[item_id][gt["request_id"]] = gt["gold_label"]

    # Load selection (for llm_percent ordering)
    selection = {item["item_id"]: item for item in loadjl(selection_path)}

    # Load rev_selection (review_ids per restaurant)
    rev_selection = {item["item_id"]: item["review_ids"] for item in loadjl(rev_selection_path)}

    # Load reviews cache into dict by review_id
    reviews_cache = {r["review_id"]: r for r in loadjl(reviews_cache_path)}

    # Load restaurants cache into dict by business_id
    restaurants_cache = {biz["business_id"]: biz for biz in loadjl(restaurants_cache_path)}

    # Sort by llm_percent descending
    sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))

    # Load groundtruth metadata for item filtering (if exists)
    meta_path = YELP_DIR / f"groundtruth_{n}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            gt_meta = json.load(f)
        gt_item_ids = set(gt_meta.get("item_ids", []))
        original_count = len(sorted_ids)
        # Filter to only items in groundtruth (preserving llm_percent order)
        sorted_ids = [sid for sid in sorted_ids if sid in gt_item_ids]
        print(f"Filtered {original_count} → {len(sorted_ids)} items (from groundtruth metadata)")

    # Apply user limit on top of groundtruth filter
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

        # Build reviews list in order (apply review_limit if set)
        item_data = []
        review_ids_to_use = review_ids[:review_limit] if review_limit else review_ids
        for rid in review_ids_to_use:
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
            "gold_labels": groundtruth_lookup.get(biz_id, {}),
            "item_data": item_data
        })

    # Load requests (error if file doesn't exist)
    if not requests_path.exists():
        msg = f"Requests file not found: {requests_path}\n\n"
        msg += f"To create, run:\n"
        msg += f"  python data/scripts/yelp_requests.py {selection_name}"
        raise FileNotFoundError(msg)

    requests = load_requests(str(requests_path))
    return items, requests


def load_groundtruth_scores(selection_name: str) -> dict:
    """Load groundtruth scores for ranking evaluation.

    Gold selection policy: Only items with gold_label=+1 are candidates.
    The gold item per request is the one with highest total_score among these.
    This matches the logic in yelp_precompute_groundtruth.py's print_hits_at_k().

    Args:
        selection_name: e.g., "selection_1"

    Returns:
        Dict of {request_id: {item_id: total_score}} for gold_label=+1 items only
    """
    n = selection_name.replace("selection_", "")
    groundtruth_path = YELP_DIR / f"groundtruth_{n}.jsonl"

    if not groundtruth_path.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {groundtruth_path}")

    # Build lookup: {request_id: {item_id: total_score}}
    # Only include items with gold_label=+1 (definitively good matches)
    scores = {}
    for gt in loadjl(groundtruth_path):
        req_id = gt["request_id"]
        item_id = gt["item_id"]
        gold_label = gt.get("gold_label", 0)

        # Gold selection: only items with gold_label=+1 are candidates
        if gold_label != 1:
            continue

        score = gt.get("total_score", 0)
        if req_id not in scores:
            scores[req_id] = {}
        scores[req_id][item_id] = score

    return scores
