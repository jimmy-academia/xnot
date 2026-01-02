#!/usr/bin/env python3
"""Data loading and formatting utilities for philly_cafes dataset."""

import json
from pathlib import Path
from typing import Any

from utils.io import loadjl


DATA_DIR = Path(__file__).parent


class Dataset:
    """Wrapper for loaded dataset with schema/stats."""

    def __init__(self, name: str, items: list, requests: list, groundtruth: dict):
        self.name = name
        self.items = items
        self.requests = requests
        self.groundtruth = groundtruth  # {request_id: gold_restaurant_id}

    def _get_schema(self) -> str:
        """Extract nested schema from first item."""
        if not self.items:
            return "  (no items)"

        item = self.items[0]
        lines = ["  Item Schema:"]

        # Top-level keys (excluding reviews)
        top_keys = [k for k in item.keys() if k != "reviews"]
        lines.append(f"    {', '.join(top_keys[:8])}")
        if len(top_keys) > 8:
            lines.append(f"    {', '.join(top_keys[8:])}")

        # reviews structure
        if "reviews" in item and item["reviews"]:
            review = item["reviews"][0]
            review_keys = list(review.keys())
            lines.append(f"    reviews[]:")
            lines.append(f"      {', '.join(review_keys)}")

        return "\n".join(lines)

    def __repr__(self):
        total_reviews = sum(len(item.get("reviews", [])) for item in self.items)
        return (
            f"Dataset({self.name})\n"
            f"  Items: {len(self.items)}\n"
            f"  Requests: {len(self.requests)}\n"
            f"  Reviews: {total_reviews}\n"
            f"  Groundtruth: {len(self.groundtruth)} mappings\n\n"
            f"{self._get_schema()}"
        )

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def load_requests(data_name: str) -> list[dict]:
    """Load user requests from requests.jsonl.

    Args:
        data_name: Dataset name (e.g., 'philly_cafes')

    Returns:
        List of request dicts with 'id', 'text', 'gold_restaurant', etc.
    """
    path = DATA_DIR / data_name / "requests.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Requests file not found: {path}")

    requests = loadjl(path)
    # Normalize: ensure 'context' field exists (for method interface)
    for req in requests:
        if "text" in req and "context" not in req:
            req["context"] = req["text"]
    return requests


def load_groundtruth(data_name: str) -> dict:
    """Load groundtruth mapping from groundtruth.jsonl.

    Args:
        data_name: Dataset name (e.g., 'philly_cafes')

    Returns:
        Dict of {request_id: {"gold_restaurant": str, "gold_idx": int}}
    """
    path = DATA_DIR / data_name / "groundtruth.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {path}")

    groundtruth = {}
    for gt in loadjl(path):
        groundtruth[gt["request_id"]] = {
            "gold_restaurant": gt["gold_restaurant"],
            "gold_idx": gt["gold_idx"],
        }
    return groundtruth


def load_dataset(data_name: str, review_limit: int = None) -> Dataset:
    """Load dataset with restaurants, reviews, requests, and groundtruth.

    Args:
        data_name: Dataset name (e.g., 'philly_cafes')
        review_limit: Max reviews per restaurant

    Returns:
        Dataset object with items, requests, and groundtruth
    """
    data_dir = DATA_DIR / data_name

    # Check required files
    restaurants_path = data_dir / "restaurants.jsonl"
    reviews_path = data_dir / "reviews.jsonl"
    requests_path = data_dir / "requests.jsonl"
    groundtruth_path = data_dir / "groundtruth.jsonl"

    missing = []
    for p in [restaurants_path, reviews_path, requests_path, groundtruth_path]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {', '.join(missing)}")

    # Load restaurants (no limit - request filtering happens in run.py)
    restaurants = loadjl(restaurants_path)

    # Build restaurant lookup by business_id
    restaurant_by_id = {r["business_id"]: r for r in restaurants}

    # Load reviews and group by business_id
    reviews_by_biz = {}
    for review in loadjl(reviews_path):
        biz_id = review["business_id"]
        if biz_id not in reviews_by_biz:
            reviews_by_biz[biz_id] = []
        reviews_by_biz[biz_id].append(review)

    # Assemble items (restaurant + reviews)
    items = []
    for rest in restaurants:
        biz_id = rest["business_id"]
        reviews = reviews_by_biz.get(biz_id, [])

        # Apply review limit
        if review_limit:
            reviews = reviews[:review_limit]

        # Parse categories string to list
        cats_str = rest.get("categories", "")
        categories = [c.strip() for c in cats_str.split(",") if c.strip()] if cats_str else []

        items.append({
            "item_id": biz_id,
            "item_name": rest.get("name", ""),
            "address": rest.get("address", ""),
            "city": rest.get("city", ""),
            "state": rest.get("state", ""),
            "postal_code": rest.get("postal_code", ""),
            "latitude": rest.get("latitude"),
            "longitude": rest.get("longitude"),
            "stars": rest.get("stars"),
            "review_count": rest.get("review_count"),
            "is_open": rest.get("is_open"),
            "attributes": rest.get("attributes", {}),
            "categories": categories,
            "hours": rest.get("hours"),
            "llm_score": rest.get("llm_score"),
            "llm_reasoning": rest.get("llm_reasoning", ""),
            "reviews": [
                {
                    "review_id": r.get("review_id", ""),
                    "review": r.get("text", ""),
                    "stars": r.get("stars", 0),
                    "date": r.get("date", ""),
                    "user_id": r.get("user_id", ""),
                }
                for r in reviews
            ]
        })

    # Load requests
    requests = load_requests(data_name)

    # Load groundtruth
    groundtruth = load_groundtruth(data_name)

    return Dataset(data_name, items, requests, groundtruth)


def filter_by_candidates(dataset: Dataset, n_candidates: int) -> Dataset:
    """Filter dataset to top-N candidate restaurants by gold frequency.

    Prioritizes restaurants that appear most frequently as gold answers,
    then fills remaining slots with other restaurants.
    Filters requests to only those whose gold is in the candidate set.

    Args:
        dataset: Original dataset
        n_candidates: Number of candidates to keep

    Returns:
        New Dataset with filtered items, requests, and remapped groundtruth
    """
    from collections import Counter

    if n_candidates is None or n_candidates >= len(dataset.items):
        return dataset

    total_items = len(dataset.items)

    # Count gold frequency (how often each restaurant is the answer)
    gold_counts = Counter(gt["gold_idx"] for gt in dataset.groundtruth.values())

    # Get all indices sorted by gold frequency (highest first), then by index
    # Restaurants with 0 gold count come last, sorted by original index
    all_indices = list(range(total_items))
    all_indices.sort(key=lambda i: (-gold_counts.get(i, 0), i))

    # Select top-N
    top_n_indices = all_indices[:n_candidates]
    top_n_set = set(top_n_indices)

    # Create index mapping (old_idx -> new_idx)
    # Sort by original index to maintain stable ordering
    sorted_top_n = sorted(top_n_indices)
    idx_mapping = {old: new for new, old in enumerate(sorted_top_n)}

    # Filter items
    new_items = [dataset.items[i] for i in sorted_top_n]

    # Filter requests to those whose gold is in top-N
    new_requests = []
    new_groundtruth = {}
    for req in dataset.requests:
        req_id = req["id"]
        gt = dataset.groundtruth.get(req_id)
        if gt and gt["gold_idx"] in top_n_set:
            new_requests.append(req)
            # Remap gold_idx to new position
            new_groundtruth[req_id] = {
                "gold_restaurant": gt["gold_restaurant"],
                "gold_idx": idx_mapping[gt["gold_idx"]],
            }

    return Dataset(
        name=f"{dataset.name}_top{n_candidates}",
        items=new_items,
        requests=new_requests,
        groundtruth=new_groundtruth,
    )


def format_query(item: dict, mode: str = "string") -> tuple[Any, int]:
    """Format restaurant item as query for LLM. Excludes ground-truth labels.

    Args:
        item: Restaurant dict with reviews
        mode: "string" for text format, "dict" for structured format

    Returns:
        (query, review_count) tuple
    """
    reviews = item.get("reviews", [])

    if mode == "dict":
        # Return clean dict for structured access (ANoT, Weaver, etc.)
        return {
            "item_id": item.get("item_id", "unknown"),
            "item_name": item.get("item_name", "Unknown"),
            "city": item.get("city", "Unknown"),
            "address": item.get("address", ""),
            "attributes": item.get("attributes", {}),
            "hours": item.get("hours"),
            "categories": item.get("categories", []),
            "item_data": [
                {
                    "review_id": r.get("review_id", ""),
                    "review": r.get("review", ""),
                    "stars": r.get("stars", 0),
                    "date": r.get("date", ""),
                }
                for r in reviews
            ]
        }, len(reviews)

    # String mode (default) - for CoT, PS, Listwise
    parts = [
        "Restaurant:",
        f"Name: {item.get('item_name', 'Unknown')}",
        f"City: {item.get('city', 'Unknown')}",
        f"Address: {item.get('address', 'Unknown')}",
    ]

    # Add key attributes
    attrs = item.get("attributes", {})
    if attrs.get("RestaurantsPriceRange2"):
        parts.append(f"Price Range: {attrs['RestaurantsPriceRange2']}")
    if attrs.get("NoiseLevel"):
        parts.append(f"Noise Level: {attrs['NoiseLevel']}")
    if attrs.get("WiFi"):
        parts.append(f"WiFi: {attrs['WiFi']}")

    categories = item.get("categories", [])
    if categories:
        parts.append(f"Categories: {', '.join(categories[:5])}")

    parts.append("")
    parts.append(f"Reviews ({len(reviews)}):")
    for r in reviews[:10]:  # Limit to first 10 reviews in string mode
        stars = r.get("stars", "?")
        text = r.get("review", "")[:300]
        if len(r.get("review", "")) > 300:
            text += "..."
        parts.append(f"  [{stars} stars] {text}")

    return "\n".join(parts), len(reviews)


def format_ranking_query(items: list[dict], mode: str = "string") -> tuple[Any, int]:
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
                    "address": item.get("address", ""),
                    "attributes": item.get("attributes", {}),
                    "categories": item.get("categories", []),
                    "hours": item.get("hours"),
                    "item_data": [
                        {
                            "review_id": r.get("review_id", ""),
                            "review": r.get("review", ""),
                            "stars": r.get("stars", 0),
                            "date": r.get("date", ""),
                        }
                        for r in item.get("reviews", [])
                    ]
                }
                for i, item in enumerate(items)
            ]
        }, len(items)

    # String mode - serialize full item dicts as JSON (no truncation)
    parts = ["Restaurants:\n"]
    for i, item in enumerate(items, 1):
        item_dict = {
            "index": i,
            "item_id": item.get("item_id"),
            "item_name": item.get("item_name", "Unknown"),
            "city": item.get("city", "Unknown"),
            "address": item.get("address", ""),
            "attributes": item.get("attributes", {}),
            "categories": item.get("categories", []),
            "hours": item.get("hours"),
            "reviews": [
                {
                    "review_id": r.get("review_id", ""),
                    "review": r.get("review", ""),
                    "stars": r.get("stars", 0),
                    "date": r.get("date", ""),
                }
                for r in item.get("reviews", [])
            ]
        }
        parts.append(json.dumps(item_dict, indent=2))
        parts.append("")

    return "\n".join(parts), len(items)
