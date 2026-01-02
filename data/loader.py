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


def load_dataset(data_name: str, limit: int = None, review_limit: int = None) -> Dataset:
    """Load dataset with restaurants, reviews, requests, and groundtruth.

    Args:
        data_name: Dataset name (e.g., 'philly_cafes')
        limit: Max restaurants to load
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

    # Load restaurants
    restaurants = loadjl(restaurants_path)
    if limit:
        restaurants = restaurants[:limit]

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

    # String mode
    parts = ["Restaurants:\n"]
    for i, item in enumerate(items, 1):
        reviews = item.get("reviews", [])
        attrs = item.get("attributes", {})

        parts.append(f"[{i}] {item.get('item_name', 'Unknown')} ({item.get('city', 'Unknown')})")

        # Key attributes
        attr_parts = []
        if attrs.get("RestaurantsPriceRange2"):
            attr_parts.append(f"Price: {attrs['RestaurantsPriceRange2']}")
        if attrs.get("WiFi"):
            attr_parts.append(f"WiFi: {attrs['WiFi']}")
        if attr_parts:
            parts.append(f"    {', '.join(attr_parts)}")

        parts.append(f"    Reviews: {len(reviews)}")

        # Include brief review excerpts (first 2 reviews, truncated)
        for r in reviews[:2]:
            excerpt = r.get("review", "")[:120]
            if len(r.get("review", "")) > 120:
                excerpt += "..."
            parts.append(f"      - [{r.get('stars', '?')}★] {excerpt}")
        parts.append("")

    return "\n".join(parts), len(items)
