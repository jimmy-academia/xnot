#!/usr/bin/env python3
"""Data loading and formatting utilities for philly_cafes dataset."""

import ast
import json
from pathlib import Path
from typing import Any

from utils.io import loadjl


DATA_DIR = Path(__file__).parent

# Fields to strip from reviews/users
# Note: user_id now passes through for social filtering tasks (G09/G10)
STRIP_USER_FIELDS = {'friends'}  # friends list is huge, strip it
STRIP_REVIEW_FIELDS = set()  # Let review_id, business_id, user_id pass through


def _load_user_mapping(data_name: str) -> dict:
    """Load user mapping for G09/G10 social data synthesis.

    Returns:
        Dict with 'user_names' and 'restaurant_reviews' or None if not found
    """
    mapping_path = DATA_DIR / data_name / "user_mapping.json"
    if not mapping_path.exists():
        return None
    with open(mapping_path) as f:
        return json.load(f)


def _strip_review_fields(review: dict) -> dict:
    """Strip bloated fields from review to reduce tokens."""
    # Strip review-level fields
    cleaned = {k: v for k, v in review.items() if k not in STRIP_REVIEW_FIELDS}

    # Strip user-level fields
    if 'user' in cleaned and isinstance(cleaned['user'], dict):
        cleaned['user'] = {k: v for k, v in cleaned['user'].items()
                          if k not in STRIP_USER_FIELDS}

    return cleaned


def _parse_string_value(value: str):
    """Parse string-encoded Python values from Yelp data.

    Handles:
    - Dict strings: "{'hipster': True, ...}" -> dict
    - List strings: "['a', 'b']" -> list
    - Boolean strings: 'True', 'False' -> bool
    - Unicode prefix: "u'average'" -> "average"
    - None string: 'None' -> None
    """
    if not isinstance(value, str):
        return value

    s = value.strip()

    # Handle None
    if s == 'None':
        return None

    # Handle booleans
    if s == 'True':
        return True
    if s == 'False':
        return False

    # Handle unicode prefix: u'value' -> value
    if s.startswith("u'") and s.endswith("'"):
        return s[2:-1]
    if s.startswith('u"') and s.endswith('"'):
        return s[2:-1]

    # Try to parse as dict or list (nested structures only)
    if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, SyntaxError):
            pass

    # Keep as string (leaf value)
    return value


def _parse_attributes(attrs: dict) -> dict:
    """Parse all string-encoded values in attributes dict."""
    if not isinstance(attrs, dict):
        return attrs

    parsed = {}
    for k, v in attrs.items():
        parsed[k] = _parse_string_value(v)
    return parsed


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


def _validate_data_files(data_dir: Path) -> dict[str, Path]:
    """Validate required dataset files exist.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dict mapping file type to path

    Raises:
        FileNotFoundError: If any required files are missing
    """
    paths = {
        "restaurants": data_dir / "restaurants.jsonl",
        "reviews": data_dir / "reviews.jsonl",
        "requests": data_dir / "requests.jsonl",
        "groundtruth": data_dir / "groundtruth.jsonl",
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {', '.join(missing)}")

    return paths


def _build_user_mapping_data(data_name: str, restaurants: list) -> dict:
    """Set up G09/G10 social synthesis data structures.

    The user_mapping.json contains pre-computed (name, friends) assignments
    for ALL reviews. This enables G09/G10 social validation without any
    runtime pattern matching.

    Args:
        data_name: Dataset name
        restaurants: List of restaurant dicts

    Returns:
        Dict with review_assignments and biz_id_to_idx
    """
    user_mapping = _load_user_mapping(data_name)

    # New format: pre-computed review_assignments
    # {rest_idx: [{review_idx, name, friends}, ...]}
    review_assignments = user_mapping.get("review_assignments", {}) if user_mapping else {}
    friend_graph = user_mapping.get("friend_graph", {}) if user_mapping else {}

    # Build restaurant index lookup (business_id -> index string)
    biz_id_to_idx = {r["business_id"]: str(i) for i, r in enumerate(restaurants)}

    return {
        "review_assignments": review_assignments,
        "friend_graph": friend_graph,
        "biz_id_to_idx": biz_id_to_idx,
    }


def _load_reviews_with_synthesis(
    reviews_path: Path,
    biz_id_to_idx: dict,
    review_assignments: dict,
    friend_graph: dict,
) -> dict[str, list]:
    """Load reviews, apply stripping and G09/G10 social synthesis.

    Uses pre-computed review_assignments to set (name, friends) on ALL reviews.
    No runtime pattern matching - assignments are indexed directly.

    Args:
        reviews_path: Path to reviews.jsonl
        biz_id_to_idx: Map of business_id to index string
        review_assignments: Pre-computed {rest_idx: [{review_idx, name, friends}, ...]}
        friend_graph: Map of name to list of friend names

    Returns:
        Dict mapping business_id to list of reviews
    """
    # First pass: load and strip reviews, group by business_id
    raw_reviews_by_biz = {}
    for review in loadjl(reviews_path):
        biz_id = review["business_id"]
        review = _strip_review_fields(review)
        if biz_id not in raw_reviews_by_biz:
            raw_reviews_by_biz[biz_id] = []
        raw_reviews_by_biz[biz_id].append(review)

    # Second pass: apply pre-computed social synthesis
    reviews_by_biz = {}
    for biz_id, reviews in raw_reviews_by_biz.items():
        rest_idx = biz_id_to_idx.get(biz_id)

        if rest_idx and rest_idx in review_assignments:
            # Build index lookup for this restaurant's assignments
            assignments_by_idx = {
                a["review_idx"]: a for a in review_assignments[rest_idx]
            }

            # Apply assignments to each review
            for rev_idx, review in enumerate(reviews):
                assignment = assignments_by_idx.get(rev_idx)
                if assignment and 'user' in review:
                    review['user']['name'] = assignment['name']
                    # Use pre-computed friends, or fall back to friend_graph
                    friends = assignment.get('friends')
                    if friends is None:
                        friends = friend_graph.get(assignment['name'], [])
                    review['user']['friends'] = friends

        reviews_by_biz[biz_id] = reviews

    return reviews_by_biz


# Fields to remove from restaurants (evaluation-only, would leak ground truth)
RESTAURANT_BLOCKLIST = {"llm_score", "llm_reasoning"}


def _assemble_items(
    restaurants: list,
    reviews_by_biz: dict[str, list],
    review_limit: int = None,
) -> list[dict]:
    """Combine restaurants with reviews, parse attributes.

    Args:
        restaurants: List of restaurant dicts
        reviews_by_biz: Dict mapping business_id to list of reviews
        review_limit: Max reviews per restaurant (None for no limit)

    Returns:
        List of item dicts with reviews attached
    """
    items = []

    for rest in restaurants:
        biz_id = rest["business_id"]
        reviews = reviews_by_biz.get(biz_id, [])

        # Apply review limit
        if review_limit:
            reviews = reviews[:review_limit]

        # Pass-through restaurant dict, remove blocked fields
        item = {k: v for k, v in rest.items() if k not in RESTAURANT_BLOCKLIST}

        # Parse string-encoded attributes (Yelp data has dicts as strings)
        if "attributes" in item:
            item["attributes"] = _parse_attributes(item["attributes"])

        # Parse categories string → list
        if isinstance(item.get("categories"), str):
            cats_str = item["categories"]
            item["categories"] = [c.strip() for c in cats_str.split(",") if c.strip()]

        # Attach full reviews (pass-through, no field renaming)
        item["reviews"] = reviews

        items.append(item)

    return items


def load_dataset(data_name: str, review_limit: int = None) -> Dataset:
    """Load dataset with restaurants, reviews, requests, and groundtruth.

    Args:
        data_name: Dataset name (e.g., 'philly_cafes')
        review_limit: Max reviews per restaurant

    Returns:
        Dataset object with items, requests, and groundtruth
    """
    # Validate required files
    paths = _validate_data_files(DATA_DIR / data_name)

    # Load restaurants
    restaurants = loadjl(paths["restaurants"])

    # Set up G09/G10 social synthesis data
    mapping_data = _build_user_mapping_data(data_name, restaurants)

    # Load reviews with synthesis applied
    reviews_by_biz = _load_reviews_with_synthesis(
        paths["reviews"],
        mapping_data["biz_id_to_idx"],
        mapping_data["review_assignments"],
        mapping_data["friend_graph"],
    )

    # Assemble items (restaurant + reviews)
    items = _assemble_items(restaurants, reviews_by_biz, review_limit)

    # Load requests and groundtruth
    requests = load_requests(data_name)
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
        # Pass-through item dict (already cleaned in load_dataset)
        # Reviews already attached as "reviews" with original field names
        return item, len(reviews)

    # String mode (default) - for CoT, PS, Listwise
    parts = [
        "Restaurant:",
        f"Name: {item.get('name', 'Unknown')}",
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
        text = r.get("text", "")[:300]
        if len(r.get("text", "")) > 300:
            text += "..."
        parts.append(f"  [{stars} stars] {text}")

    return "\n".join(parts), len(reviews)


def format_ranking_query(items: list[dict], mode: str = "string",
                         max_reviews: int = None) -> tuple[Any, int, dict]:
    """Format all items with indices for ranking task.

    Args:
        items: List of item dicts (all restaurants to rank)
        mode: "string" for text format, "dict" for structured format
        max_reviews: Max reviews per restaurant (string mode only, None=unlimited)

    Returns:
        (query, item_count, coverage_stats) where query has items indexed 1 to N
    """
    if mode == "dict":
        # Dict format for 1-indexed access: items["1"], items["2"], etc.
        return {
            "items": {
                str(i + 1): {**item, "index": i + 1}
                for i, item in enumerate(items)
            }
        }, len(items), None

    # String mode - serialize full item dicts as JSON (pass-through with index)
    # Apply max_reviews limit if specified
    total_reviews = sum(len(item.get("reviews", [])) for item in items)
    reviews_included = 0

    parts = ["Restaurants:\n"]
    for i, item in enumerate(items, 1):
        item_dict = {**item, "index": i}
        reviews = item.get("reviews", [])

        if max_reviews is not None:
            item_dict["reviews"] = reviews[:max_reviews]
            reviews_included += min(len(reviews), max_reviews)
        else:
            reviews_included += len(reviews)

        parts.append(json.dumps(item_dict, indent=2))
        parts.append("")

    coverage_stats = {
        "restaurants": len(items),
        "reviews_included": reviews_included,
        "reviews_total": total_reviews,
        "reviews_dropped": total_reviews - reviews_included,
        "max_reviews_per_item": max_reviews,
    }

    return "\n".join(parts), len(items), coverage_stats


# Methods using dict mode (don't need truncation - they access data selectively)
DICT_MODE_METHODS = {"anot", "anot_original", "weaver", "react"}


def format_ranking_query_packed(
    items: list[dict],
    token_budget: int,
    model: str = "gpt-4o"
) -> tuple[str, int, dict]:
    """Pack items into context up to token budget.

    Uses deterministic two-pass policy to ensure ALL restaurants are included:
    - Pass 1: Include all restaurants with metadata (no reviews)
    - Pass 2: Add reviews round-robin until budget exhausted

    This ensures fair evaluation where all candidates are visible to the model.

    Args:
        items: List of item dicts (restaurants to rank)
        token_budget: Maximum tokens for the context
        model: Model name for tokenizer selection

    Returns:
        (context_str, item_count, coverage_stats)
    """
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    # Track total reviews for coverage stats
    total_reviews = sum(len(item.get("reviews", [])) for item in items)

    # Pass 1: Build all items with metadata only
    item_data = []  # List of (metadata_dict, reviews_list, included_reviews_list)
    for i, item in enumerate(items, 1):
        metadata = {k: v for k, v in item.items() if k != "reviews"}
        metadata["index"] = i
        reviews = item.get("reviews", [])
        item_data.append({"metadata": metadata, "reviews": reviews, "included": []})

    def build_context() -> str:
        """Build context string from current item_data state."""
        parts = ["Restaurants:\n"]
        for data in item_data:
            item_dict = {**data["metadata"]}
            if data["included"]:
                item_dict["reviews"] = data["included"]
            parts.append(json.dumps(item_dict, indent=2))
            parts.append("")
        return "\n".join(parts)

    # Check if metadata-only fits
    context = build_context()
    current_tokens = count_tokens(context)

    if current_tokens > token_budget:
        # Even metadata doesn't fit - return what we can with warning
        coverage_stats = {
            "restaurants": len(items),
            "reviews_included": 0,
            "reviews_total": total_reviews,
            "reviews_truncated_count": len(items),
            "tokens_used": current_tokens,
            "budget_exceeded": True,
        }
        return context, len(items), coverage_stats

    # Pass 2: Add reviews round-robin until budget exhausted
    # Track next review index for each restaurant
    review_indices = [0] * len(item_data)
    made_progress = True

    while made_progress:
        made_progress = False
        for idx, data in enumerate(item_data):
            review_idx = review_indices[idx]
            if review_idx >= len(data["reviews"]):
                continue  # No more reviews for this restaurant

            # Try adding next review
            test_review = data["reviews"][review_idx]
            data["included"].append(test_review)
            test_context = build_context()
            test_tokens = count_tokens(test_context)

            if test_tokens > token_budget:
                # Remove the review, can't fit
                data["included"].pop()
            else:
                # Keep it, move to next review
                review_indices[idx] += 1
                current_tokens = test_tokens
                made_progress = True

    # Build final context
    context = build_context()
    reviews_included = sum(len(data["included"]) for data in item_data)
    truncated_count = sum(
        1 for data in item_data
        if len(data["included"]) < len(data["reviews"])
    )

    coverage_stats = {
        "restaurants": len(items),
        "reviews_included": reviews_included,
        "reviews_total": total_reviews,
        "reviews_truncated_count": truncated_count,
        "tokens_used": count_tokens(context),
    }

    return context, len(items), coverage_stats
