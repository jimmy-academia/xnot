#!/usr/bin/env python3
"""Precompute ground truth labels for Yelp dataset.

Evaluates each (item × request) pair and outputs ground truth labels
based on the request's structured conditions.

Uses three-value logic:
- 1 = true/satisfied/recommend
- -1 = false/not satisfied/not recommend
- 0 = unknown (attribute missing or null)

Usage:
    python data/scripts/yelp_precompute_groundtruth.py selection_1
    python data/scripts/yelp_precompute_groundtruth.py selection_1 --limit 0  # all items
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from data.loader import YELP_DIR
from utils.utils import loadjl

console = Console()


def parse_attr_value(value):
    """Parse Yelp attribute string to Python value.

    Examples:
        "True" → True
        "False" → False
        "u'free'" → "free"
        "'quiet'" → "quiet"
        "{'romantic': False}" → {"romantic": False}
        None → None
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    # Handle boolean strings
    if value == "True":
        return True
    if value == "False":
        return False
    if value.lower() == "none":
        return None

    # Handle unicode string prefix: u'value' or u"value"
    match = re.match(r"^u?['\"](.+)['\"]$", value)
    if match:
        return match.group(1)

    # Handle dict/list literals
    if value.startswith("{") or value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


def get_nested_value(item: dict, path: list):
    """Navigate nested dict using path, parsing attribute strings.

    Returns None if path doesn't exist or value is missing.
    """
    current = item
    for i, key in enumerate(path):
        if current is None:
            return None
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None

        value = current[key]

        # Parse string values (Yelp attributes are stored as strings)
        if isinstance(value, str) and i < len(path) - 1:
            # Need to navigate deeper, parse the string
            value = parse_attr_value(value)
        elif i == len(path) - 1:
            # Final value, parse it
            value = parse_attr_value(value)

        current = value

    return current


def evaluate_condition(item: dict, condition: dict) -> tuple[int, dict]:
    """Evaluate a single condition against an item.

    Returns:
        (result, evidence) where:
        - result: 1 (satisfied), -1 (not satisfied), 0 (unknown)
        - evidence: {"value": ..., "satisfied": ...}
    """
    aspect = condition.get("aspect", "")
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])

    # Get the raw value
    value = get_nested_value(item, path)

    # If value is None or missing, return unknown
    if value is None:
        return 0, {"value": None, "satisfied": 0}

    # Evaluate based on aspect
    satisfied = evaluate_aspect(aspect, value)

    return satisfied, {"value": value, "satisfied": satisfied}


def evaluate_aspect(aspect: str, value) -> int:
    """Evaluate if an aspect is satisfied given a value.

    Returns: 1 (satisfied), -1 (not satisfied), 0 (unknown)
    """
    if value is None:
        return 0

    # Boolean aspects (value should be True)
    boolean_true_aspects = {
        "takes_reservations", "has_outdoor_seating", "dogs_allowed",
        "has_bike_parking", "good_for_groups", "wheelchair_accessible",
        "offers_takeout", "accepts_credit_cards", "good_for_breakfast",
        "good_for_lunch", "good_for_dinner", "good_for_brunch",
        "good_for_dessert", "good_for_latenight", "romantic_ambience",
        "intimate_ambience", "trendy_ambience", "casual_ambience",
        "classy_ambience", "hipster_ambience", "touristy_ambience",
        "upscale_ambience", "divey_ambience"
    }

    # Boolean aspects (value should be False)
    boolean_false_aspects = {
        "no_tv"
    }

    # String contains aspects
    if aspect == "has_free_wifi":
        if isinstance(value, str):
            return 1 if "free" in value.lower() else -1
        return -1

    if aspect == "quiet_ambience":
        if isinstance(value, str):
            return 1 if "quiet" in value.lower() else -1
        return -1

    if aspect == "serves_alcohol":
        if isinstance(value, str):
            return 1 if value.lower() not in ("none", "no") else -1
        if isinstance(value, bool):
            return 1 if value else -1
        return 0

    if aspect == "has_business_parking":
        # BusinessParking is a dict, check if any parking type is True
        if isinstance(value, dict):
            has_parking = any(v for v in value.values() if v is True)
            return 1 if has_parking else -1
        if isinstance(value, bool):
            return 1 if value else -1
        return 0

    # Generic boolean True aspects
    if aspect in boolean_true_aspects:
        if isinstance(value, bool):
            return 1 if value else -1
        if isinstance(value, str):
            return 1 if value.lower() == "true" else -1
        return 0

    # Generic boolean False aspects (satisfied when value is False)
    if aspect in boolean_false_aspects:
        if isinstance(value, bool):
            return 1 if not value else -1
        if isinstance(value, str):
            return 1 if value.lower() == "false" else -1
        return 0

    # Default: if we have a truthy value, consider it satisfied
    if value:
        return 1
    return -1


def evaluate_structure(item: dict, structure: dict) -> tuple[int, dict]:
    """Evaluate full AND/OR structure with three-value logic.

    Returns:
        (result, evidence) where:
        - result: 1 (recommend), -1 (not recommend), 0 (unknown)
        - evidence: dict of aspect → {value, satisfied}
    """
    op = structure.get("op", "AND")
    args = structure.get("args", [])

    all_evidence = {}
    results = []

    for arg in args:
        if "op" in arg:
            # Nested structure
            result, nested_evidence = evaluate_structure(item, arg)
            all_evidence.update(nested_evidence)
            results.append(result)
        else:
            # Single condition
            result, evidence = evaluate_condition(item, arg)
            aspect = arg.get("aspect", "unknown")
            all_evidence[aspect] = evidence
            results.append(result)

    # Apply three-value logic
    if op == "AND":
        # AND: min(results) — any -1 → -1, any 0 (no -1) → 0, all 1 → 1
        final = min(results) if results else 0
    else:  # OR
        # OR: max(results) — any 1 → 1, any 0 (no 1) → 0, all -1 → -1
        final = max(results) if results else 0

    return final, all_evidence


def generate_groundtruth(selection_name: str, limit: int = 10):
    """Generate ground truth labels for a selection.

    Args:
        selection_name: e.g., "selection_1"
        limit: Max items to process (0 for all, default 10 for testing)
    """
    n = selection_name.replace("selection_", "")

    # Load restaurants
    restaurants_cache_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"
    selection_path = YELP_DIR / f"{selection_name}.jsonl"
    requests_path = YELP_DIR / f"requests_{n}.jsonl"
    output_path = YELP_DIR / f"groundtruth_{n}.jsonl"

    # Check files exist
    for p in [restaurants_cache_path, selection_path, requests_path]:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/red]")
            sys.exit(1)

    # Load data
    console.print(f"[cyan]Loading data...[/cyan]")
    restaurants = {r["business_id"]: r for r in loadjl(restaurants_cache_path)}
    selection = {item["item_id"]: item for item in loadjl(selection_path)}
    requests = loadjl(requests_path)

    # Sort by llm_percent and apply limit
    sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))
    if limit > 0:
        sorted_ids = sorted_ids[:limit]

    console.print(f"  Restaurants: {len(sorted_ids)}")
    console.print(f"  Requests: {len(requests)}")

    # Generate ground truth
    console.print(f"[cyan]Computing ground truth...[/cyan]")
    output = []
    stats = {1: 0, 0: 0, -1: 0}

    for item_id in sorted_ids:
        item = restaurants.get(item_id, {})

        for req in requests:
            structure = req.get("structure", {})
            gold_label, evidence = evaluate_structure(item, structure)

            output.append({
                "item_id": item_id,
                "request_id": req["id"],
                "gold_label": gold_label,
                "evidence": evidence
            })
            stats[gold_label] += 1

    # Write output
    console.print(f"[cyan]Writing output...[/cyan]")
    with open(output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Saved {len(output)} entries to {output_path}[/green]")
    console.print(f"\n[bold]Label distribution:[/bold]")
    console.print(f"  Recommend (1):     {stats[1]}")
    console.print(f"  Unknown (0):       {stats[0]}")
    console.print(f"  Not recommend (-1): {stats[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Precompute ground truth labels")
    parser.add_argument("selection", help="Selection name (e.g., selection_1)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max items to process (0 for all, default: 10)")
    args = parser.parse_args()

    console.print(f"\n[bold]=== Ground Truth Generator ===[/bold]")
    console.print(f"Selection: {args.selection}")
    console.print(f"Limit: {args.limit if args.limit > 0 else 'all'}")

    generate_groundtruth(args.selection, args.limit)


if __name__ == "__main__":
    main()
