#!/usr/bin/env python3
"""Validate ground truth dataset.

Checks that each request matches exactly 1 restaurant.
Generates groundtruth.jsonl for downstream evaluation.

Usage:
    python -m data.validate              # interactive dataset selection
    python -m data.validate philly_cafes # specific dataset
"""

import argparse
import ast
import json
import re
import sys
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


DATA_DIR = Path(__file__).parent


# --- Hours Parsing ---

def parse_hours_range(hours_str: str) -> tuple[int, int] | None:
    """Parse 'H:M-H:M' to (start_minutes, end_minutes)."""
    if not hours_str or "-" not in hours_str:
        return None
    try:
        start, end = hours_str.split("-")
        def to_mins(t):
            h, m = t.split(":")
            return int(h) * 60 + int(m)
        return to_mins(start), to_mins(end)
    except:
        return None


def hours_contains(item_hours: str, required: str) -> bool:
    """Check if restaurant hours cover the required time range."""
    item_range = parse_hours_range(item_hours)
    req_range = parse_hours_range(required)
    if not item_range or not req_range:
        return False
    item_start, item_end = item_range
    req_start, req_end = req_range
    # Handle overnight hours (end < start means crosses midnight)
    if item_end < item_start:
        item_end += 24 * 60
    return item_start <= req_start and item_end >= req_end


# --- Review Metadata Weighting ---

def compute_review_weights(reviews: list, weight_by: dict) -> list[float]:
    """Compute weights for each review based on metadata.

    weight_by spec:
        field: path to value (e.g., ["user", "review_count"], ["useful"])
        normalize: "max" (default) - normalize to [0.5, 1.0] range

    Returns list of weights, one per review.
    """
    field_path = weight_by.get("field", [])
    normalize = weight_by.get("normalize", "max")

    values = []
    for r in reviews:
        # Navigate to the value
        v = r
        for key in field_path:
            if isinstance(v, dict):
                v = v.get(key)
            else:
                v = None
                break

        # Special handling for elite (list of years)
        if field_path == ["user", "elite"] and isinstance(v, list):
            v = len([e for e in v if e and e != 'None'])

        values.append(float(v) if v is not None else 0.0)

    # Normalize to [0.5, 1.0] range (so low values still count somewhat)
    max_val = max(values) if values else 1.0
    if normalize == "max" and max_val > 0:
        weights = [0.5 + 0.5 * (v / max_val) for v in values]
    else:
        weights = [1.0] * len(values)  # Equal weights if can't normalize

    return weights


def evaluate_weighted_review_text(reviews: list, pattern: str, weights: list, threshold: float = 0.2) -> int:
    """Evaluate review text pattern with weighted aggregation.

    Returns 1 if weighted matches exceed threshold, -1 otherwise.
    threshold: fraction of total weight that must match (default 0.2 = 20%)
    """
    if not pattern or not reviews:
        return 0

    regex = re.compile(pattern, re.IGNORECASE)
    weighted_matches = 0.0
    total_weight = sum(weights)

    for r, w in zip(reviews, weights):
        text = r.get("text", "")
        if regex.search(text):
            weighted_matches += w

    # Threshold: if weighted matches > threshold of total weight
    if total_weight > 0 and weighted_matches / total_weight > threshold:
        return 1
    return -1


def evaluate_review_meta(reviews: list, evidence_spec: dict) -> int:
    """Evaluate standalone review_meta condition.

    Checks metadata across reviews with optional filtering.

    evidence_spec:
        path: ["user", "elite"] or ["useful"]
        op: "not_empty", "gte", "lte"
        value: threshold for gte/lte
        min_stars: optional star filter (only count reviews with stars >= this)
        agg: "any" (default), "all", "count"
        count: for agg="count", require this many matches
    """
    if not reviews:
        return evidence_spec.get("missing", 0)

    path = evidence_spec.get("path", [])
    op = evidence_spec.get("op", "not_empty")
    value = evidence_spec.get("value")
    min_stars = evidence_spec.get("min_stars")
    agg = evidence_spec.get("agg", "any")
    count_needed = evidence_spec.get("count", 1)

    matching = 0
    total = 0

    for r in reviews:
        # Filter by min_stars if specified
        if min_stars is not None:
            stars = r.get("stars", 0)
            if stars < min_stars:
                continue

        total += 1

        # Navigate to the value
        v = r
        for key in path:
            if isinstance(v, dict):
                v = v.get(key)
            else:
                v = None
                break

        # Check condition
        match = False
        if op == "not_empty":
            # For lists like elite years
            if isinstance(v, list):
                match = bool(v) and len([e for e in v if e and e != 'None']) > 0
            else:
                match = v is not None and v != "" and v != 0
        elif op == "gte" and v is not None:
            match = float(v) >= float(value)
        elif op == "lte" and v is not None:
            match = float(v) <= float(value)

        if match:
            matching += 1

    # Aggregation
    if agg == "any":
        return 1 if matching > 0 else -1
    elif agg == "all":
        return 1 if total > 0 and matching == total else -1
    elif agg == "count":
        return 1 if matching >= count_needed else -1

    return 0


# --- Three-Value Logic (from oldsrc) ---

class TV(Enum):
    """Three-value logic: True, Unknown, False."""
    T = 1
    U = 0
    F = -1


def reduce_tv(op: str, values: list[TV]) -> TV:
    """Reduce list of TV values using AND or OR.

    - AND: False dominates, then Unknown, then True
    - OR: True dominates, then Unknown, then False
    """
    if not values:
        return TV.U

    if op == "AND":
        if any(v is TV.F for v in values):
            return TV.F
        if any(v is TV.U for v in values):
            return TV.U
        return TV.T
    else:  # OR
        if any(v is TV.T for v in values):
            return TV.T
        if any(v is TV.U for v in values):
            return TV.U
        return TV.F


# --- Attribute Parsing (from oldsrc) ---

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
            value = parse_attr_value(value)
        elif i == len(path) - 1:
            value = parse_attr_value(value)

        current = value

    return current


def match_value(actual, expected) -> bool:
    """Normalized string matching (case-insensitive).

    Also parses expected values that may be in Yelp string format.
    """
    if expected is None:
        return False
    if isinstance(expected, bool):
        return actual == expected
    if isinstance(expected, list):
        return any(match_value(actual, e) for e in expected)

    # Parse expected value if it's a Yelp-format string
    parsed_expected = parse_attr_value(expected)

    # Try exact match first (case-insensitive for strings)
    if isinstance(actual, str) and isinstance(parsed_expected, str):
        if actual.lower() == parsed_expected.lower():
            return True
    elif actual == parsed_expected:
        return True

    # Fall back to contains matching
    return str(parsed_expected).lower() in str(actual).lower()


# --- Condition Evaluation (from oldsrc, simplified) ---

def evaluate_item_meta_rule(value, evidence_spec) -> int:
    """Evaluate item_meta using declarative rules from evidence_spec.

    Rules:
    - "not_true" / "true_not" given → value should NOT match (None passes)
    - Missing value → use "missing" field (default 0=neutral)
    - Dict of booleans → OR across children (e.g., BusinessParking)
    - None given → default boolean check (True→1, False→-1, else→0)
    - Only "true" given → match→1, no match→-1
    - Only "false" given → match→-1, no match→1 (negative logic)
    - "contains" given → substring check in string repr
    """
    true_cond = evidence_spec.get("true")
    false_cond = evidence_spec.get("false")
    not_true_cond = evidence_spec.get("not_true") or evidence_spec.get("true_not")
    neutral_cond = evidence_spec.get("neutral")
    contains_cond = evidence_spec.get("contains")
    missing_val = evidence_spec.get("missing", 0)

    # "not_true" / "true_not" - negative check (value should NOT match)
    # Handle BEFORE missing check: None is "not True" so it passes
    if not_true_cond is not None:
        if value is None:
            return 1  # None is not the target value, so passes
        return -1 if match_value(value, not_true_cond) else 1

    # Missing value → use "missing" field (default 0=neutral)
    if value is None:
        return missing_val

    # "contains" check - look for substring in string repr
    if contains_cond is not None:
        value_str = str(value).lower()
        contains_str = str(contains_cond).lower()
        return 1 if contains_str in value_str else -1

    # Dict of booleans → OR across children (e.g., BusinessParking)
    if isinstance(value, dict) and value and all(isinstance(v, bool) for v in value.values()):
        has_true = any(v for v in value.values())
        return 1 if has_true else -1

    # Case 1: None given → default boolean check
    if true_cond is None and false_cond is None and neutral_cond is None:
        if value is True:
            return 1
        if value is False:
            return -1
        return 0

    # Case 2: Only "false" given → negative logic (no match = true)
    if false_cond is not None and true_cond is None and neutral_cond is None:
        return -1 if match_value(value, false_cond) else 1

    # Case 3: Only "true" given → no match = false
    if true_cond is not None and false_cond is None and neutral_cond is None:
        return 1 if match_value(value, true_cond) else -1

    # Case 4: Multiple conditions given → check each explicitly
    if true_cond is not None and match_value(value, true_cond):
        return 1
    if false_cond is not None and match_value(value, false_cond):
        return -1
    if neutral_cond is not None and match_value(value, neutral_cond):
        return 0

    # None matched → neutral
    return 0


def evaluate_review_text_pattern(reviews: list, pattern: str) -> int:
    """Check if any review contains the pattern.

    Returns: 1 if found, -1 if not found
    """
    if not pattern:
        return 0
    pattern_lower = pattern.lower()
    for r in reviews:
        text = r.get("text", "")
        if pattern_lower in text.lower():
            return 1
    return -1


def evaluate_condition(item: dict, condition: dict, reviews: list = None) -> int:
    """Evaluate a single condition against an item.

    Returns: 1 (satisfied), -1 (not satisfied), 0 (unknown)
    """
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])

    if kind == "item_meta":
        value = get_nested_value(item, path)
        return evaluate_item_meta_rule(value, evidence_spec)

    elif kind == "review_text":
        pattern = evidence_spec.get("pattern", "")
        weight_by = evidence_spec.get("weight_by")

        if not reviews:
            return 0

        # If weight_by specified, use weighted aggregation
        if weight_by:
            weights = compute_review_weights(reviews, weight_by)
            threshold = evidence_spec.get("threshold", 0.2)
            return evaluate_weighted_review_text(reviews, pattern, weights, threshold)
        else:
            return evaluate_review_text_pattern(reviews, pattern)

    elif kind == "review_meta":
        # Standalone review metadata check
        if not reviews:
            return evidence_spec.get("missing", 0)
        return evaluate_review_meta(reviews, evidence_spec)

    elif kind == "item_meta_hours":
        day_hours = get_nested_value(item, path)
        required = evidence_spec.get("true", "")
        if not day_hours:
            return evidence_spec.get("missing", 0)
        return 1 if hours_contains(day_hours, required) else -1

    return 0


def evaluate_structure(item: dict, structure: dict, reviews: list = None) -> int:
    """Evaluate full AND/OR structure.

    Returns: 1 (satisfied), -1 (not satisfied), 0 (unknown)
    """
    op = structure.get("op", "AND")
    args = structure.get("args", [])

    child_results = []

    for arg in args:
        if "op" in arg:
            # Nested structure
            result = evaluate_structure(item, arg, reviews)
        else:
            # Single condition
            result = evaluate_condition(item, arg, reviews)
        child_results.append(TV(result))

    final_tv = reduce_tv(op, child_results)
    return final_tv.value


# --- Validation Logic ---

def validate_request(request: dict, restaurants: list, reviews_by_id: dict) -> dict:
    """Check how many restaurants match this request.

    Returns:
        {
            "request_id": str,
            "gold_restaurant": str,
            "gold_idx": int,
            "matches": [business_ids],
            "status": "ok" | "no_match" | "multi_match" | "gold_not_match"
        }
    """
    request_id = request.get("id", "?")
    gold_restaurant = request.get("gold_restaurant", "")
    structure = request.get("structure", {})

    matches = []
    gold_idx = -1

    for idx, restaurant in enumerate(restaurants):
        business_id = restaurant.get("business_id", "")
        reviews = reviews_by_id.get(business_id, [])

        result = evaluate_structure(restaurant, structure, reviews)

        if result == 1:
            matches.append(business_id)

        if business_id == gold_restaurant:
            gold_idx = idx

    # Determine status
    if len(matches) == 0:
        status = "no_match"
    elif len(matches) == 1:
        if matches[0] == gold_restaurant:
            status = "ok"
        else:
            status = "gold_not_match"
    else:
        if gold_restaurant in matches:
            status = "multi_match"
        else:
            status = "multi_match_no_gold"

    return {
        "request_id": request_id,
        "gold_restaurant": gold_restaurant,
        "gold_idx": gold_idx,
        "matches": matches,
        "matches_count": len(matches),
        "status": status
    }


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def validate_dataset(name: str):
    """Validate a dataset and generate groundtruth.jsonl."""
    dataset_dir = DATA_DIR / name

    # Check required files
    restaurants_path = dataset_dir / "restaurants.jsonl"
    reviews_path = dataset_dir / "reviews.jsonl"
    requests_path = dataset_dir / "requests.jsonl"

    for p in [restaurants_path, reviews_path, requests_path]:
        if not p.exists():
            print(f"Error: Missing {p}")
            sys.exit(1)

    # Load data
    restaurants = load_jsonl(restaurants_path)
    reviews = load_jsonl(reviews_path)
    requests = load_jsonl(requests_path)

    # Group reviews by business_id
    reviews_by_id = {}
    for r in reviews:
        bid = r.get("business_id", "")
        if bid not in reviews_by_id:
            reviews_by_id[bid] = []
        reviews_by_id[bid].append(r)

    print(f"Dataset: {name}")
    print(f"  Restaurants: {len(restaurants)}")
    print(f"  Reviews: {len(reviews)}")
    print(f"  Requests: {len(requests)}")
    print()

    # Build evaluation matrix: (item_id, request_id) -> result
    eval_matrix = {}
    item_ids = [r["business_id"] for r in restaurants]
    request_ids = [r["id"] for r in requests]

    for request in requests:
        req_id = request["id"]
        structure = request["structure"]
        for restaurant in restaurants:
            item_id = restaurant["business_id"]
            reviews = reviews_by_id.get(item_id, [])
            result = evaluate_structure(restaurant, structure, reviews)
            eval_matrix[(item_id, req_id)] = result

    # Validate each request
    results = []
    ok_count = 0
    error_count = 0

    for request in requests:
        result = validate_request(request, restaurants, reviews_by_id)
        results.append(result)

        status = result["status"]
        if status == "ok":
            ok_count += 1
        else:
            error_count += 1

    # Compute statistics
    stats = {1: 0, 0: 0, -1: 0}
    for val in eval_matrix.values():
        stats[val] = stats.get(val, 0) + 1

    # Print matrix and stats
    print_groundtruth_matrix(eval_matrix, item_ids, request_ids)
    print_statistics(stats)
    print_hits_summary(results, restaurants)

    print()

    # Summary
    if error_count == 0:
        console.print(f"[green]All {ok_count} requests validated successfully![/green]")
    else:
        console.print(f"[yellow]{ok_count} OK, {error_count} errors[/yellow]")

    # Write groundtruth.jsonl
    groundtruth_path = dataset_dir / "groundtruth.jsonl"
    with open(groundtruth_path, "w") as f:
        for result in results:
            entry = {
                "request_id": result["request_id"],
                "gold_restaurant": result["gold_restaurant"],
                "gold_idx": result["gold_idx"],
                "status": result["status"]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"\nSaved: {groundtruth_path}")

    return error_count == 0


def print_groundtruth_matrix(eval_matrix: dict, item_ids: list, request_ids: list):
    """Print items × requests matrix with colored labels.

    Splits into multiple tables if more than 10 requests (10 per table).

    Args:
        eval_matrix: {(item_id, request_id): result} where result is 1/-1/0
        item_ids: Ordered list of item IDs
        request_ids: Ordered list of request IDs
    """

    # Split requests into chunks of 10
    chunk_size = 10
    request_chunks = [request_ids[i:i+chunk_size] for i in range(0, len(request_ids), chunk_size)]

    for chunk_idx, req_chunk in enumerate(request_chunks):
        if len(request_chunks) > 1:
            title = f"Validation Matrix ({chunk_idx+1}/{len(request_chunks)})"
        else:
            title = "Validation Matrix"

        table = Table(title=title, show_lines=False)
        table.add_column("Item", style="cyan", width=14)
        for req_id in req_chunk:
            table.add_column(req_id, justify="center", width=5)

        for item_id in item_ids:
            row = [item_id[:12] + ".."]
            for req_id in req_chunk:
                label = eval_matrix.get((item_id, req_id), "?")
                if label == 1:
                    row.append("[green]+1[/green]")
                elif label == -1:
                    row.append("[red]-1[/red]")
                elif label == 0:
                    row.append("[yellow]0[/yellow]")
                else:
                    row.append("[dim]?[/dim]")
            table.add_row(*row)

        console.print(table)
        if chunk_idx < len(request_chunks) - 1:
            console.print()  # Add spacing between tables


def print_statistics(stats: dict):
    """Print distribution statistics."""

    total = sum(stats.values())
    if total == 0:
        return

    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total entries: {total}")
    console.print(f"  Match (+1):    [green]{stats.get(1, 0):4d}[/green] ({stats.get(1, 0)/total*100:5.1f}%)")
    console.print(f"  Unknown (0):   [yellow]{stats.get(0, 0):4d}[/yellow] ({stats.get(0, 0)/total*100:5.1f}%)")
    console.print(f"  No match (-1): [red]{stats.get(-1, 0):4d}[/red] ({stats.get(-1, 0)/total*100:5.1f}%)")


def print_hits_summary(results: list, restaurants: list):
    """Print Hits@1 summary showing which restaurant matches each request."""

    console.print("\n[bold]Hits@1 Summary:[/bold]")
    hits = 0
    total = 0

    # Build business_id to index lookup
    id_to_idx = {r["business_id"]: i for i, r in enumerate(restaurants)}
    id_to_name = {r["business_id"]: r.get("name", "?")[:20] for r in restaurants}

    for r in results:
        total += 1
        req_id = r["request_id"]
        gold_id = r["gold_restaurant"]
        gold_idx = r["gold_idx"]
        gold_name = id_to_name.get(gold_id, "?")
        status = r["status"]

        if status == "ok":
            hits += 1
            console.print(f"  {req_id}: [green]✓[/green] [{gold_idx}] {gold_name}")
        elif status == "no_match":
            console.print(f"  {req_id}: [red]✗[/red] no match (gold=[{gold_idx}] {gold_name})")
        elif status == "multi_match":
            match_idxs = [id_to_idx.get(m, "?") for m in r["matches"][:3]]
            console.print(f"  {req_id}: [yellow]⚠[/yellow] multi ({r['matches_count']}): {match_idxs}")
        else:
            console.print(f"  {req_id}: [red]✗[/red] {status}")

    pct = hits / total * 100 if total else 0
    console.print(f"\n[bold]Validation: {hits}/{total} = {pct:.0f}%[/bold]")


def list_datasets() -> list[str]:
    """List available datasets in data/ directory."""
    datasets = []
    for p in DATA_DIR.iterdir():
        if p.is_dir() and not p.name.startswith(('_', '.')):
            # Check if it has required files
            if (p / "requests.jsonl").exists():
                datasets.append(p.name)
    return sorted(datasets)


def choose_dataset() -> str:
    """Interactively choose a dataset."""
    datasets = list_datasets()

    if not datasets:
        print("No datasets found in data/")
        sys.exit(1)

    if len(datasets) == 1:
        print(f"Using dataset: {datasets[0]}")
        return datasets[0]

    print("Available datasets:")
    for i, name in enumerate(datasets, 1):
        print(f"  [{i}] {name}")

    while True:
        try:
            choice = input("\nSelect dataset (number or name): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    return datasets[idx]
            elif choice in datasets:
                return choice
            print(f"Invalid choice. Enter 1-{len(datasets)} or dataset name.")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Validate ground truth dataset")
    parser.add_argument("name", nargs="?", help="Dataset name (e.g., philly_cafes)")
    args = parser.parse_args()

    name = args.name if args.name else choose_dataset()
    success = validate_dataset(name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
