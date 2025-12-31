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
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from data.loader import YELP_DIR
from utils.llm import call_llm
from utils.utils import loadjl

console = Console()


# --- Incremental Update Helpers ---

def load_existing_groundtruth(output_path: Path) -> dict:
    """Load existing groundtruth as {(item_id, request_id): entry}."""
    if not output_path.exists():
        return {}
    existing = {}
    for entry in loadjl(output_path):
        key = (entry["item_id"], entry["request_id"])
        existing[key] = entry
    return existing


def find_missing_pairs(item_ids: list, requests: list, existing: dict) -> list[tuple]:
    """Find (item_id, request_id) pairs not in existing groundtruth."""
    missing = []
    for item_id in item_ids:
        for req in requests:
            if (item_id, req["id"]) not in existing:
                missing.append((item_id, req))
    return missing


# --- Visualization ---

def print_groundtruth_matrix(entries: list, item_ids: list, request_ids: list):
    """Print items × requests matrix with colored labels."""
    lookup = {(e["item_id"], e["request_id"]): e["gold_label"] for e in entries}

    table = Table(title="Ground Truth Matrix", show_lines=False)
    table.add_column("Item", style="cyan", width=14)
    for req_id in request_ids:
        table.add_column(req_id, justify="center", width=5)

    for item_id in item_ids:
        row = [item_id[:12] + ".."]
        for req_id in request_ids:
            label = lookup.get((item_id, req_id), "?")
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


def print_statistics(stats: dict, per_request: dict = None):
    """Print distribution statistics."""
    total = sum(stats.values())
    if total == 0:
        return

    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total entries: {total}")
    console.print(f"  Recommend (+1): [green]{stats[1]:4d}[/green] ({stats[1]/total*100:5.1f}%)")
    console.print(f"  Unknown (0):    [yellow]{stats[0]:4d}[/yellow] ({stats[0]/total*100:5.1f}%)")
    console.print(f"  Not rec (-1):   [red]{stats[-1]:4d}[/red] ({stats[-1]/total*100:5.1f}%)")

    if per_request:
        console.print("\n[bold]Per-Request Breakdown:[/bold]")
        for req_id, req_stats in per_request.items():
            req_total = sum(req_stats.values())
            if req_total > 0:
                pct_pos = req_stats[1] / req_total * 100
                pct_neg = req_stats[-1] / req_total * 100
                console.print(f"  {req_id}: [green]+1:{req_stats[1]:2d}[/green] [yellow]0:{req_stats[0]:2d}[/yellow] [red]-1:{req_stats[-1]:2d}[/red]  ({pct_pos:.0f}%//{pct_neg:.0f}%)")


# --- Three-Value Logic ---

class TV(Enum):
    """Three-value logic: True, Unknown, False."""
    T = 1
    U = 0
    F = -1


def tv_and(a: TV, b: TV) -> TV:
    """Three-value AND: F dominates, then U, then T."""
    if a is TV.F or b is TV.F:
        return TV.F
    if a is TV.U or b is TV.U:
        return TV.U
    return TV.T


def tv_or(a: TV, b: TV) -> TV:
    """Three-value OR: T dominates, then U, then F."""
    if a is TV.T or b is TV.T:
        return TV.T
    if a is TV.U or b is TV.U:
        return TV.U
    return TV.F


def reduce_tv(op: str, values: list[TV]) -> TV:
    """Reduce list of TV values using AND or OR."""
    if not values:
        return TV.U
    result = values[0]
    for v in values[1:]:
        result = tv_and(result, v) if op == "AND" else tv_or(result, v)
    return result


# --- LLM Judgment Cache (shared across all datasets) ---

JUDGMENT_CACHE_PATH = YELP_DIR / "judgments_cache.jsonl"
_judgment_cache = {}


def load_judgment_cache():
    """Load cached LLM judgments from file (shared across datasets)."""
    global _judgment_cache
    _judgment_cache = {}
    if JUDGMENT_CACHE_PATH.exists():
        for entry in loadjl(JUDGMENT_CACHE_PATH):
            key = (entry["review_id"], entry["aspect"])
            _judgment_cache[key] = entry["judgment"]
        console.print(f"  Loaded {len(_judgment_cache)} cached judgments")


def save_judgment_cache():
    """Persist judgment cache to file."""
    with open(JUDGMENT_CACHE_PATH, "w") as f:
        for (review_id, aspect), judgment in sorted(_judgment_cache.items()):
            f.write(json.dumps({"review_id": review_id, "aspect": aspect, "judgment": judgment}) + "\n")
    console.print(f"  Saved {len(_judgment_cache)} judgments to cache")


def get_cached_judgment(review_id: str, aspect: str) -> int | None:
    """Get cached judgment or None."""
    return _judgment_cache.get((review_id, aspect))


def set_cached_judgment(review_id: str, aspect: str, judgment: int):
    """Cache a judgment."""
    _judgment_cache[(review_id, aspect)] = judgment


# --- LLM Review Judgment ---

def llm_judge_review(review_id: str, text: str, aspect: str) -> int:
    """Use LLM to judge a single review for an aspect (with caching).

    Returns: 1 (positive), 0 (neutral/unknown), -1 (negative)
    """
    # Check cache first
    cached = get_cached_judgment(review_id, aspect)
    if cached is not None:
        return cached

    prompt = f"""Does this review indicate "{aspect}"?

Review: {text}

Answer with exactly one of: +1 (positive), 0 (neutral/not mentioned), -1 (negative)"""

    response = call_llm(prompt)
    # Parse response to int
    if "+1" in response or response.strip() == "1":
        judgment = 1
    elif "-1" in response:
        judgment = -1
    else:
        judgment = 0

    # Cache result
    set_cached_judgment(review_id, aspect, judgment)
    return judgment


def aggregate_judgments(judgments: list[int], weights: list[float] = None) -> int:
    """Aggregate LLM judgments with optional weights.

    Uses weighted voting: sum(judgment * weight) / sum(weights)
    Then maps to {-1, 0, 1} based on thresholds.
    """
    if not judgments:
        return 0

    if weights is None:
        weights = [1.0] * len(judgments)

    weighted_sum = sum(j * w for j, w in zip(judgments, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        return 0

    avg = weighted_sum / total_weight
    if avg >= 0.5:
        return 1
    elif avg <= -0.5:
        return -1
    return 0


def evaluate_review_meta(reviews: list, path: list, match_list: list = None, aspect: str = "") -> tuple[int, list[float]]:
    """Lookup or match, return (score, per_review_weights).

    Weight rules:
    - friend match: 2.0, non-friend: 1.0
    - review_count: scale between 0.5 (low) to 1.5 (high)
    """
    weights = []

    for r in reviews:
        val = get_nested_value(r, path)
        if val is None:
            weights.append(1.0)  # default weight
            continue

        if match_list:
            # List match (e.g., friends)
            if isinstance(val, list) and any(f in match_list for f in val):
                weights.append(2.0)  # friend's review
            else:
                weights.append(1.0)  # non-friend (neutral)

        elif "review_count" in aspect:
            # Scale review_count to weight: 0.5 to 1.5
            # Assume typical range is 0-100 reviews
            count = val if isinstance(val, (int, float)) else 0
            weight = 0.5 + min(count / 100, 1.0)  # caps at 1.5
            weights.append(weight)

        else:
            weights.append(1.0)  # default

    # Compute overall score based on whether any match found
    if match_list:
        has_match = any(w > 1.0 for w in weights)
        score = 1 if has_match else 0  # 0 if no match (not -1)
    else:
        score = 0  # review_meta doesn't produce its own score

    return score, weights


# --- Attribute Parsing ---

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


def evaluate_condition(item: dict, condition: dict, reviews: list = None, review_weights: list[float] = None) -> tuple[int, dict]:
    """Evaluate a single condition against an item.

    Args:
        item: Restaurant item dict
        condition: Condition dict with aspect and evidence spec
        reviews: List of review dicts (for review_text/review_meta kinds)
        review_weights: Optional per-review weights for weighted aggregation

    Returns:
        (result, evidence) where:
        - result: 1 (satisfied), -1 (not satisfied), 0 (unknown)
        - evidence: {"kind": ..., "value": ..., "satisfied": ...}
    """
    aspect = condition.get("aspect", "")
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])
    match_list = evidence_spec.get("match_list", None)

    if kind == "item_meta":
        # Original behavior: lookup in item attributes
        value = get_nested_value(item, path)
        if value is None:
            return 0, {"kind": kind, "value": None, "satisfied": 0}
        satisfied = evaluate_aspect(aspect, value)

    elif kind == "review_text":
        # LLM judges each review, then aggregate with weights
        if not reviews:
            return 0, {"kind": kind, "value": None, "satisfied": 0}
        judgments = [llm_judge_review(r.get("review_id", ""), r.get("text", ""), aspect) for r in reviews]
        satisfied = aggregate_judgments(judgments, review_weights)
        value = {"judgments": judgments, "weights": review_weights}

    elif kind == "review_meta":
        # Lookup or list match - returns per-review weights for later use
        if not reviews:
            return 0, {"kind": kind, "value": None, "satisfied": 0}
        satisfied, per_review_weights = evaluate_review_meta(reviews, path, match_list, aspect)
        value = {"path": path, "match_list": match_list, "weights": per_review_weights}

    else:
        return 0, {"kind": kind, "value": None, "satisfied": 0}

    return satisfied, {"kind": kind, "value": value, "satisfied": satisfied}


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


def evaluate_structure(item: dict, structure: dict, reviews: list = None) -> tuple[int, dict]:
    """Evaluate full AND/OR structure with three-value logic.

    Args:
        item: Restaurant item dict
        structure: AND/OR structure with conditions
        reviews: List of review dicts (for review_text/review_meta kinds)

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
            result, nested_evidence = evaluate_structure(item, arg, reviews)
            all_evidence.update(nested_evidence)
        else:
            # Single condition
            result, evidence = evaluate_condition(item, arg, reviews)
            aspect = arg.get("aspect", "unknown")
            all_evidence[aspect] = evidence
        results.append(TV(result))

    # Apply three-value logic using explicit TV operations
    final = reduce_tv(op, results)
    return final.value, all_evidence


def generate_groundtruth(selection_name: str, limit: int = 10, force: bool = False):
    """Generate ground truth labels for a selection.

    Args:
        selection_name: e.g., "selection_1"
        limit: Max items to process (0 for all, default 10 for testing)
        force: If True, recompute all entries (ignore existing groundtruth)
    """
    n = selection_name.replace("selection_", "")

    # Define file paths
    restaurants_cache_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"
    selection_path = YELP_DIR / f"{selection_name}.jsonl"
    requests_path = YELP_DIR / f"requests_{n}.jsonl"
    rev_selection_path = YELP_DIR / f"rev_{selection_name}.jsonl"
    reviews_cache_path = YELP_DIR / f"reviews_cache_{n}.jsonl"
    output_path = YELP_DIR / f"groundtruth_{n}.jsonl"

    # Check required files exist
    required_files = [restaurants_cache_path, selection_path, requests_path]
    for p in required_files:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/red]")
            sys.exit(1)

    # Load data
    console.print(f"[cyan]Loading data...[/cyan]")
    restaurants = {r["business_id"]: r for r in loadjl(restaurants_cache_path)}
    selection = {item["item_id"]: item for item in loadjl(selection_path)}
    requests = loadjl(requests_path)
    request_ids = [r["id"] for r in requests]
    requests_lookup = {r["id"]: r for r in requests}

    # Load reviews (optional - only needed for review_text/review_meta kinds)
    reviews_by_item = {}
    if rev_selection_path.exists() and reviews_cache_path.exists():
        rev_selection = {r["item_id"]: r["review_ids"] for r in loadjl(rev_selection_path)}
        reviews_cache = {r["review_id"]: r for r in loadjl(reviews_cache_path)}
        for item_id, review_ids in rev_selection.items():
            reviews_by_item[item_id] = [reviews_cache[rid] for rid in review_ids if rid in reviews_cache]
        console.print(f"  Reviews loaded: {sum(len(v) for v in reviews_by_item.values())} total")
    else:
        console.print(f"  [yellow]No review files found - review_text/review_meta will return unknown[/yellow]")

    # Load LLM judgment cache (shared across datasets)
    load_judgment_cache()

    # Sort by llm_percent and apply limit
    sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))
    if limit > 0:
        sorted_ids = sorted_ids[:limit]

    console.print(f"  Restaurants: {len(sorted_ids)}")
    console.print(f"  Requests: {len(requests)}")

    # Load existing groundtruth (unless --force)
    existing = {} if force else load_existing_groundtruth(output_path)
    if existing:
        console.print(f"  Existing entries: {len(existing)}")

    # Find missing (item, request) pairs
    missing_pairs = find_missing_pairs(sorted_ids, requests, existing)
    console.print(f"  Missing pairs: {len(missing_pairs)}")

    if not missing_pairs:
        console.print("[green]Ground truth is up to date![/green]")
        # Show visualization of existing data
        all_entries = list(existing.values())
        stats = {1: 0, 0: 0, -1: 0}
        per_request = {rid: {1: 0, 0: 0, -1: 0} for rid in request_ids}
        for e in all_entries:
            label = e["gold_label"]
            stats[label] += 1
            if e["request_id"] in per_request:
                per_request[e["request_id"]][label] += 1

        print_groundtruth_matrix(all_entries, sorted_ids, request_ids)
        print_statistics(stats, per_request)
        return

    # Generate ground truth for missing pairs with progress bar
    console.print(f"[cyan]Computing ground truth...[/cyan]")
    new_entries = []
    stats = {1: 0, 0: 0, -1: 0}
    per_request = {rid: {1: 0, 0: 0, -1: 0} for rid in request_ids}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(missing_pairs))

        for item_id, req in missing_pairs:
            item = restaurants.get(item_id, {})
            reviews = reviews_by_item.get(item_id, [])
            structure = req.get("structure", {})

            gold_label, evidence = evaluate_structure(item, structure, reviews)

            new_entries.append({
                "item_id": item_id,
                "request_id": req["id"],
                "gold_label": gold_label,
                "evidence": evidence
            })
            stats[gold_label] += 1
            per_request[req["id"]][gold_label] += 1

            progress.update(task, advance=1, description=f"{item_id[:12]}.. × {req['id']}")

    # Save LLM judgment cache
    save_judgment_cache()

    # Merge with existing entries
    all_entries = list(existing.values()) + new_entries

    # Update stats to include existing entries
    for e in existing.values():
        label = e["gold_label"]
        stats[label] += 1
        if e["request_id"] in per_request:
            per_request[e["request_id"]][label] += 1

    # Write output (sorted by item_id, request_id)
    console.print(f"[cyan]Writing output...[/cyan]")
    all_entries_sorted = sorted(all_entries, key=lambda x: (x["item_id"], x["request_id"]))
    with open(output_path, "w") as f:
        for entry in all_entries_sorted:
            f.write(json.dumps(entry) + "\n")

    console.print(f"[green]Saved {len(all_entries)} entries to {output_path}[/green]")
    console.print(f"  (New: {len(new_entries)}, Existing: {len(existing)})")

    # Show visualization
    print_groundtruth_matrix(all_entries, sorted_ids, request_ids)
    print_statistics(stats, per_request)


def main():
    parser = argparse.ArgumentParser(description="Precompute ground truth labels")
    parser.add_argument("selection", help="Selection name (e.g., selection_1)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max items to process (0 for all, default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute all entries (ignore existing)")
    args = parser.parse_args()

    console.print(f"\n[bold]=== Ground Truth Generator ===[/bold]")
    console.print(f"Selection: {args.selection}")
    console.print(f"Limit: {args.limit if args.limit > 0 else 'all'}")
    if args.force:
        console.print(f"[yellow]Force mode: recomputing all entries[/yellow]")

    generate_groundtruth(args.selection, args.limit, args.force)


if __name__ == "__main__":
    main()
