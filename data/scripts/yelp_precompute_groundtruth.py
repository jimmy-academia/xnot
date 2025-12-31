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
import asyncio
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from data.loader import YELP_DIR
from utils.llm import call_llm
from utils.io import loadjl

console = Console()

# --- Async Infrastructure ---

_executor: ThreadPoolExecutor = None
_llm_semaphore: asyncio.Semaphore = None


def init_async_executor(max_workers: int = 50):
    """Initialize thread pool executor and semaphore for async LLM calls."""
    global _executor, _llm_semaphore
    _executor = ThreadPoolExecutor(max_workers=max_workers)
    _llm_semaphore = asyncio.Semaphore(max_workers)


# --- Incremental Update Helpers ---

def hash_request(req: dict) -> str:
    """Hash the request structure for change detection."""
    structure_str = json.dumps(req.get("structure", {}), sort_keys=True)
    return hashlib.md5(structure_str.encode()).hexdigest()[:8]


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
    """Find (item_id, request) pairs needing computation.

    A pair needs computation if:
    - It doesn't exist in groundtruth, OR
    - The request structure hash changed (request was modified)
    """
    missing = []
    for item_id in item_ids:
        for req in requests:
            key = (item_id, req["id"])
            if key not in existing:
                missing.append((item_id, req))
            elif existing[key].get("request_hash") != hash_request(req):
                # Request definition changed - needs recomputation
                missing.append((item_id, req))
    return missing


# --- Field-Level Edits System ---

def load_edits(n: str) -> tuple[dict, dict]:
    """Load field-level edits for restaurants and reviews.

    Args:
        n: Selection number (e.g., "1")

    Returns:
        (restaurant_edits, review_edits) where each is {id: {field_path: value}}
    """
    edits_path = YELP_DIR / f"edits_{n}.jsonl"
    if not edits_path.exists():
        return {}, {}

    restaurant_edits = {}  # {business_id: {"hours.Monday": "7:0-19:0"}}
    review_edits = {}      # {review_id: {"user.name": "Anon"}}

    for entry in loadjl(edits_path):
        entry = dict(entry)  # Copy to avoid modifying original
        entry_type = entry.pop("type", "restaurant")
        entry_id = entry.pop("id")
        if entry_type == "restaurant":
            restaurant_edits[entry_id] = entry
        else:
            review_edits[entry_id] = entry

    return restaurant_edits, review_edits


def apply_field_edits(obj: dict, edits: dict) -> dict:
    """Apply field-level patches to an object using dot notation.

    Args:
        obj: Original object (restaurant or review)
        edits: {field_path: new_value} e.g., {"hours.Monday": "7:0-19:0"}

    Returns:
        Modified copy of object
    """
    import copy
    result = copy.deepcopy(obj)

    for path, value in edits.items():
        keys = path.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    return result


def find_field_diffs(original: dict, modified: dict, prefix: str = "") -> dict:
    """Find field-level differences between two dicts.

    Args:
        original: Original object
        modified: Modified object
        prefix: Current path prefix for recursion

    Returns:
        {field_path: new_value} for changed fields
    """
    diffs = {}

    # Check all keys in modified
    for key in modified:
        path = f"{prefix}.{key}" if prefix else key
        orig_val = original.get(key)
        mod_val = modified[key]

        if isinstance(mod_val, dict) and isinstance(orig_val, dict):
            # Recurse into nested dicts
            nested_diffs = find_field_diffs(orig_val, mod_val, path)
            diffs.update(nested_diffs)
        elif mod_val != orig_val:
            # Value changed
            diffs[path] = mod_val

    return diffs


def extract_edits_from_backup(n: str) -> int:
    """Compare backup files with cache and extract differences as edits.

    Args:
        n: Selection number (e.g., "1")

    Returns:
        Number of edits extracted
    """
    edits = []

    # Extract restaurant edits
    backup_path = YELP_DIR / f"restaurants_selected_{n}.jsonl.backup"
    cache_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"

    if backup_path.exists() and cache_path.exists():
        backup = {r["business_id"]: r for r in loadjl(backup_path)}
        cache = {r["business_id"]: r for r in loadjl(cache_path)}

        for bid, modified in backup.items():
            original = cache.get(bid, {})
            diffs = find_field_diffs(original, modified)
            if diffs:
                edits.append({"type": "restaurant", "id": bid, **diffs})
                console.print(f"  [cyan]Restaurant {bid[:16]}:[/cyan] {len(diffs)} field(s) changed")

    # Extract review edits
    backup_path = YELP_DIR / f"reviews_selected_{n}.jsonl.backup"
    cache_path = YELP_DIR / f"reviews_cache_{n}.jsonl"

    if backup_path.exists() and cache_path.exists():
        backup = {r["review_id"]: r for r in loadjl(backup_path)}
        cache = {r["review_id"]: r for r in loadjl(cache_path)}

        for rid, modified in backup.items():
            original = cache.get(rid, {})
            diffs = find_field_diffs(original, modified)
            if diffs:
                edits.append({"type": "review", "id": rid, **diffs})
                console.print(f"  [cyan]Review {rid[:16]}:[/cyan] {len(diffs)} field(s) changed")

    # Write edits file
    if edits:
        edits_path = YELP_DIR / f"edits_{n}.jsonl"
        with open(edits_path, "w") as f:
            for edit in edits:
                f.write(json.dumps(edit) + "\n")
        console.print(f"[green]Saved {len(edits)} edits to {edits_path}[/green]")

    return len(edits)


# --- User ID Anonymization ---

def build_user_id_mapping(reviews_by_item: dict) -> dict:
    """Build consistent user_id -> User_XXX mapping.

    Collects all user_ids from reviews and assigns sequential User_001, User_002, etc.
    Sorting ensures deterministic ordering.

    Returns:
        {original_user_id: "User_001", ...}
    """
    all_user_ids = set()
    for reviews in reviews_by_item.values():
        for r in reviews:
            # user_id can be at top level or nested under "user"
            uid = r.get("user_id") or r.get("user", {}).get("user_id")
            if uid:
                all_user_ids.add(uid)

    # Sort for deterministic ordering, then assign sequential IDs
    sorted_ids = sorted(all_user_ids)
    return {uid: f"User_{i+1:03d}" for i, uid in enumerate(sorted_ids)}


def anonymize_friends(friends: list, user_mapping: dict, min_friends: int = 3, max_friends: int = 20) -> list:
    """Shorten friend list to 3-20 and map to User_XXX format.

    Only includes friends who exist in the user_mapping (i.e., are reviewers in our dataset).

    Args:
        friends: Original list of friend user_ids
        user_mapping: {original_user_id: "User_XXX"} mapping
        min_friends: Minimum friends to keep (default 3)
        max_friends: Maximum friends to keep (default 20)

    Returns:
        List of anonymized friend IDs like ["User_007", "User_123"]
    """
    import random

    if not friends:
        return []

    # Filter to friends who are in our reviewer set
    valid_friends = [f for f in friends if f in user_mapping]

    if not valid_friends:
        return []

    # Random subset of min_friends to max_friends
    target_count = random.randint(min_friends, min(max_friends, len(valid_friends)))
    selected = random.sample(valid_friends, min(target_count, len(valid_friends)))

    # Map to User_XXX format
    return [user_mapping[f] for f in selected]


def create_selected_files(n: str, sorted_ids: list, restaurants: dict, reviews_by_item: dict, user_mapping: dict = None) -> tuple[Path, Path] | None:
    """Create selected restaurants and reviews files in correct order.

    Args:
        n: Selection number (e.g., "1")
        sorted_ids: Ordered list of item_ids to include
        restaurants: {business_id: restaurant_dict}
        reviews_by_item: {item_id: [review_dicts]}

    Returns:
        (restaurants_path, reviews_path) if created, None if skipped
    """
    restaurants_path = YELP_DIR / f"restaurants_selected_{n}.jsonl"
    reviews_path = YELP_DIR / f"reviews_selected_{n}.jsonl"

    # Check if files exist and items match
    if restaurants_path.exists():
        existing_ids = []
        for r in loadjl(restaurants_path):
            existing_ids.append(r.get("business_id"))

        if existing_ids == sorted_ids:
            # Files exist and match - do nothing
            console.print(f"[dim]Selected files exist and match ({len(sorted_ids)} items)[/dim]")
            return restaurants_path, reviews_path

        # Files exist but items don't match - ask user
        console.print(f"\n[yellow]Selected files exist but items differ:[/yellow]")
        console.print(f"  Existing: {len(existing_ids)} items")
        console.print(f"  Expected: {len(sorted_ids)} items")
        response = input("Overwrite selected files? [y/N]: ").strip().lower()
        if response != 'y':
            console.print("[dim]Skipped creating selected files[/dim]")
            return None

    # Load field-level edits
    restaurant_edits, review_edits = load_edits(n)
    if restaurant_edits or review_edits:
        console.print(f"[cyan]Applying {len(restaurant_edits)} restaurant edits, {len(review_edits)} review edits[/cyan]")

    # Write restaurants in sorted order (with edits applied)
    with open(restaurants_path, "w") as f:
        for item_id in sorted_ids:
            restaurant = restaurants.get(item_id, {})
            # Apply field-level edits if any
            if item_id in restaurant_edits:
                restaurant = apply_field_edits(restaurant, restaurant_edits[item_id])
            f.write(json.dumps(restaurant) + "\n")

    # Write reviews grouped by restaurant in sorted order
    # Apply user_id anonymization if mapping provided, then apply edits
    with open(reviews_path, "w") as f:
        for item_id in sorted_ids:
            reviews = reviews_by_item.get(item_id, [])
            for review in reviews:
                # Deep copy to avoid modifying original
                review = json.loads(json.dumps(review))

                if user_mapping:
                    # Get original user_id (can be at top level or nested)
                    orig_user_id = review.get("user_id") or review.get("user", {}).get("user_id")

                    if orig_user_id and orig_user_id in user_mapping:
                        # Map user_id to User_XXX
                        if "user" in review:
                            review["user"]["user_id"] = user_mapping[orig_user_id]

                            # Anonymize friends list
                            friends = review["user"].get("friends", [])
                            if friends:
                                review["user"]["friends"] = anonymize_friends(friends, user_mapping)
                        else:
                            review["user_id"] = user_mapping[orig_user_id]

                # Apply field-level edits if any
                review_id = review.get("review_id")
                if review_id and review_id in review_edits:
                    review = apply_field_edits(review, review_edits[review_id])

                f.write(json.dumps(review) + "\n")

    console.print(f"[green]Created selected files ({len(sorted_ids)} items)[/green]")
    if user_mapping:
        console.print(f"  [dim]Anonymized {len(user_mapping)} user IDs[/dim]")
    if restaurant_edits or review_edits:
        console.print(f"  [dim]Applied {len(restaurant_edits) + len(review_edits)} field edits[/dim]")
    return restaurants_path, reviews_path


# --- Visualization ---

def print_groundtruth_matrix(entries: list, item_ids: list, request_ids: list):
    """Print items × requests matrix with colored labels.

    Splits into multiple tables if more than 10 requests (10 per table).
    """
    lookup = {(e["item_id"], e["request_id"]): e["gold_label"] for e in entries}

    # Split requests into chunks of 10
    chunk_size = 10
    request_chunks = [request_ids[i:i+chunk_size] for i in range(0, len(request_ids), chunk_size)]

    for chunk_idx, req_chunk in enumerate(request_chunks):
        if len(request_chunks) > 1:
            title = f"Ground Truth Matrix ({chunk_idx+1}/{len(request_chunks)})"
        else:
            title = "Ground Truth Matrix"

        table = Table(title=title, show_lines=False)
        table.add_column("Item", style="cyan", width=14)
        for req_id in req_chunk:
            table.add_column(req_id, justify="center", width=5)

        for item_id in item_ids:
            row = [item_id[:12] + ".."]
            for req_id in req_chunk:
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
        if chunk_idx < len(request_chunks) - 1:
            console.print()  # Add spacing between tables


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


def print_hits_at_k(entries: list, request_ids: list, item_ids: list, k: int = 1):
    """Print Hits@K ranking summary with item indices.

    Gold selection: First filter to gold_label=+1 items, then rank by total_score.
    This ensures the gold item is definitively good, not just high-scoring.

    Args:
        entries: List of groundtruth entries
        request_ids: List of request IDs to display
        item_ids: Ordered list of item IDs (for index lookup)
        k: Number of top items to check (default 1)
    """
    console.print(f"\n[bold]Hits@{k} Summary:[/bold]")
    hits = 0
    valid_requests = 0

    # Build index lookup (1-based)
    item_id_to_index = {item_id: i + 1 for i, item_id in enumerate(item_ids)}

    for req_id in request_ids:
        req_entries = [e for e in entries if e["request_id"] == req_id]
        if not req_entries:
            continue

        # Filter to only gold_label=+1 items first
        positive_entries = [e for e in req_entries if e["gold_label"] == 1]

        if not positive_entries:
            # No valid gold item for this request
            console.print(f"  {req_id}: [yellow]⚠[/yellow] No gold_label=+1 items")
            continue

        valid_requests += 1

        # Rank by total_score within gold_label=+1 items
        ranked = sorted(positive_entries, key=lambda x: -x.get("total_score", 0))
        top_item = ranked[0]

        # Always a hit since we filtered to gold_label=+1
        hits += 1

        score = top_item.get("total_score", 0)
        index = item_id_to_index.get(top_item["item_id"], 0)
        console.print(f"  {req_id}: [green]✓[/green] gold=[{index}] {top_item['item_id'][:16]} (score={score}, [green]+1[/green])")

    pct = hits / valid_requests * 100 if valid_requests else 0
    console.print(f"\n[bold]Hits@{k}: {hits}/{valid_requests} = {pct:.0f}%[/bold]")
    if valid_requests < len(request_ids):
        console.print(f"[yellow]({len(request_ids) - valid_requests} requests skipped - no gold_label=+1 items)[/yellow]")


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
    """Reduce list of TV values using AND, OR, or ANY.

    - AND: False dominates, then Unknown, then True
    - OR/ANY: True dominates, then Unknown, then False
    """
    if not values:
        return TV.U

    if op == "AND":
        # False dominates
        if any(v is TV.F for v in values):
            return TV.F
        if any(v is TV.U for v in values):
            return TV.U
        return TV.T
    else:  # OR or ANY - both use True dominates
        if any(v is TV.T for v in values):
            return TV.T
        if any(v is TV.U for v in values):
            return TV.U
        return TV.F


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


def _sync_llm_judge(text: str, aspect: str) -> int:
    """Synchronous LLM call for use in executor (no caching here)."""
    prompt = f"""Does this review indicate "{aspect}"?

Review: {text}

Answer with exactly one of: +1 (positive), 0 (neutral/not mentioned), -1 (negative)"""

    response = call_llm(prompt)
    if "+1" in response or response.strip() == "1":
        return 1
    elif "-1" in response:
        return -1
    return 0


async def llm_judge_review_async(review_id: str, text: str, aspect: str) -> int:
    """Async LLM judgment with rate limiting and caching.

    Returns: 1 (positive), 0 (neutral/unknown), -1 (negative)
    """
    # Check cache first (sync, fast)
    cached = get_cached_judgment(review_id, aspect)
    if cached is not None:
        return cached

    # Async LLM call with semaphore for rate limiting
    async with _llm_semaphore:
        loop = asyncio.get_event_loop()
        judgment = await loop.run_in_executor(_executor, _sync_llm_judge, text, aspect)

    # Cache result
    set_cached_judgment(review_id, aspect, judgment)
    return judgment


def aggregate_judgments(judgments: list[int], weights: list[float] = None) -> tuple[int, float]:
    """Aggregate LLM judgments using symmetric strong consensus (75% threshold).

    Returns (result, score) where:
    - result: 1 (positive consensus), 0 (mixed/no consensus), -1 (negative consensus)
    - score: depends on result:
        - If satisfied (+1): positive ratio (0.75 to 1.0)
        - If neutral (0): 0
        - If not satisfied (-1): -1

    Result rules (symmetric 75% threshold):
    - Positive (+1): pos_ratio >= 75% (strong positive consensus)
    - Negative (-1): neg_ratio >= 75% (strong negative consensus)
    - Mixed (0): otherwise (no clear consensus)

    This removes the "strict veto" logic - outliers don't override majority sentiment.

    Note: weights parameter kept for API compatibility but not used.
    """
    if not judgments:
        return 0, 0.0

    pos = sum(1 for j in judgments if j == 1)
    neg = sum(1 for j in judgments if j == -1)
    total_opinions = pos + neg

    if total_opinions == 0:
        return 0, 0.0

    pos_ratio = pos / total_opinions
    neg_ratio = neg / total_opinions

    # Symmetric 75% threshold for consensus
    if pos >= 1 and pos_ratio >= 0.75:
        result = 1   # Strong Consensus Positive
        score = pos_ratio  # Score = positive ratio for satisfied aspects
    elif neg >= 1 and neg_ratio >= 0.75:
        result = -1  # Strong Consensus Negative
        score = -1.0
    else:
        result = 0   # Mixed / No Consensus
        score = 0.0

    return result, score


def filter_reviews(reviews: list, filter_spec: dict) -> list:
    """Filter reviews before aggregation.

    Supported filters:
    - {"elite": "not_empty"} - only reviews from elite users
    - {"date": "min_year", "value": 2018} - only reviews from 2018+

    Args:
        reviews: List of review dicts
        filter_spec: Filter specification dict

    Returns:
        Filtered list of reviews
    """
    if not filter_spec:
        return reviews

    result = []
    for r in reviews:
        include = True

        # Elite filter
        if "elite" in filter_spec:
            user_elite = r.get("user", {}).get("elite", [])
            if filter_spec["elite"] == "not_empty":
                include = include and bool(user_elite)

        # Date filter
        if "date" in filter_spec and include:
            date_str = r.get("date", "")
            if filter_spec["date"] == "min_year":
                min_year = filter_spec.get("value", 0)
                try:
                    review_year = int(date_str[:4])
                    include = include and review_year >= min_year
                except (ValueError, TypeError):
                    include = False

        if include:
            result.append(r)

    return result


def calculate_reviewer_weight(user_meta: dict, review: dict = None, weight_fields: list = None) -> float:
    """Calculate weight for a review based on reviewer metadata.

    Only considers fields explicitly specified in weight_fields.

    Args:
        user_meta: User metadata dict with keys like review_count, average_stars, elite, fans
        weight_fields: List of fields to consider, e.g., ["review_count", "average_stars"]

    Returns:
        Weight multiplier (typically 0.5 to 2.0)
    """
    if not weight_fields:
        return 1.0

    weight = 1.0

    # review_count: scale 0.5 to 1.5 based on experience
    if "review_count" in weight_fields:
        count = user_meta.get("review_count", 0)
        weight *= 0.5 + min(count / 100, 1.0)  # caps at 1.5

    # average_stars: prefer moderate raters (not extreme 1 or 5)
    if "average_stars" in weight_fields:
        avg = user_meta.get("average_stars", 3.0)
        # Higher weight for 2.5-4.0 range (balanced reviewers)
        if 2.5 <= avg <= 4.0:
            weight *= 1.2
        elif avg < 2.0 or avg > 4.5:
            weight *= 0.8

    # elite status: boost elite reviewers
    if "elite" in weight_fields:
        elite_years = user_meta.get("elite", [])
        if elite_years:
            weight *= 1.0 + min(len(elite_years) * 0.1, 0.5)  # up to 1.5x

    # fans: slight boost for popular reviewers
    if "fans" in weight_fields:
        fans = user_meta.get("fans", 0)
        weight *= 1.0 + min(fans / 100, 0.3)  # up to 1.3x

    return weight


def evaluate_review_meta(reviews: list, path: list, match_list: list = None, aspect: str = "", weight_fields: list = None) -> tuple[int, list[float]]:
    """Lookup or match, return (score, per_review_weights).

    Weight rules:
    - friend match: 2.0, non-friend: 1.0
    - review_count: scale between 0.5 (low) to 1.5 (high)
    - If weight_fields specified, uses calculate_reviewer_weight()
    """
    weights = []

    for r in reviews:
        # Get user metadata for weight calculation
        user_meta = r.get("user", {})

        # If weight_fields specified, use the new weight calculation
        if weight_fields:
            weight = calculate_reviewer_weight(user_meta, weight_fields)
            weights.append(weight)
            continue

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


# --- Declarative item_meta Evaluation ---

def parse_time(time_str: str) -> int | None:
    """Parse time string like '6:30' or '16:0' to minutes since midnight."""
    if not time_str:
        return None
    parts = time_str.split(":")
    if len(parts) != 2:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes
    except ValueError:
        return None


def parse_hours_range(hours_str: str) -> tuple[int, int] | None:
    """Parse hours string like '6:30-16:0' to (start_minutes, end_minutes)."""
    if not hours_str or hours_str == "0:0-0:0":
        return None  # Closed
    parts = hours_str.split("-")
    if len(parts) != 2:
        return None
    start = parse_time(parts[0])
    end = parse_time(parts[1])
    if start is None or end is None:
        return None
    return (start, end)


def hours_contains_range(actual_hours: str, required_range: str) -> bool:
    """Check if actual business hours fully contain the required time range.

    Args:
        actual_hours: Business hours like "6:30-16:0"
        required_range: Required time range like "10:00-15:00"

    Returns:
        True if required_range falls entirely within actual_hours
    """
    actual = parse_hours_range(actual_hours)
    required = parse_hours_range(required_range)

    if actual is None or required is None:
        return False

    actual_start, actual_end = actual
    req_start, req_end = required

    # Required range must fall entirely within actual range
    return actual_start <= req_start and req_end <= actual_end


def match_value(actual, expected) -> bool:
    """Normalized string matching (case-insensitive contains)."""
    if expected is None:
        return False
    if isinstance(expected, bool):
        return actual == expected
    if isinstance(expected, list):
        return any(match_value(actual, e) for e in expected)
    # String: case-insensitive contains
    return str(expected).lower() in str(actual).lower()


def evaluate_item_meta_rule(value, evidence_spec) -> int:
    """Evaluate item_meta using declarative rules from evidence_spec.

    Rules:
    - Missing value → use "missing" field (default 0=neutral)
    - Dict of booleans → OR across children (e.g., BusinessParking)
    - None given → default boolean check (True→1, False→-1, else→0)
    - Only "true" given → match→1, no match→-1
    - Only "false" given → match→-1, no match→1 (negative logic)
    - Multiple given → check each, none matched→0
    """
    true_cond = evidence_spec.get("true")
    false_cond = evidence_spec.get("false")
    neutral_cond = evidence_spec.get("neutral")
    missing_val = evidence_spec.get("missing", 0)  # default: neutral

    # Missing value → use "missing" field (default 0=neutral)
    if value is None:
        return missing_val

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


def evaluate_condition(item: dict, condition: dict, reviews: list = None, review_weights: list[float] = None) -> tuple[int, int, dict]:
    """Evaluate a single condition against an item.

    Args:
        item: Restaurant item dict
        condition: Condition dict with aspect and evidence spec
        reviews: List of review dicts (for review_text/review_meta kinds)
        review_weights: Optional per-review weights for weighted aggregation

    Returns:
        (result, score, evidence) where:
        - result: 1 (satisfied), -1 (not satisfied), 0 (unknown)
        - score: numeric score for ranking
        - evidence: {"kind": ..., "value": ..., "satisfied": ..., "score": ...}
    """
    aspect = condition.get("aspect", "")
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])
    match_list = evidence_spec.get("match_list", None)

    if kind == "item_meta":
        # Declarative rule-based evaluation
        value = get_nested_value(item, path)
        satisfied = evaluate_item_meta_rule(value, evidence_spec)
        score = satisfied

    elif kind == "item_meta_hours":
        # Hours containment check: does actual hours contain required range?
        value = get_nested_value(item, path)
        required_range = evidence_spec.get("true", "")
        if value and hours_contains_range(str(value), required_range):
            satisfied = 1   # Hours contain required range
        else:
            satisfied = -1  # Missing or doesn't contain → false (no neutral)
        score = satisfied

    elif kind == "review_text":
        # LLM judges each review, then aggregate with weights
        if not reviews:
            return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}
        judgments = [llm_judge_review(r.get("review_id", ""), r.get("text", ""), aspect) for r in reviews]
        satisfied, score = aggregate_judgments(judgments, review_weights)
        value = {"judgments": judgments, "weights": review_weights}

    elif kind == "review_meta":
        # Lookup or list match - returns per-review weights for later use
        if not reviews:
            return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}
        satisfied, per_review_weights = evaluate_review_meta(reviews, path, match_list, aspect)
        score = satisfied
        value = {"path": path, "match_list": match_list, "weights": per_review_weights}

    else:
        return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}

    return satisfied, score, {"kind": kind, "value": value, "satisfied": satisfied, "score": score}


def evaluate_structure(item: dict, structure: dict, reviews: list = None) -> tuple[int, int, dict]:
    """Evaluate full AND/OR structure with three-value logic.

    Args:
        item: Restaurant item dict
        structure: AND/OR structure with conditions
        reviews: List of review dicts (for review_text/review_meta kinds)

    Returns:
        (result, score, evidence) where:
        - result: 1 (recommend), -1 (not recommend), 0 (unknown)
        - score: aggregated score based on operator
        - evidence: dict of aspect → {value, satisfied, score}
    """
    op = structure.get("op", "AND")
    args = structure.get("args", [])

    all_evidence = {}
    child_results = []  # list of (TV, score)

    for arg in args:
        if "op" in arg:
            # Nested structure
            result, score, nested_evidence = evaluate_structure(item, arg, reviews)
            all_evidence.update(nested_evidence)
        else:
            # Single condition
            result, score, evidence = evaluate_condition(item, arg, reviews)
            aspect = arg.get("aspect", "unknown")
            all_evidence[aspect] = evidence
        child_results.append((TV(result), score))

    # Boolean logic
    final_tv = reduce_tv(op, [r for r, s in child_results])

    # Score logic
    scores = [s for r, s in child_results]
    positive_scores = [s for s in scores if s > 0]

    if op == "AND":
        final_score = min(scores) if scores else 0
    elif op == "OR":
        final_score = max(positive_scores) if positive_scores else 0
    else:  # ANY
        final_score = sum(positive_scores)

    return final_tv.value, final_score, all_evidence


# --- Async Evaluation (parallel LLM calls) ---

async def evaluate_condition_async(item: dict, condition: dict, reviews: list = None, review_weights: list[float] = None) -> tuple[int, float, dict]:
    """Async version of evaluate_condition - parallelizes LLM calls for review_text.

    Returns (result, score, evidence) where:
    - result: 1 (satisfied), 0 (unknown), -1 (not satisfied)
    - score: positive ratio (0.0-1.0) for review_text, result value for others
    - evidence: dict with kind, value, satisfied, score
    """
    aspect = condition.get("aspect", "")
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])
    match_list = evidence_spec.get("match_list", None)

    if kind == "item_meta":
        # Sync - no LLM calls, use declarative rule-based evaluation
        value = get_nested_value(item, path)
        satisfied = evaluate_item_meta_rule(value, evidence_spec)
        score = satisfied  # item_meta score = its result value
        return satisfied, score, {"kind": kind, "value": value, "satisfied": satisfied, "score": score}

    elif kind == "item_meta_hours":
        # Hours containment check: does actual hours contain required range?
        value = get_nested_value(item, path)
        required_range = evidence_spec.get("true", "")
        if value and hours_contains_range(str(value), required_range):
            satisfied = 1   # Hours contain required range
        else:
            satisfied = -1  # Missing or doesn't contain → false (no neutral)
        score = satisfied
        return satisfied, score, {"kind": kind, "value": value, "satisfied": satisfied, "score": score}

    elif kind == "review_text":
        # Async - parallel LLM calls for all reviews
        if not reviews:
            return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}
        tasks = [llm_judge_review_async(r.get("review_id", ""), r.get("text", ""), aspect) for r in reviews]
        judgments = await asyncio.gather(*tasks)
        satisfied, score = aggregate_judgments(list(judgments), review_weights)
        value = {"judgments": list(judgments), "weights": review_weights}
        return satisfied, score, {"kind": kind, "value": value, "satisfied": satisfied, "score": score}

    elif kind == "review_meta":
        # Sync - no LLM calls
        if not reviews:
            return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}
        satisfied, per_review_weights = evaluate_review_meta(reviews, path, match_list, aspect)
        score = satisfied  # review_meta score = its result value
        value = {"path": path, "match_list": match_list, "weights": per_review_weights}
        return satisfied, score, {"kind": kind, "value": value, "satisfied": satisfied, "score": score}

    return 0, 0, {"kind": kind, "value": None, "satisfied": 0, "score": 0}


async def evaluate_structure_async(item: dict, structure: dict, reviews: list = None) -> tuple[int, float, dict]:
    """Async version of evaluate_structure - parallelizes condition evaluation.

    Returns (result, score, evidence) where:
    - result: 1 (satisfied), 0 (unknown), -1 (not satisfied)
    - score: aggregated ratio based on operator (AND=min, OR=max(positive), ANY=sum(positive ratios))
    - evidence: dict of aspect → {kind, value, satisfied, score}
    """
    op = structure.get("op", "AND")
    args = structure.get("args", [])

    all_evidence = {}
    child_results = []  # list of (TV, score)

    # Evaluate all conditions in parallel
    async def eval_arg(arg):
        if "op" in arg:
            return await evaluate_structure_async(item, arg, reviews)
        else:
            return await evaluate_condition_async(item, arg, reviews)

    eval_results = await asyncio.gather(*[eval_arg(arg) for arg in args])

    for arg, (result, score, evidence) in zip(args, eval_results):
        if "op" in arg:
            all_evidence.update(evidence)
        else:
            aspect = arg.get("aspect", "unknown")
            all_evidence[aspect] = evidence
        child_results.append((TV(result), score))

    # Boolean logic
    final_tv = reduce_tv(op, [r for r, s in child_results])

    # Score logic
    scores = [s for r, s in child_results]
    positive_scores = [s for s in scores if s > 0]

    if op == "AND":
        final_score = min(scores) if scores else 0
    elif op == "OR":
        final_score = max(positive_scores) if positive_scores else 0
    else:  # ANY
        final_score = sum(positive_scores)

    return final_tv.value, final_score, all_evidence


# --- Parallel Restaurant Processing ---

async def process_restaurant(
    item_id: str,
    requests_for_item: list,
    restaurants: dict,
    reviews_by_item: dict,
    progress: Progress,
    task_id
) -> list[dict]:
    """Process specified requests for one restaurant asynchronously."""
    item = restaurants.get(item_id, {})
    reviews = reviews_by_item.get(item_id, [])
    entries = []

    for req in requests_for_item:
        structure = req.get("structure", {})
        gold_label, total_score, evidence = await evaluate_structure_async(item, structure, reviews)
        entries.append({
            "item_id": item_id,
            "request_id": req["id"],
            "request_hash": hash_request(req),
            "gold_label": gold_label,
            "total_score": total_score,
            "evidence": evidence
        })
        progress.update(task_id, advance=1)

    return entries


async def process_all_parallel(
    missing_by_item: dict,
    restaurants: dict,
    reviews_by_item: dict,
    progress: Progress,
    max_workers: int = 20
) -> list[dict]:
    """Process all restaurants in parallel with worker pool.

    Args:
        missing_by_item: {item_id: [list of requests to process]}
    """
    worker_semaphore = asyncio.Semaphore(max_workers)

    async def worker(item_id, requests_for_item):
        async with worker_semaphore:
            task_id = progress.add_task(
                f"[cyan]{item_id[:14]}[/cyan]",
                total=len(requests_for_item),
            )
            try:
                result = await process_restaurant(
                    item_id, requests_for_item, restaurants, reviews_by_item, progress, task_id
                )
                progress.update(task_id, description=f"[green]✓ {item_id[:14]}[/green]")
                return result
            finally:
                # Remove task line after a short delay
                await asyncio.sleep(0.1)
                progress.remove_task(task_id)

    # Run all workers
    results = await asyncio.gather(*[
        worker(item_id, reqs) for item_id, reqs in missing_by_item.items()
    ])

    # Flatten results
    return [entry for batch in results for entry in batch]


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
    required_files = [selection_path, requests_path]
    for p in required_files:
        if not p.exists():
            console.print(f"[red]Missing: {p}[/red]")
            sys.exit(1)

    # Load selection to get sorted_ids first (needed for data file creation)
    selection = {item["item_id"]: item for item in loadjl(selection_path)}
    sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))
    if limit > 0:
        sorted_ids = sorted_ids[:limit]
    sorted_ids_set = set(sorted_ids)

    # Check for selected files (editable) or fall back to cache files
    restaurants_selected_path = YELP_DIR / f"restaurants_selected_{n}.jsonl"
    reviews_selected_path = YELP_DIR / f"reviews_selected_{n}.jsonl"

    if restaurants_selected_path.exists() and reviews_selected_path.exists():
        # Load from selected files (editable)
        console.print(f"[cyan]Loading data from selected files (editable)...[/cyan]")
        console.print(f"  [green]Using: {restaurants_selected_path.name}[/green]")
        console.print(f"  [green]Using: {reviews_selected_path.name}[/green]")

        # Load restaurants - order in file = sorted_ids order
        restaurants_list = loadjl(restaurants_selected_path)
        restaurants = {r["business_id"]: r for r in restaurants_list}

        # Load reviews grouped by business_id
        reviews_list = loadjl(reviews_selected_path)
        reviews_by_item = {}
        for r in reviews_list:
            bid = r["business_id"]
            if bid not in reviews_by_item:
                reviews_by_item[bid] = []
            reviews_by_item[bid].append(r)
    else:
        # Fall back to cache files
        console.print(f"[cyan]Loading data from cache files...[/cyan]")

        cache_files = [restaurants_cache_path, rev_selection_path, reviews_cache_path]
        missing_cache = [p for p in cache_files if not p.exists()]
        if missing_cache:
            console.print(f"[red]Missing cache files: {[p.name for p in missing_cache]}[/red]")
            sys.exit(1)

        # Load restaurants
        restaurants = {r["business_id"]: r for r in loadjl(restaurants_cache_path)}
        restaurants = {k: v for k, v in restaurants.items() if k in sorted_ids_set}

        # Load reviews via rev_selection mapping
        rev_selection = {r["item_id"]: r["review_ids"] for r in loadjl(rev_selection_path)}
        all_reviews = {r["review_id"]: r for r in loadjl(reviews_cache_path)}
        reviews_by_item = {}
        for item_id in sorted_ids:
            review_ids = rev_selection.get(item_id, [])
            reviews_by_item[item_id] = [all_reviews[rid] for rid in review_ids if rid in all_reviews]

    requests = loadjl(requests_path)
    request_ids = [r["id"] for r in requests]
    requests_lookup = {r["id"]: r for r in requests}

    console.print(f"  Restaurants: {len(restaurants)}")
    console.print(f"  Reviews: {sum(len(v) for v in reviews_by_item.values())}")
    console.print(f"  Requests: {len(requests)}")

    # Load LLM judgment cache (shared across datasets)
    load_judgment_cache()

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
        print_hits_at_k(all_entries, request_ids, sorted_ids)

        # Build user_id mapping for anonymization
        user_mapping = build_user_id_mapping(reviews_by_item)

        # Create selected files (ordered by sorted_ids) if needed
        result = create_selected_files(n, sorted_ids, restaurants, reviews_by_item, user_mapping)

        # Print file paths at the bottom
        if result:
            rest_path, rev_path = result
            console.print(f"\n[bold]Selected Data Files:[/bold]")
            console.print(f"  {rest_path}")
            console.print(f"  {rev_path}")
        return

    # Initialize async executor for parallel LLM calls (conservative limit)
    init_async_executor(max_workers=50)

    # Group missing pairs by item_id -> list of requests to process
    missing_by_item = {}
    for item_id, req in missing_pairs:
        if item_id not in missing_by_item:
            missing_by_item[item_id] = []
        missing_by_item[item_id].append(req)

    items_to_process = list(missing_by_item.keys())

    # Generate ground truth with parallel processing
    console.print(f"[cyan]Computing ground truth ({len(items_to_process)} restaurants, {len(missing_pairs)} pairs)...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Run async processing
        new_entries = asyncio.run(process_all_parallel(
            missing_by_item,
            restaurants,
            reviews_by_item,
            progress,
            max_workers=10
        ))

    # Compute stats for new entries
    stats = {1: 0, 0: 0, -1: 0}
    per_request = {rid: {1: 0, 0: 0, -1: 0} for rid in request_ids}
    for entry in new_entries:
        label = entry["gold_label"]
        stats[label] += 1
        if entry["request_id"] in per_request:
            per_request[entry["request_id"]][label] += 1

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

    # Save metadata file for data loader to use
    from datetime import datetime
    meta_path = YELP_DIR / f"groundtruth_{n}_meta.json"
    metadata = {
        "item_count": len(sorted_ids),
        "item_ids": sorted_ids,
        "selection_name": selection_name,
        "created_at": datetime.now().isoformat()
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[green]Saved metadata to {meta_path}[/green]")

    # Show visualization
    print_groundtruth_matrix(all_entries, sorted_ids, request_ids)
    print_statistics(stats, per_request)
    print_hits_at_k(all_entries, request_ids, sorted_ids)

    # Build user_id mapping for anonymization
    user_mapping = build_user_id_mapping(reviews_by_item)

    # Create selected files (ordered by sorted_ids) if needed
    result = create_selected_files(n, sorted_ids, restaurants, reviews_by_item, user_mapping)

    # Print file paths at the bottom
    if result:
        rest_path, rev_path = result
        console.print(f"\n[bold]Selected Data Files:[/bold]")
        console.print(f"  {rest_path}")
        console.print(f"  {rev_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute ground truth labels")
    parser.add_argument("selection", help="Selection name (e.g., selection_1)")
    parser.add_argument("--limit", type=int, default=20,
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
