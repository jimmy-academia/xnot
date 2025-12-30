#!/usr/bin/env python3
"""Review sampler for Yelp data selection.

Given a selection_n.jsonl file (restaurant IDs + LLM scores), this script:
1. Extracts restaurant metadata from raw business file
2. Extracts all reviews for selected restaurants from raw review file
3. Extracts user metadata for reviewers from raw user file
4. Samples 20 reviews per restaurant (4 per star bucket, longest first)
5. Writes: rev_selection_n.jsonl, reviews_cache_n.jsonl, restaurants_cache_n.jsonl

Usage:
    python data/scripts/yelp_review_sampler.py selection_1
    python data/scripts/yelp_review_sampler.py selection_1 --per-bucket 4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Paths
YELP_DIR = Path("data/yelp")
RAW_DIR = YELP_DIR / "raw"
BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
USER_FILE = RAW_DIR / "yelp_academic_dataset_user.json"

console = Console()


def load_selection(selection_name: str) -> list[dict]:
    """Load selection file and return list of {item_id, llm_percent, llm_reasoning}."""
    selection_path = YELP_DIR / f"{selection_name}.jsonl"
    if not selection_path.exists():
        console.print(f"[red]Selection file not found: {selection_path}[/red]")
        sys.exit(1)

    items = []
    with open(selection_path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def extract_restaurants(business_ids: set) -> dict[str, dict]:
    """Scan raw business file, extract metadata for target businesses."""
    restaurants = {}
    total_lines = sum(1 for _ in open(BUSINESS_FILE))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scanning business file...", total=total_lines)

        with open(BUSINESS_FILE) as f:
            for line in f:
                progress.advance(task)
                if line.strip():
                    biz = json.loads(line)
                    if biz["business_id"] in business_ids:
                        restaurants[biz["business_id"]] = biz

    console.print(f"[green]Extracted {len(restaurants)} restaurants[/green]")
    return restaurants


def extract_reviews(business_ids: set) -> dict[str, list[dict]]:
    """Scan raw review file, extract all reviews for target businesses."""
    reviews_by_biz = defaultdict(list)
    total_lines = 6990280  # Known count to avoid re-scanning

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scanning review file (5.3GB)...", total=total_lines)

        with open(REVIEW_FILE) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    progress.update(task, completed=i)
                if line.strip():
                    review = json.loads(line)
                    if review["business_id"] in business_ids:
                        reviews_by_biz[review["business_id"]].append(review)

        progress.update(task, completed=total_lines)

    total_reviews = sum(len(r) for r in reviews_by_biz.values())
    console.print(f"[green]Extracted {total_reviews} reviews for {len(reviews_by_biz)} restaurants[/green]")
    return dict(reviews_by_biz)


def extract_users(user_ids: set) -> dict[str, dict]:
    """Scan raw user file, extract metadata for target users."""
    users = {}
    total_lines = 1987897  # Approximate count

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scanning user file (3.4GB)...", total=total_lines)

        with open(USER_FILE) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    progress.update(task, completed=i)
                if line.strip():
                    user = json.loads(line)
                    if user["user_id"] in user_ids:
                        # Extract relevant fields
                        users[user["user_id"]] = {
                            "user_id": user["user_id"],
                            "name": user.get("name", ""),
                            "review_count": user.get("review_count", 0),
                            "yelping_since": user.get("yelping_since", ""),
                            "friends": user.get("friends", "").split(", ") if user.get("friends") else [],
                            "elite": user.get("elite", "").split(",") if user.get("elite") else [],
                            "average_stars": user.get("average_stars", 0),
                            "fans": user.get("fans", 0),
                        }

        progress.update(task, completed=total_lines)

    console.print(f"[green]Extracted {len(users)} users[/green]")
    return users


def check_cache_valid(selection_name: str, selection_ids: set) -> bool:
    """Check if cache files exist and match selection."""
    n = selection_name.replace("selection_", "")
    rev_path = YELP_DIR / f"rev_{selection_name}.jsonl"
    reviews_path = YELP_DIR / f"reviews_cache_{n}.jsonl"
    restaurants_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"

    # All files must exist
    if not all(p.exists() for p in [rev_path, reviews_path, restaurants_path]):
        return False

    # Load item_ids from rev_selection
    cached_ids = set()
    with open(rev_path) as f:
        for line in f:
            if line.strip():
                cached_ids.add(json.loads(line)["item_id"])

    return cached_ids == selection_ids


def sample_reviews(reviews: list[dict], per_bucket: int = 4) -> list[str]:
    """Sample top N longest reviews from each star bucket (1-5).

    Returns list of review_id strings in order: 1-star, 2-star, ..., 5-star
    """
    # Bucket by stars
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for r in reviews:
        star = int(r.get("stars", 3))
        star = max(1, min(5, star))  # Clamp to 1-5
        buckets[star].append(r)

    # Sort each bucket by text length descending
    for star in buckets:
        buckets[star].sort(key=lambda r: -len(r.get("text", "")))

    # Take top N from each bucket
    sampled_ids = []
    for star in range(1, 6):
        for r in buckets[star][:per_bucket]:
            sampled_ids.append(r["review_id"])

    return sampled_ids


def main():
    parser = argparse.ArgumentParser(description="Sample reviews for Yelp selection")
    parser.add_argument("selection", help="Selection name (e.g., selection_1)")
    parser.add_argument("--per-bucket", type=int, default=4,
                        help="Reviews per star bucket (default: 4, total: 20)")
    args = parser.parse_args()

    selection_name = args.selection
    per_bucket = args.per_bucket

    console.print(f"\n[bold]Yelp Review Sampler[/bold]")
    console.print(f"Selection: {selection_name}")
    console.print(f"Reviews per bucket: {per_bucket} (total: {per_bucket * 5})")

    # Step 1: Load selection
    console.print(f"\n[bold]Step 1: Loading selection...[/bold]")
    selection = load_selection(selection_name)
    business_ids = {item["item_id"] for item in selection}
    console.print(f"Found {len(business_ids)} restaurants in selection")

    # Check if cache is valid
    if check_cache_valid(selection_name, business_ids):
        console.print("[yellow]Cache files already exist and match selection.[/yellow]")
        response = console.input("[bold]Regenerate? (y/N): [/bold]")
        if response.lower() != "y":
            console.print("[green]Skipping regeneration.[/green]")
            return
        console.print("[cyan]Regenerating...[/cyan]")

    # Step 2: Extract restaurant metadata
    console.print(f"\n[bold]Step 2: Extracting restaurant metadata...[/bold]")
    restaurants = extract_restaurants(business_ids)

    # Step 3: Extract reviews
    console.print(f"\n[bold]Step 3: Extracting reviews...[/bold]")
    reviews_by_biz = extract_reviews(business_ids)

    # Step 4: Collect user IDs and extract user metadata
    console.print(f"\n[bold]Step 4: Extracting user metadata...[/bold]")
    user_ids = set()
    for reviews in reviews_by_biz.values():
        for r in reviews:
            user_ids.add(r["user_id"])
    console.print(f"Found {len(user_ids)} unique reviewers")
    users = extract_users(user_ids)

    # Step 5: Sample reviews and build output
    console.print(f"\n[bold]Step 5: Sampling reviews...[/bold]")
    rev_selection = []  # {item_id, review_ids}
    reviews_cache = []  # All sampled reviews with user metadata
    sampled_review_ids = set()

    for item in selection:
        biz_id = item["item_id"]
        reviews = reviews_by_biz.get(biz_id, [])

        if not reviews:
            console.print(f"[yellow]Warning: No reviews for {biz_id}[/yellow]")
            continue

        # Sample reviews
        sampled_ids = sample_reviews(reviews, per_bucket)
        rev_selection.append({
            "item_id": biz_id,
            "review_ids": sampled_ids
        })
        sampled_review_ids.update(sampled_ids)

    # Build reviews cache with user metadata
    for reviews in reviews_by_biz.values():
        for r in reviews:
            if r["review_id"] in sampled_review_ids:
                user_meta = users.get(r["user_id"], {})
                reviews_cache.append({
                    "review_id": r["review_id"],
                    "business_id": r["business_id"],
                    "user_id": r["user_id"],
                    "stars": r["stars"],
                    "text": r["text"],
                    "date": r.get("date", ""),
                    "useful": r.get("useful", 0),
                    "funny": r.get("funny", 0),
                    "cool": r.get("cool", 0),
                    "user": user_meta
                })

    # Step 6: Write output files
    console.print(f"\n[bold]Step 6: Writing output files...[/bold]")

    # rev_selection_n.jsonl
    rev_selection_path = YELP_DIR / f"rev_{selection_name}.jsonl"
    with open(rev_selection_path, "w") as f:
        for item in rev_selection:
            f.write(json.dumps(item) + "\n")
    console.print(f"[green]Wrote {rev_selection_path}[/green]")

    # reviews_cache_n.jsonl
    reviews_cache_path = YELP_DIR / f"reviews_cache_{selection_name.replace('selection_', '')}.jsonl"
    with open(reviews_cache_path, "w") as f:
        for r in reviews_cache:
            f.write(json.dumps(r) + "\n")
    console.print(f"[green]Wrote {reviews_cache_path} ({len(reviews_cache)} reviews)[/green]")

    # restaurants_cache_n.jsonl
    restaurants_cache_path = YELP_DIR / f"restaurants_cache_{selection_name.replace('selection_', '')}.jsonl"
    with open(restaurants_cache_path, "w") as f:
        for biz_id in business_ids:
            if biz_id in restaurants:
                f.write(json.dumps(restaurants[biz_id]) + "\n")
    console.print(f"[green]Wrote {restaurants_cache_path} ({len(restaurants)} restaurants)[/green]")

    # Summary
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Restaurants: {len(rev_selection)}")
    console.print(f"  Sampled reviews: {len(reviews_cache)}")
    console.print(f"  Reviews per restaurant: {per_bucket * 5}")


if __name__ == "__main__":
    main()
