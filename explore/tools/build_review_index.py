#!/usr/bin/env python3
"""
Build Review Index - One-time keyword indexing for all tasks.

Scans all reviews in dataset_K200.jsonl and computes keyword matches
for ALL tasks defined in pipeline_config.json.

Usage:
    python tools/build_review_index.py
    python tools/build_review_index.py --stats  # Show statistics only
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_FILE = DATA_DIR / "dataset_K200.jsonl"
CONFIG_FILE = DATA_DIR / "semantic_gt" / "pipeline_config.json"
INDEX_DIR = DATA_DIR / "semantic_gt" / "review_index"


def hash_text(text: str) -> str:
    """Create SHA256 hash of text (first 16 chars for brevity)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sanitize_filename(name: str) -> str:
    """Convert restaurant name to safe filename."""
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name).strip().replace(' ', '_')


def load_config():
    """Load pipeline configuration."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def load_source_data():
    """Load all restaurants from source file."""
    restaurants = []
    with open(SOURCE_FILE, 'r') as f:
        for line in f:
            if line.strip():
                restaurants.append(json.loads(line))
    return restaurants


def build_review_index():
    """
    Scan all reviews, compute keyword matches for ALL tasks.
    Run once per source file update.
    """
    config = load_config()
    restaurants = load_source_data()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        "total_restaurants": len(restaurants),
        "total_reviews": 0,
        "task_matches": defaultdict(lambda: {"reviews": 0, "restaurants": 0})
    }

    for restaurant in restaurants:
        name = restaurant["business"]["name"]
        reviews = restaurant.get("reviews", [])

        index = {
            "restaurant": name,
            "source_file": config["source_file"],
            "n_reviews": len(reviews),
            "indexed_at": datetime.now().isoformat(),
            "reviews": []
        }

        # Track per-restaurant task matches
        restaurant_task_matches = defaultdict(bool)

        for idx, review in enumerate(reviews):
            text_lower = review.get("text", "").lower()

            review_entry = {
                "idx": idx,
                "date": review.get("date", ""),
                "stars": review.get("stars", 0),
                "useful": review.get("useful", 0),
                "text_hash": hash_text(review.get("text", "")),
                "task_relevance": {}
            }

            # Check relevance for ALL tasks
            for task_id, task_config in config["tasks"].items():
                keywords = task_config.get("keywords", [])
                keywords_found = [k for k in keywords if k in text_lower]
                has_match = len(keywords_found) > 0

                review_entry["task_relevance"][task_id] = {
                    "keyword_match": has_match,
                    "keywords_found": keywords_found
                }

                if has_match:
                    stats["task_matches"][task_id]["reviews"] += 1
                    restaurant_task_matches[task_id] = True

            index["reviews"].append(review_entry)
            stats["total_reviews"] += 1

        # Update restaurant counts
        for task_id, has_match in restaurant_task_matches.items():
            if has_match:
                stats["task_matches"][task_id]["restaurants"] += 1

        # Save index
        filename = sanitize_filename(name) + ".json"
        with open(INDEX_DIR / filename, 'w') as f:
            json.dump(index, f, indent=2)

    # Save stats
    stats_summary = {
        "indexed_at": datetime.now().isoformat(),
        "source_file": config["source_file"],
        "source_hash": config["source_hash"],
        "total_restaurants": stats["total_restaurants"],
        "total_reviews": stats["total_reviews"],
        "tasks": {}
    }

    for task_id, matches in stats["task_matches"].items():
        task_config = config["tasks"][task_id]
        stats_summary["tasks"][task_id] = {
            "name": task_config["name"],
            "keywords_count": len(task_config.get("keywords", [])),
            "matching_reviews": matches["reviews"],
            "matching_restaurants": matches["restaurants"],
            "review_match_rate": f"{matches['reviews'] / stats['total_reviews'] * 100:.1f}%"
        }

    with open(INDEX_DIR / "_stats.json", 'w') as f:
        json.dump(stats_summary, f, indent=2)

    return stats_summary


def print_stats():
    """Print statistics from existing index."""
    stats_file = INDEX_DIR / "_stats.json"
    if not stats_file.exists():
        print("No index found. Run: python tools/build_review_index.py")
        return

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    print("=" * 60)
    print("REVIEW INDEX STATISTICS")
    print("=" * 60)
    print(f"Source: {stats['source_file']}")
    print(f"Indexed: {stats['indexed_at']}")
    print(f"Total Restaurants: {stats['total_restaurants']}")
    print(f"Total Reviews: {stats['total_reviews']}")
    print()

    print("TASK KEYWORD MATCHES:")
    print("-" * 60)
    for task_id, task_stats in stats.get("tasks", {}).items():
        print(f"\n  {task_id}: {task_stats['name']}")
        print(f"    Keywords: {task_stats['keywords_count']}")
        print(f"    Matching Reviews: {task_stats['matching_reviews']} ({task_stats['review_match_rate']})")
        print(f"    Restaurants with Matches: {task_stats['matching_restaurants']}/100")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Review Index")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        print("Building review index...")
        stats = build_review_index()
        print(f"\nIndexed {stats['total_restaurants']} restaurants, {stats['total_reviews']} reviews")
        print(f"\nTask matches:")
        for task_id, task_stats in stats.get("tasks", {}).items():
            print(f"  {task_id}: {task_stats['matching_reviews']} reviews ({task_stats['review_match_rate']})")
        print(f"\nIndex saved to: {INDEX_DIR}")
