#!/usr/bin/env python3
"""Request generator for Yelp data selection.

Given a selection_n, this script:
1. Loads the dataset (items with reviews)
2. Computes distributions (stars, categories, cities, reviewer metadata)
3. Samples representative reviews
4. Prints data context and 4 level prompts to console

Levels (5 requests each, 20 total):
- L1 (R0-R4): Text conditions only (review_text source) with AND/OR
- L2 (R5-R9): Text + item meta conditions with AND/OR
- L3 (R10-R14): Nested text + item meta (AND containing OR or vice versa)
- L4 (R15-R19): Nested text + item meta + user social (reviewer_meta, review_meta)

Output format: JSONL (one JSON per line)

Usage:
    python data/scripts/yelp_requests.py selection_1
    # Then copy DATA CONTEXT + one LEVEL PROMPT to ChatGPT
    # Save output to data/yelp/requests_1.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from data.loader import load_yelp_dataset, YELP_DIR

console = Console()

# Sample aspect keywords for detection
ASPECT_KEYWORDS = {
    "food": ["food", "delicious", "tasty", "bland", "fresh", "flavor", "dish", "meal"],
    "service": ["service", "staff", "waiter", "server", "attentive", "rude", "friendly"],
    "speed": ["fast", "quick", "slow", "wait", "waited", "prompt", "forever"],
    "value": ["price", "value", "expensive", "cheap", "worth", "overpriced", "affordable"],
    "ambiance": ["atmosphere", "ambiance", "cozy", "loud", "noisy", "romantic", "vibe"],
    "portions": ["portion", "serving", "generous", "small", "huge", "filling"],
    "parking": ["parking", "park", "valet"],
    "outdoor": ["outdoor", "patio", "outside"],
}

# Allowed aspects by source type
ALLOWED_ASPECTS = {
    "review_text": [
        "food_quality", "service", "speed", "value",
        "ambiance", "portions", "parking", "outdoor"
    ],
    "item_meta": [
        # Categories
        "Sandwiches", "Cafes", "Coffee & Tea", "Bakeries",
        "Breakfast & Brunch", "American (New)", "Nightlife",
        # Attributes
        "WiFi", "BusinessAcceptsCreditCards", "RestaurantsReservations",
        "outdoor_seating", "BusinessParking", "BusinessParking_garage",
        "BusinessParking_lot", "BusinessParking_street", "RestaurantsPriceRange2",
        "stars", "review_count"
    ],
    "reviewer_meta": [
        "friends_count", "common_friend", "reviewer_review_count"
        # common_friend takes user_ids param: list of user_ids to check network overlap
    ],
    "review_meta": [
        "stars", "date", "useful_votes", "funny_votes", "cool_votes"
    ]
}

# Hard-coded level prompts for request generation
LEVEL_PROMPTS = {
    1: """Generate 5 restaurant requests using ONLY review text sentiment.
IDs: R0-R4

Allowed aspects (source=review_text):
- food_quality, service, speed, value, ambiance, portions, parking, outdoor

Operators: AND, OR
Levels: MUST, SHOULD, or NICE

Requirements:
1. Each request MUST have exactly 3 conditions.
2. The conditions must be connected by 2 operators (mixed allowed, e.g., AND then OR).
3. Structure format: A flat list alternating between condition objects and operator strings.

Output JSONL format (one JSON per line, NO array wrapper):
{"id": "R0", "text": "...", "structure": [Condition1, "OP1", Condition2, "OP2", Condition3]}

Example:
{"id": "R0", "text": "I need great food and fast service, or at least cheap prices.", "structure": [{"aspect": "food_quality", "level": "MUST", "source": "review_text"}, "AND", {"aspect": "speed", "level": "SHOULD", "source": "review_text"}, "OR", {"aspect": "value", "level": "MUST", "source": "review_text"}]}
{"id": "R1", "text": "Good ambiance and outdoor seating, plus tasty food.", "structure": [{"aspect": "ambiance", "level": "NICE", "source": "review_text"}, "AND", {"aspect": "outdoor", "level": "MUST", "source": "review_text"}, "AND", {"aspect": "food_quality", "level": "MUST", "source": "review_text"}]}

IMPORTANT: DO NOT use item_meta, reviewer_meta, or review_meta sources. ONLY review_text.
""",

    2: """Generate 5 restaurant requests using review text AND item metadata.
IDs: R5-R9

Allowed aspects:
- review_text: food_quality, service, speed, value, ambiance, portions
- item_meta: categories (Sandwiches, Cafes, Coffee & Tea, Bakeries, Breakfast & Brunch, American (New), Nightlife), WiFi, BusinessAcceptsCreditCards, RestaurantsReservations, outdoor_seating, BusinessParking, BusinessParking_garage, BusinessParking_lot, BusinessParking_street, RestaurantsPriceRange2, stars, review_count

Operators: AND, OR
Levels: MUST, SHOULD, or NICE

Requirements:
1. Structure format: A list alternating between condition objects and operator strings (e.g., [C1, "OP", C2]).
2. Each request MUST use at least one condition from 'review_text' AND one from 'item_meta'.

Output JSONL format (one JSON per line, NO array wrapper).

Example:
{"id": "R5", "text": "I want a cafe with good food.", "structure": [{"aspect": "Cafes", "level": "MUST", "source": "item_meta"}, "AND", {"aspect": "food_quality", "level": "SHOULD", "source": "review_text"}]}
{"id": "R6", "text": "Looking for a sandwich shop or bakery that is cheap.", "structure": [{"aspect": "Sandwiches", "level": "MUST", "source": "item_meta"}, "OR", {"aspect": "Bakeries", "level": "MUST", "source": "item_meta"}, "AND", {"aspect": "RestaurantsPriceRange2", "level": "MUST", "source": "item_meta"}]}

IMPORTANT: DO NOT use reviewer_meta or review_meta.
""",

    3: """Generate 5 requests with NESTED conditions (AND containing OR, or OR containing AND).
IDs: R10-R14

Allowed aspects:
- review_text: food_quality, service, speed, value, ambiance, portions
- item_meta: categories, WiFi, parking attributes, outdoor_seating, stars, etc.

Structure MUST be nested (lists inside lists).
Levels: MUST, SHOULD, or NICE

Output JSONL format (one JSON per line, NO array wrapper).

Example (AND containing OR):
{"id": "R10", "text": "I need parking (garage or street) and good food.", "structure": [{"aspect": "food_quality", "level": "MUST", "source": "review_text"}, "AND", [{"aspect": "BusinessParking_garage", "level": "MUST", "source": "item_meta"}, "OR", {"aspect": "BusinessParking_street", "level": "MUST", "source": "item_meta"}]]}

Example (OR containing AND):
{"id": "R11", "text": "Either a cafe with WiFi, or a restaurant with outdoor seating.", "structure": [[{"aspect": "Cafes", "level": "MUST", "source": "item_meta"}, "AND", {"aspect": "WiFi", "level": "MUST", "source": "item_meta"}], "OR", [{"aspect": "outdoor_seating", "level": "MUST", "source": "item_meta"}, "AND", {"aspect": "food_quality", "level": "SHOULD", "source": "review_text"}]]}

IMPORTANT: DO NOT use reviewer_meta or review_meta. Use sub-lists for nested groups.
""",

    4: """Generate 5 requests with nested conditions INCLUDING reviewer/review metadata.
IDs: R15-R19

Allowed aspects:
- review_text: food_quality, service, speed, value, ambiance, portions
- item_meta: categories, WiFi, parking, outdoor_seating, stars, etc.
- reviewer_meta: friends_count, common_friend (takes user_ids list), reviewer_review_count
- review_meta: stars, date, useful_votes

Note: common_friend checks if reviewer has a friend in common with a provided list of user_ids.

Structure MUST be nested AND must include at least one reviewer_meta or review_meta condition.
Levels: MUST, SHOULD, or NICE

Output JSONL format (one JSON per line, NO array wrapper).

Example with common_friend:
{"id": "R15", "text": "Show me places my friends' network trusts for good food.", "structure": [{"aspect": "food_quality", "level": "MUST", "source": "review_text"}, "AND", [{"aspect": "common_friend", "level": "SHOULD", "source": "reviewer_meta", "user_ids": ["user_abc123", "user_xyz789"]}, "OR", {"aspect": "friends_count", "level": "MUST", "source": "reviewer_meta", "min_value": 50}]]}

Example with review_meta:
{"id": "R17", "text": "Find cafes with recent positive reviews that my network trusts.", "structure": [{"aspect": "Cafes", "level": "MUST", "source": "item_meta"}, "AND", [{"aspect": "date", "level": "SHOULD", "source": "review_meta", "recency": "6_months"}, "OR", {"aspect": "common_friend", "level": "SHOULD", "source": "reviewer_meta", "user_ids": ["user_abc123"]}]]}

IMPORTANT: Each request MUST include at least one reviewer_meta or review_meta condition.
"""
}


def compute_distributions(items: list) -> dict:
    """Compute various distributions from items."""
    stats = {
        "item_stars": Counter(),
        "categories": Counter(),
        "cities": Counter(),
        "review_stars": Counter(),
        "reviewer_elite": 0,
        "reviewer_friends_avg": 0,
        "reviewer_review_count_avg": 0,
    }

    total_reviews = 0
    total_friends = 0
    total_review_count = 0
    elite_count = 0

    for item in items:
        # Item-level stats
        if item.get("stars"):
            stats["item_stars"][int(item["stars"])] += 1
        for cat in item.get("categories", []):
            stats["categories"][cat] += 1
        if item.get("city"):
            stats["cities"][item["city"]] += 1

        # Review-level stats
        for review in item.get("item_data", []):
            total_reviews += 1
            stats["review_stars"][int(review.get("stars", 3))] += 1

            user = review.get("user", {})
            if user.get("elite"):
                elite_count += 1
            total_friends += len(user.get("friends", []))
            total_review_count += user.get("review_count", 0)

    if total_reviews > 0:
        stats["reviewer_elite"] = elite_count
        stats["reviewer_friends_avg"] = total_friends / total_reviews
        stats["reviewer_review_count_avg"] = total_review_count / total_reviews

    return stats


def sample_reviews(items: list, n_per_star: int = 2) -> list:
    """Sample reviews from each star bucket."""
    buckets = defaultdict(list)
    for item in items:
        for review in item.get("item_data", []):
            star = int(review.get("stars", 3))
            user = review.get("user", {})
            buckets[star].append({
                "text": review.get("review", "")[:300],  # Truncate
                "stars": star,
                "elite": bool(user.get("elite")),
                "friends": len(user.get("friends", [])),
                "reviewer_reviews": user.get("review_count", 0),
            })

    samples = []
    for star in range(1, 6):
        bucket = buckets.get(star, [])
        if bucket:
            samples.extend(random.sample(bucket, min(n_per_star, len(bucket))))
    return samples


def detect_aspects(items: list) -> dict:
    """Detect aspect coverage in reviews."""
    aspect_counts = {aspect: 0 for aspect in ASPECT_KEYWORDS}
    total_reviews = 0

    for item in items:
        for review in item.get("item_data", []):
            total_reviews += 1
            text = review.get("review", "").lower()
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    aspect_counts[aspect] += 1

    # Convert to percentages
    if total_reviews > 0:
        return {k: round(v / total_reviews * 100, 1) for k, v in aspect_counts.items()}
    return aspect_counts


def get_sample_item(items: list) -> dict:
    """Get a representative sample item with truncated reviews."""
    if not items:
        return {}

    item = random.choice(items)
    sample = {
        "item_id": item.get("item_id", ""),
        "item_name": item.get("item_name", ""),
        "city": item.get("city", ""),
        "state": item.get("state", ""),
        "stars": item.get("stars"),
        "categories": item.get("categories", [])[:5],
        "attributes": {k: v for k, v in list(item.get("attributes", {}).items())[:5]},
        "item_data": []
    }

    # Include 3 sample reviews
    for review in item.get("item_data", [])[:3]:
        user = review.get("user", {})
        sample["item_data"].append({
            "review_id": review.get("review_id", ""),
            "review": review.get("review", "")[:200] + "...",
            "stars": review.get("stars"),
            "date": review.get("date", ""),
            "useful": review.get("useful", 0),
            "user": {
                "name": user.get("name", ""),
                "elite": user.get("elite", []),
                "friends": f"[{len(user.get('friends', []))} friends]",
                "review_count": user.get("review_count", 0),
            }
        })

    return sample


def print_data_context(selection_name: str, items: list, stats: dict, samples: list, aspects: dict, sample_item: dict):
    """Print data context to console (no file writing)."""
    total_reviews = sum(len(item.get('item_data', [])) for item in items)

    print("\n" + "="*80)
    print("DATA CONTEXT - Copy this section with each level prompt")
    print("="*80)

    print(f"""
## Dataset Context
- Selection: {selection_name}
- Items: {len(items)} restaurants
- Total Reviews: {total_reviews}
- Cities: {dict(stats['cities'].most_common(5))}
- Top Categories: {dict(stats['categories'].most_common(10))}

## Sample Item (with metadata structure)
```json
{json.dumps(sample_item, indent=2)}
```

## Sample Reviews (with reviewer metadata)""")

    for s in samples[:6]:
        elite_str = "[ELITE]" if s["elite"] else ""
        print(f'- [{s["stars"]}-star, {s["friends"]} friends, {s["reviewer_reviews"]} reviews] {elite_str}')
        print(f'  "{s["text"][:150]}..."')

    print("\n## Aspect Coverage in Reviews")
    for aspect, pct in sorted(aspects.items(), key=lambda x: -x[1]):
        print(f"- {aspect}: {pct}% of reviews mention this")

    # Print sample user_ids for common_friend examples
    user_ids = []
    for item in items[:5]:
        for review in item.get("item_data", [])[:2]:
            user = review.get("user", {})
            if user.get("user_id"):
                user_ids.append(user["user_id"])
    if user_ids:
        print(f"\n## Sample User IDs (for common_friend)")
        for uid in user_ids[:5]:
            print(f"- {uid}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate ChatGPT prompt for request creation")
    parser.add_argument("selection", help="Selection name (e.g., selection_1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit items to analyze")
    args = parser.parse_args()

    selection_name = args.selection
    n = selection_name.replace("selection_", "")

    console.print(f"\n[bold]=== Yelp Request Generator ===[/bold]")
    console.print(f"Selection: {selection_name}")

    # Load data
    console.print(f"\n[cyan]Loading data...[/cyan]")
    try:
        items, _ = load_yelp_dataset(selection_name, args.limit)
    except FileNotFoundError as e:
        # Ignore requests file error - we're generating it
        if "requests" in str(e).lower():
            # Load items directly without requests
            from data.loader import YELP_DIR, load_requests
            import json

            # Manual load to bypass requests check
            selection_path = YELP_DIR / f"{selection_name}.jsonl"
            rev_selection_path = YELP_DIR / f"rev_{selection_name}.jsonl"
            reviews_cache_path = YELP_DIR / f"reviews_cache_{n}.jsonl"
            restaurants_cache_path = YELP_DIR / f"restaurants_cache_{n}.jsonl"

            # Check cache files exist
            for p in [selection_path, rev_selection_path, reviews_cache_path, restaurants_cache_path]:
                if not p.exists():
                    console.print(f"[red]Missing: {p}[/red]")
                    sys.exit(1)

            # Load manually (simplified)
            selection = {}
            with open(selection_path) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        selection[item["item_id"]] = item

            rev_selection = {}
            with open(rev_selection_path) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        rev_selection[item["item_id"]] = item["review_ids"]

            reviews_cache = {}
            with open(reviews_cache_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        reviews_cache[r["review_id"]] = r

            restaurants_cache = {}
            with open(restaurants_cache_path) as f:
                for line in f:
                    if line.strip():
                        biz = json.loads(line)
                        restaurants_cache[biz["business_id"]] = biz

            # Build items
            sorted_ids = sorted(selection.keys(), key=lambda x: -selection[x].get("llm_percent", 0))
            if args.limit:
                sorted_ids = sorted_ids[:args.limit]

            items = []
            for biz_id in sorted_ids:
                biz = restaurants_cache.get(biz_id, {})
                sel = selection.get(biz_id, {})
                review_ids = rev_selection.get(biz_id, [])

                cats_str = biz.get("categories", "")
                categories = [c.strip() for c in cats_str.split(",") if c.strip()] if cats_str else []

                item_data = []
                for rid in review_ids:
                    r = reviews_cache.get(rid)
                    if r:
                        item_data.append({
                            "review_id": r["review_id"],
                            "review": r["text"],
                            "stars": r["stars"],
                            "date": r.get("date", ""),
                            "useful": r.get("useful", 0),
                            "user": r.get("user", {})
                        })

                items.append({
                    "item_id": biz_id,
                    "item_name": biz.get("name", ""),
                    "city": biz.get("city", ""),
                    "state": biz.get("state", ""),
                    "stars": biz.get("stars"),
                    "categories": categories,
                    "attributes": biz.get("attributes", {}),
                    "item_data": item_data
                })
        else:
            raise

    console.print(f"Loaded {len(items)} items")

    # Compute stats
    console.print(f"\n[cyan]Computing distributions...[/cyan]")
    stats = compute_distributions(items)

    console.print(f"\n[bold]=== Item Stats ===[/bold]")
    console.print(f"  Stars distribution: {dict(stats['item_stars'])}")
    console.print(f"  Top categories: {dict(stats['categories'].most_common(5))}")
    console.print(f"  Cities: {dict(stats['cities'].most_common(5))}")

    console.print(f"\n[bold]=== Reviewer Stats ===[/bold]")
    console.print(f"  Elite reviewers: {stats['reviewer_elite']}")
    console.print(f"  Avg friends: {stats['reviewer_friends_avg']:.1f}")
    console.print(f"  Avg review count: {stats['reviewer_review_count_avg']:.1f}")

    # Sample reviews
    console.print(f"\n[cyan]Sampling reviews...[/cyan]")
    samples = sample_reviews(items)

    # Detect aspects
    console.print(f"\n[cyan]Detecting aspects...[/cyan]")
    aspects = detect_aspects(items)

    console.print(f"\n[bold]=== Aspect Coverage ===[/bold]")
    for aspect, pct in sorted(aspects.items(), key=lambda x: -x[1]):
        console.print(f"  {aspect}: {pct}%")

    # Get sample item
    sample_item = get_sample_item(items)

    # Print data context (no file writing)
    print_data_context(selection_name, items, stats, samples, aspects, sample_item)

    # Print level prompts
    console.print(f"\n[bold]=== Level Prompts ===[/bold]")
    console.print("[yellow]Copy the DATA CONTEXT above + one level prompt below to ChatGPT[/yellow]")
    console.print("[yellow]Output should be saved to: data/yelp/requests_{n}.jsonl[/yellow]\n")

    for level, prompt in LEVEL_PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"LEVEL {level} PROMPT")
        print("="*80)
        print(prompt)
        print("="*80)


if __name__ == "__main__":
    main()
