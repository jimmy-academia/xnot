#!/usr/bin/env python3
"""Request generator for Yelp data selection.

Given a selection_n, this script:
1. Loads the dataset (items with reviews)
2. Computes distributions (stars, categories, cities, reviewer metadata)
3. Samples representative reviews
4. Generates a ChatGPT prompt for creating 20 structured requests

Usage:
    python data/scripts/yelp_requests.py selection_1
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


def generate_prompt(selection_name: str, items: list, stats: dict, samples: list, aspects: dict, sample_item: dict) -> str:
    """Generate ChatGPT prompt."""
    n = selection_name.replace("selection_", "")

    prompt = f"""You are designing 20 user requests for a restaurant recommendation benchmark.

**Task:** Analyze the sample data below and create diverse, realistic user requests that leverage the data's rich structure.

## Dataset Context
- Selection: {selection_name}
- Items: {len(items)} restaurants
- Total Reviews: {sum(len(item.get('item_data', [])) for item in items)}
- Cities: {dict(stats['cities'].most_common(5))}
- Top Categories: {dict(stats['categories'].most_common(10))}

## Sample Item (with metadata structure)
```json
{json.dumps(sample_item, indent=2)}
```

## Sample Reviews (with reviewer metadata)
"""
    for s in samples[:6]:
        elite_str = "[ELITE]" if s["elite"] else ""
        prompt += f'\n- [{s["stars"]}-star, {s["friends"]} friends, {s["reviewer_reviews"]} reviews] {elite_str}\n  "{s["text"][:150]}..."\n'

    prompt += f"""

## Aspect Coverage in Reviews
"""
    for aspect, pct in sorted(aspects.items(), key=lambda x: -x[1]):
        prompt += f"- {aspect}: {pct}% of reviews mention this\n"

    prompt += """

## Available Data Dimensions

**1. Review Text** - Extract sentiment about:
   - food quality, service, speed, value, ambiance, portions, etc.

**2. Item Metadata**:
   - categories (cuisine type)
   - stars (overall rating)
   - attributes (parking, outdoor, wifi, etc.)
   - city, hours

**3. Reviewer Metadata**:
   - elite status (expert opinion weight)
   - friends list (trust network)
   - review_count (experience level)
   - average_stars (reviewer tendency)

**4. Review Metadata**:
   - stars (per-review rating)
   - date (recency)
   - useful/funny/cool votes (community validation)

## Request Schema
```json
{
  "id": "R0",
  "text": "Natural language user request...",
  "structure": {
    "op": "AND|OR",
    "conditions": [
      {"aspect": "aspect_name", "level": "MUST|SHOULD|NICE", "source": "review_text|item_meta|reviewer_meta|review_meta"}
    ]
  }
}
```

**Levels:**
- MUST: Non-negotiable (fail = reject restaurant)
- SHOULD: Important preference (affects score)
- NICE: Bonus (never penalizes)

## Requirements
1. Generate exactly 20 requests (R0-R19)
2. **Mix aspect sources** - don't only use review text:
   - Include 3+ requests using reviewer trust (e.g., "weight elite reviewers higher")
   - Include 3+ requests using item metadata (e.g., "must have parking", "Italian cuisine")
   - Include 2+ requests using review metadata (e.g., "recent reviews only")
3. **Vary complexity**:
   - 4 simple (1 condition)
   - 6 medium (2-3 conditions with AND)
   - 6 complex (3+ conditions with AND/OR)
   - 4 nested (AND containing OR or vice versa)
4. Natural, realistic user language in "text" field
5. Cover the aspects that appear frequently in the data

Output as a JSON array.
"""
    return prompt


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

    # Generate prompt
    console.print(f"\n[bold]=== ChatGPT Prompt ===[/bold]")
    prompt = generate_prompt(selection_name, items, stats, samples, aspects, sample_item)

    # print("\n" + "="*80)
    # print(prompt)
    # print("="*80)

    # Save prompt to file
    prompt_path = YELP_DIR / f"requests_{n}_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)
    console.print(f"\n[green]Prompt saved to: {prompt_path}[/green]")
    console.print(f"[yellow]Copy the prompt above to ChatGPT, then save output to: data/yelp/requests_{n}.json[/yellow]")


if __name__ == "__main__":
    main()
