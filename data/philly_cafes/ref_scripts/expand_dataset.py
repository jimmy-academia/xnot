#!/usr/bin/env python3
"""Expand philly_cafes from 20 to 50 restaurants.

Usage:
    python data/scripts/expand_dataset.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREPROC_DIR = PROJECT_ROOT / "preprocessing" / "output" / "philly_cafes"
DATA_DIR = PROJECT_ROOT / "data" / "philly_cafes"

TARGET_RESTAURANTS = 50
REVIEWS_PER_RESTAURANT = 20


def load_jsonl(path):
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def select_reviews(reviews, n=20):
    """Select n reviews with diverse star ratings."""
    if len(reviews) <= n:
        return reviews

    # Group by stars
    by_stars = defaultdict(list)
    for r in reviews:
        by_stars[r.get("stars", 3)].append(r)

    selected = []
    per_star = n // 5  # 4 per star rating

    # Take from each star rating
    for star in range(1, 6):
        pool = by_stars.get(star, [])
        if pool:
            take = min(per_star, len(pool))
            selected.extend(random.sample(pool, take))

    # Fill remaining from all reviews
    remaining = n - len(selected)
    if remaining > 0:
        used_ids = {r["review_id"] for r in selected}
        unused = [r for r in reviews if r["review_id"] not in used_ids]
        if unused:
            selected.extend(random.sample(unused, min(remaining, len(unused))))

    return selected[:n]


def count_attribute(restaurants, key, val):
    """Count restaurants with this attribute value."""
    count = 0
    for r in restaurants:
        attrs = r.get("attributes", {})
        if str(attrs.get(key, "")) == str(val):
            count += 1
    return count


def get_rarity_score(restaurant, all_restaurants):
    """Higher score = more unique/rare attributes."""
    attrs = restaurant.get("attributes", {})
    score = 0

    for key, val in attrs.items():
        if val is None or val == "" or val == "None":
            continue
        match_count = count_attribute(all_restaurants, key, val)
        # Rarer = higher score
        if match_count <= 3:
            score += 10
        elif match_count <= 5:
            score += 5
        elif match_count <= 10:
            score += 2

    return score


def main():
    print("=== Expanding philly_cafes dataset ===\n")

    # Load source data
    print("Loading source data from preprocessing/output...")
    all_restaurants = load_jsonl(PREPROC_DIR / "restaurants.jsonl")
    all_reviews = load_jsonl(PREPROC_DIR / "reviews.jsonl")
    print(f"  Source: {len(all_restaurants)} restaurants, {len(all_reviews)} reviews")

    # Group reviews by business_id
    reviews_by_biz = defaultdict(list)
    for r in all_reviews:
        reviews_by_biz[r["business_id"]].append(r)

    # Load current data
    print("\nLoading current dataset from data/philly_cafes...")
    current_restaurants = load_jsonl(DATA_DIR / "restaurants.jsonl")
    current_ids = {r["business_id"] for r in current_restaurants}
    print(f"  Current: {len(current_restaurants)} restaurants")

    # Load groundtruth to preserve gold restaurants
    groundtruth = load_jsonl(DATA_DIR / "groundtruth.jsonl")
    gold_ids = {gt["gold_restaurant"] for gt in groundtruth}
    print(f"  Gold restaurants in requests: {len(gold_ids)}")

    # Build restaurant lookup
    all_by_id = {r["business_id"]: r for r in all_restaurants}

    # Start with existing restaurants (preserves requests)
    selected = list(current_restaurants)
    selected_ids = set(current_ids)

    # Find candidates for expansion
    candidates = []
    for r in all_restaurants:
        bid = r["business_id"]
        if bid in selected_ids:
            continue

        # Must have enough reviews
        review_count = len(reviews_by_biz.get(bid, []))
        if review_count < REVIEWS_PER_RESTAURANT:
            continue

        # Prefer high LLM score
        llm_score = r.get("llm_score", 0)
        if llm_score < 80:
            continue

        # Calculate rarity score
        rarity = get_rarity_score(r, all_restaurants)

        candidates.append({
            "restaurant": r,
            "review_count": review_count,
            "llm_score": llm_score,
            "rarity_score": rarity,
        })

    print(f"\nCandidates for expansion: {len(candidates)}")

    # Sort by rarity + review quality
    candidates.sort(key=lambda x: (-x["rarity_score"], -x["llm_score"]))

    # Add top candidates
    need = TARGET_RESTAURANTS - len(selected)
    print(f"Need {need} more restaurants")

    for c in candidates[:need]:
        selected.append(c["restaurant"])
        selected_ids.add(c["restaurant"]["business_id"])
        print(f"  + {c['restaurant']['name'][:30]} (rarity={c['rarity_score']}, reviews={c['review_count']})")

    print(f"\nTotal selected: {len(selected)} restaurants")

    # Build output data
    output_restaurants = []
    output_reviews = []

    for r in selected:
        bid = r["business_id"]
        output_restaurants.append(r)

        # Select reviews for this restaurant
        biz_reviews = reviews_by_biz.get(bid, [])
        selected_reviews = select_reviews(biz_reviews, REVIEWS_PER_RESTAURANT)
        output_reviews.extend(selected_reviews)

    print(f"Output: {len(output_restaurants)} restaurants, {len(output_reviews)} reviews")

    # Create index for groundtruth update
    restaurant_idx = {r["business_id"]: i for i, r in enumerate(output_restaurants)}

    # Update groundtruth with new indices
    updated_groundtruth = []
    for gt in groundtruth:
        gold_id = gt["gold_restaurant"]
        new_idx = restaurant_idx.get(gold_id)
        if new_idx is None:
            print(f"  WARNING: gold_restaurant {gold_id} not in output!")
            new_idx = gt["gold_idx"]  # keep old
        updated_groundtruth.append({
            "request_id": gt["request_id"],
            "gold_restaurant": gold_id,
            "gold_idx": new_idx,
        })

    # Write output files
    print("\nWriting output files...")

    # restaurants.jsonl
    with open(DATA_DIR / "restaurants.jsonl", "w") as f:
        for r in output_restaurants:
            f.write(json.dumps(r) + "\n")
    print(f"  restaurants.jsonl: {len(output_restaurants)} records")

    # reviews.jsonl
    with open(DATA_DIR / "reviews.jsonl", "w") as f:
        for r in output_reviews:
            f.write(json.dumps(r) + "\n")
    print(f"  reviews.jsonl: {len(output_reviews)} records")

    # groundtruth.jsonl
    with open(DATA_DIR / "groundtruth.jsonl", "w") as f:
        for gt in updated_groundtruth:
            f.write(json.dumps(gt) + "\n")
    print(f"  groundtruth.jsonl: {len(updated_groundtruth)} records")

    # Update meta.json
    meta_path = DATA_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["restaurants"]["count"] = len(output_restaurants)
    meta["reviews"]["count"] = len(output_reviews)
    meta["reviews"]["per_restaurant"] = REVIEWS_PER_RESTAURANT
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json: updated")

    print("\n=== Done! ===")
    print(f"Dataset expanded from 20 to {len(output_restaurants)} restaurants")


if __name__ == "__main__":
    main()
