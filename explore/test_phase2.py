#!/usr/bin/env python3
"""
Test Phase 2: Compare general ANoT vs hardcoded G1a-ANoT

Runs both implementations on the same restaurants and compares results.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import configure

# Import both implementations
from general_anot.phase1_step3 import FormulaSeed
from general_anot.phase2 import Phase2Executor
from g1_anot.core import G1aANoT


async def load_restaurants(limit: int = 5):
    """Load sample restaurants from philly_cafes dataset."""
    data_dir = Path(__file__).parent.parent / "data" / "philly_cafes"
    restaurants_file = data_dir / "restaurants.jsonl"
    reviews_file = data_dir / "reviews.jsonl"

    # Read all restaurants
    restaurants = []
    with open(restaurants_file) as f:
        for line in f:
            restaurants.append(json.loads(line))

    # Read all reviews and group by business_id
    reviews_by_business = {}
    with open(reviews_file) as f:
        for line in f:
            review = json.loads(line)
            bid = review.get('business_id')
            if bid not in reviews_by_business:
                reviews_by_business[bid] = []
            reviews_by_business[bid].append(review)

    # Build restaurant objects with reviews
    result = []
    for business in restaurants[:limit]:
        bid = business.get('business_id')
        reviews = reviews_by_business.get(bid, [])
        result.append({
            'business': business,
            'reviews': reviews,
        })

    return result


async def test_comparison():
    """Compare general ANoT vs hardcoded G1a-ANoT."""
    configure(temperature=0.0)

    # Load Formula Seed
    seed_file = Path(__file__).parent / "results" / "phase1_steps" / "step1_3_formula_seed_v2.json"
    with open(seed_file) as f:
        seed = FormulaSeed.from_dict(json.load(f))

    # Load restaurants
    restaurants = await load_restaurants(limit=3)

    print("="*70)
    print("COMPARING GENERAL ANOT vs HARDCODED G1A-ANOT")
    print("="*70)

    # Initialize both
    general_anot = Phase2Executor(seed, verbose=False)
    hardcoded_anot = G1aANoT(verbose=False)

    for restaurant in restaurants:
        name = restaurant['business'].get('name', 'Unknown')
        print(f"\n{'='*70}")
        print(f"Restaurant: {name}")
        print(f"  Reviews: {len(restaurant['reviews'])}")
        print(f"{'='*70}")

        # Run both
        general_result = await general_anot.execute(restaurant)
        hardcoded_result = await hardcoded_anot.evaluate(restaurant)

        print("\nGeneral ANoT:")
        for k, v in sorted(general_result.items()):
            print(f"  {k}: {v}")

        print("\nHardcoded G1a-ANoT:")
        for k, v in sorted(hardcoded_result.items()):
            print(f"  {k}: {v}")

        # Compare key fields
        print("\nComparison:")
        for field in ['FINAL_RISK_SCORE', 'VERDICT']:
            gen_val = general_result.get(field, 'N/A')
            hard_val = hardcoded_result.get(field, 'N/A')
            match = "✓" if gen_val == hard_val else "✗"
            print(f"  {field}: {match} (General: {gen_val}, Hardcoded: {hard_val})")


if __name__ == "__main__":
    asyncio.run(test_comparison())
