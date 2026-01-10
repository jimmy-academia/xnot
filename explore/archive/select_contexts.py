#!/usr/bin/env python3
"""
explore/select_contexts.py

Selects the Core 100 Contexts for the SCALE benchmark.
Strategy: Stratified Sampling (City=Philadelphia).
Dimensions:
    1. Cuisine (Top 5 Categories)
    2. Volume (High/Low)
    3. Stars (High/Low)

Output: explore/contexts.json
"""

import json
import random
import statistics
from pathlib import Path
from collections import Counter, defaultdict

RAW_BUSINESS = Path("preprocessing/raw/yelp_academic_dataset_business.json")
OUTPUT_FILE = Path("explore/contexts.json")
TARGET_CITY = "Philadelphia"
MIN_REVIEWS = 50

TARGET_CATEGORIES = [
    "Italian",       # G2 (Date/Social), G1 (Gluten)
    "Coffee & Tea",  # G7 (Psych - Work vs Chat), G3 (Value)
    "Pizza",         # G3 (Value), G5 (Ops - Delivery)
    "Steakhouses",   # G4 (Service), G6 (Strategy - Premium)
    "Chinese"        # G1 (Allergy), G8 (Trends - Authenticity)
]

def load_candidates():
    candidates = []
    print(f"Scanning {RAW_BUSINESS} for {TARGET_CITY} Restaurants...")
    
    with open(RAW_BUSINESS, 'r') as f:
        for line in f:
            b = json.loads(line)
            if b['city'].lower() == TARGET_CITY.lower():
                categories = (b['categories'] or "").split(', ')
                # Must match at least one target category
                if any(tc in categories for tc in TARGET_CATEGORIES):
                    if b['review_count'] >= MIN_REVIEWS:
                        b['category_list'] = categories
                        candidates.append(b)
    return candidates

def stratify_by_quadrant(candidates, cuisine, n_target=20):
    # Filter for this cuisine
    subset = [b for b in candidates if cuisine in b['category_list']]
    
    if not subset:
        return []
        
    reviews = [b['review_count'] for b in subset]
    stars = [b['stars'] for b in subset]
    
    med_rev = statistics.median(reviews)
    med_stars = statistics.median(stars)
    
    quadrants = defaultdict(list)
    for b in subset:
        is_high_vol = b['review_count'] > med_rev
        is_high_star = b['stars'] > med_stars
        
        key = "HighVol_HighStars" if (is_high_vol and is_high_star) else \
              "HighVol_LowStars" if (is_high_vol and not is_high_star) else \
              "LowVol_HighStars" if (not is_high_vol and is_high_star) else \
              "LowVol_LowStars"
        quadrants[key].append(b)
    
    # Select n_target equally from quadrants (5 from each)
    n_per_q = n_target // 4
    selected = []
    
    for key, bucket in quadrants.items():
        sample = random.sample(bucket, min(len(bucket), n_per_q))
        selected.extend(sample)
    
    return selected

def main():
    candidates = load_candidates()
    if not candidates:
        print("No candidates found!")
        return

    # Stratify within each Fixed Cuisine
    core_100 = []
    seen_ids = set()
    
    random.seed(42)
    
    for cuisine in TARGET_CATEGORIES:
        print(f"\nSelecting for: {cuisine}")
        selection = stratify_by_quadrant(candidates, cuisine, n_target=20)
        
        # Avoid duplicates if a place has multiple top categories
        count = 0
        for b in selection:
            if b['business_id'] not in seen_ids:
                b['stratification_tag'] = cuisine
                core_100.append(b)
                seen_ids.add(b['business_id'])
                count += 1
        print(f"  -> Added {count} unique contexts")
    
    # Fill remaining if duplicates reduced count
    if len(core_100) < 100:
        remaining_needed = 100 - len(core_100)
        print(f"\nNeed {remaining_needed} more to reach 100...")
        pool = [b for b in candidates if b['business_id'] not in seen_ids]
        if pool:
            # Prefer target cuisinse in pool
            core_100.extend(random.sample(pool, min(len(pool), remaining_needed)))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(core_100, f, indent=2)
        
    print(f"\nSaved {len(core_100)} Stratified Contexts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
