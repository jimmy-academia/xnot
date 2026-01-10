#!/usr/bin/env python3
"""
preprocessing/select_contexts.py

Selects the Core 100 Contexts for the SCALE benchmark.
Strategy: Stratified Sampling (City=Philadelphia).
Quadrants:
    1. High Vol (> Med), High Stars (> Med)
    2. High Vol (> Med), Low Stars (<= Med)
    3. Low Vol (<= Med), High Stars (> Med)
    4. Low Vol (<= Med), Low Stars (<= Med)

Output: data/scale/contexts.json
"""

import json
import random
import statistics
from pathlib import Path
from collections import defaultdict

RAW_BUSINESS = Path("preprocessing/raw/yelp_academic_dataset_business.json")
OUTPUT_FILE = Path("data/scale/contexts.json")
TARGET_CITY = "Philadelphia"
MIN_REVIEWS = 50  # Minimum to be considered a "Context"

def load_candidates():
    candidates = []
    print(f"Scanning {RAW_BUSINESS} for {TARGET_CITY} Restaurants...")
    
    with open(RAW_BUSINESS, 'r') as f:
        for line in f:
            b = json.loads(line)
            if b['city'].lower() == TARGET_CITY.lower():
                if 'Restaurants' in (b['categories'] or ""):
                    if b['review_count'] >= MIN_REVIEWS:
                        candidates.append(b)
    return candidates

def stratify_and_select(candidates, n_per_quadrant=25):
    # 1. Calculate Medians for Split
    reviews = [b['review_count'] for b in candidates]
    stars = [b['stars'] for b in candidates]
    
    median_reviews = statistics.median(reviews)
    median_stars = statistics.median(stars)
    
    print(f"Total Candidates: {len(candidates)}")
    print(f"Median Reviews: {median_reviews}")
    print(f"Median Stars: {median_stars}")
    
    # 2. Bucket
    quadrants = {
        "Q1_HighVol_HighStars": [],
        "Q2_HighVol_LowStars": [],
        "Q3_LowVol_HighStars": [],
        "Q4_LowVol_LowStars": [],
    }
    
    for b in candidates:
        is_high_vol = b['review_count'] > median_reviews
        is_high_star = b['stars'] > median_stars
        
        if is_high_vol and is_high_star:
            quadrants["Q1_HighVol_HighStars"].append(b)
        elif is_high_vol and not is_high_star:
            quadrants["Q2_HighVol_LowStars"].append(b)
        elif not is_high_vol and is_high_star:
            quadrants["Q3_LowVol_HighStars"].append(b)
        else:
            quadrants["Q4_LowVol_LowStars"].append(b)
            
    # 3. Select
    selected = []
    random.seed(42) # Reproducibility
    
    print("\nSelection Distribution:")
    for name, bucket in quadrants.items():
        # Sort by review count to pick 'representative' ones, or pure random?
        # Pure random is better to avoid bias towards extreme outliers only
        sample = random.sample(bucket, min(len(bucket), n_per_quadrant))
        selected.extend(sample)
        print(f"  {name}: {len(sample)} / {len(bucket)} candidates")
        
    return selected

def main():
    candidates = load_candidates()
    if not candidates:
        print("No candidates found! Check path or city name.")
        return
        
    core_100 = stratify_and_select(candidates, 25)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        # Saving as list of dicts
        json.dump(core_100, f, indent=2)
        
    print(f"\nSaved {len(core_100)} contexts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
