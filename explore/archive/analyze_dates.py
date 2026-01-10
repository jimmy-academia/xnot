#!/usr/bin/env python3
"""
explore/analyze_dates.py

Analyzes the review timelines of the Core 100 Contexts.
Goal: Determine the 'Scale (N)' needed to cross the 2020 threshold (Pre/Post Covid).

Checks:
1. If we sort Newest-First, what N is required to reach Jan 1, 2020?
2. If we sort Oldest-First, what N is required to reach Jan 1, 2020?
3. Review density per year.
"""

import json
from pathlib import Path
from datetime import datetime
from statistics import median, mean

CONTEXTS_FILE = Path("explore/contexts.json")
REVIEW_FILE = Path("preprocessing/raw/yelp_academic_dataset_review.json")
TARGET_DATE = datetime(2020, 1, 1)

def load_core_ids():
    with open(CONTEXTS_FILE) as f:
        data = json.load(f)
    return {b['business_id']: b for b in data}

def analyze():
    core_map = load_core_ids()
    core_ids = set(core_map.keys())
    
    # Store dates by business
    reviews_by_biz = {bid: [] for bid in core_ids}
    
    print(f"Loading reviews for {len(core_ids)} contexts...")
    count = 0
    with open(REVIEW_FILE) as f:
        for line in f:
            if count % 1_000_000 == 0:
                print(f"  Scanned {count} lines...", end='\r')
            count += 1
            
            # Fast check
            if '"business_id":' in line:
                # We have to parse to check ID match accurately
                try:
                    r = json.loads(line)
                    if r['business_id'] in core_ids:
                        dt = datetime.strptime(r['date'], "%Y-%m-%d %H:%M:%S")
                        reviews_by_biz[r['business_id']].append(dt)
                except:
                    continue

    print("\n\nAnalysis Results (Threshold: 2020-01-01):")
    
    n_for_2020_desc = [] # Sort Newest -> Oldest
    n_for_2020_asc = []  # Sort Oldest -> Newest
    
    for bid, dates in reviews_by_biz.items():
        if not dates:
            continue
            
        # Sort DESC (Newest First)
        dates_desc = sorted(dates, reverse=True)
        found_desc = False
        for i, d in enumerate(dates_desc):
            if d < TARGET_DATE:
                n_for_2020_desc.append(i + 1)
                found_desc = True
                break
        if not found_desc:
            # Never hit pre-2020 (Unlikely for established places, but possible)
            n_for_2020_desc.append(len(dates))

        # Sort ASC (Oldest First)
        dates_asc = sorted(dates)
        found_asc = False
        for i, d in enumerate(dates_asc):
            if d >= TARGET_DATE:
                n_for_2020_asc.append(i + 1)
                found_asc = True
                break
        if not found_asc:
            n_for_2020_asc.append(len(dates))

    print("-" * 50)
    print(f"Strategy: Make Context from NEWEST reviews (Desc)")
    print(f"  Avg N to hit 2019: {mean(n_for_2020_desc):.1f}")
    print(f"  Med N to hit 2019: {median(n_for_2020_desc)}")
    print(f"  Max N to hit 2019: {max(n_for_2020_desc)}")
    
    print("-" * 50)
    print(f"Strategy: Make Context from OLDEST reviews (Asc)")
    print(f"  Avg N to hit 2020: {mean(n_for_2020_asc):.1f}")
    print(f"  Med N to hit 2020: {median(n_for_2020_asc)}")
    
    print("-" * 50)
    
if __name__ == "__main__":
    analyze()
