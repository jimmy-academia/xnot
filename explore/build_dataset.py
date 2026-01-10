#!/usr/bin/env python3
"""
explore/build_dataset.py

Constructs the final SCALE datasets.
Scales: N=25, 50, 100, 200.
Structure: One JSON object per Context.
    {
        "context_id": "Italian_HighVol_HighStars_1",
        "business": { ... item_meta ... },
        "reviews": [
            {
                "text": "...",
                "date": "...",
                "user": { ... reviewer_meta ... }
            },
            ...
        ]
    }
"""

import json
from pathlib import Path

CONTEXTS_FILE = Path("explore/contexts.json")
REVIEW_FILE = Path("preprocessing/raw/yelp_academic_dataset_review.json")
USER_FILE = Path("preprocessing/raw/yelp_academic_dataset_user.json")
OUTPUT_DIR = Path("explore/data")

SCALES = [25, 50, 100, 200]

def load_data():
    # 1. Load Business Contexts
    with open(CONTEXTS_FILE) as f:
        contexts = json.load(f)
    core_ids = {b['business_id'] for b in contexts}
    print(f"Loaded {len(contexts)} contexts.")
    
    # 2. Load Reviews (filtered)
    print("Scanning reviews...")
    reviews_by_biz = {bid: [] for bid in core_ids}
    user_ids_needed = set()
    
    with open(REVIEW_FILE) as f:
        for line in f:
            if '"business_id":' in line: # fast filter
                try:
                    r = json.loads(line)
                    if r['business_id'] in core_ids:
                        reviews_by_biz[r['business_id']].append(r)
                        user_ids_needed.add(r['user_id'])
                except: continue
                
    # Sort Reviews by Date DESC (Newest First)
    for bid in reviews_by_biz:
        reviews_by_biz[bid].sort(key=lambda x: x['date'], reverse=True)
        
    print(f"Found {sum(len(v) for v in reviews_by_biz.values())} reviews.")
    print(f"Need {len(user_ids_needed)} users.")
    
    # 3. Load Users (filtered)
    print("Scanning users...")
    users_map = {}
    with open(USER_FILE) as f:
        for line in f:
            if '"user_id":' in line:
                try:
                    u = json.loads(line)
                    if u['user_id'] in user_ids_needed:
                        # Keep only relevant meta
                        users_map[u['user_id']] = {
                            "name": u.get("name"),
                            "review_count": u.get("review_count"),
                            "yelping_since": u.get("yelping_since"),
                            "elite": u.get("elite"),
                            "average_stars": u.get("average_stars"),
                            "fans": u.get("fans")
                        }
                except: continue
                
    return contexts, reviews_by_biz, users_map

def build_datasets(contexts, reviews_by_biz, users_map):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for n in SCALES:
        out_path = OUTPUT_DIR / f"dataset_K{n}.jsonl"
        print(f"Building K={n} (reviews per restaurant) -> {out_path}")
        
        with open(out_path, 'w') as f:
            for b in contexts:
                bid = b['business_id']
                # Take top N
                top_reviews = reviews_by_biz.get(bid, [])[:n]
                
                # Enrich with User Meta
                enriched_reviews = []
                for r in top_reviews:
                    r_enriched = r.copy()
                    uid = r['user_id']
                    if uid in users_map:
                        r_enriched['user'] = users_map[uid]
                    enriched_reviews.append(r_enriched)
                
                record = {
                    "business_id": bid,
                    "stratification": b.get('stratification_tag'),
                    "business": b,
                    "reviews": enriched_reviews,
                    "review_count_actual": len(enriched_reviews)
                }
                
                f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    c, r, u = load_data()
    build_datasets(c, r, u)
