
import json
import re
from pathlib import Path
from collections import defaultdict

# Paths
SOURCE_RES = Path('data/philly_cafes/restaurants.jsonl')
SOURCE_REV = Path('data/philly_cafes/reviews.jsonl')
OUTPUT_DIR = Path('explore/data')

def sanitize_filename(name):
    # Replace non-alphanumeric with _
    return re.sub(r'[^a-zA-Z0-9]', '_', name)

def main():
    print(f"Reading restaurants from {SOURCE_RES}...")
    restaurants = {}
    with open(SOURCE_RES, 'r') as f:
        for line in f:
            r = json.loads(line)
            restaurants[r['business_id']] = r

    print(f"Reading reviews from {SOURCE_REV}...")
    reviews_by_biz = defaultdict(list)
    with open(SOURCE_REV, 'r') as f:
        for line in f:
            r = json.loads(line)
            reviews_by_biz[r['business_id']].append(r)

    # Sort restaurants by review count (descending) to get the most data-rich ones
    # We use the length of the reviews list we just built, or the 'review_count' field
    # Let's use the actual reviews we have.
    
    sorted_biz_ids = sorted(reviews_by_biz.keys(), key=lambda bid: len(reviews_by_biz[bid]), reverse=True)
    
    # Take top 100
    top_100 = sorted_biz_ids[:100]
    
    print(f"Generating files for top {len(top_100)} restaurants...")
    
    count = 0
    for i, bid in enumerate(top_100, 1):
        if bid not in restaurants:
            continue
            
        r = restaurants[bid]
        revs = reviews_by_biz[bid]
        
        # Construct filename: Index_Name_BusinessID.jsonl
        # Truncate ID to 7 chars to match existing style (optional, but cleaner)
        safe_name = sanitize_filename(r['name'])
        filename = f"{i:03d}_{safe_name}_{bid}.jsonl"
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w') as f:
            # Write restaurant metadata
            f.write(json.dumps(r) + '\n')
            # Write reviews
            for rev in revs:
                f.write(json.dumps(rev) + '\n')
        
        count += 1
        
    print(f"Successfully created {count} restaurant files in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
