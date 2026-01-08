
import json
import sys
from pathlib import Path

# Add parent dir to path to find data.validate
sys.path.append(".")

# 1. Raw Load
print("--- Raw Load ---")
tria_raw = None
with open("data/philly_cafes/restaurants.jsonl", "r") as f:
    for line in f:
        r = json.loads(line)
        if r["business_id"] == "eJaeTZlIdM3HWCq__Ve4Wg":
            tria_raw = r
            break

if tria_raw:
    print(f"Categories: {tria_raw.get('categories')}")
    print(f"GoodForDancing: {tria_raw.get('attributes', {}).get('GoodForDancing')}")
else:
    print("Tria not found in raw file")

# 2. Loader Load
print("\n--- Loader Load ---")
try:
    from data.loader import load_dataset
    ds = load_dataset("philly_cafes")
    items = ds.items
    if isinstance(items, dict):
        items = list(items.values())
        
    targets = ["Sabrina's Café", "Cafe La Maude"]
    for t_name in targets:
        l = next((r for r in items if r.get("name") == t_name), None)
        if l:
            print(f"--- {t_name} ---")
            print(f"Parking: {l.get('attributes', {}).get('BusinessParking')}")
            print(f"Categories: {l.get('categories')}")
            print(f"GoodForKids: {l.get('attributes', {}).get('GoodForKids')}")
        else:
            print(f"{t_name} not found")

except Exception as e:
    print(f"Loader failed: {e}")
