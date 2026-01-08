import json
from analyze_progressive import get_flat_attrs

count_intimate = 0
count_garage = 0
count_missing_garage = 0

with open("data/philly_cafes/restaurants.jsonl", "r") as f:
    for line in f:
        r = json.loads(line)
        attrs = get_flat_attrs(r)
        
        if "ambience_intimate" in attrs:
            count_intimate += 1
            
        # Check raw attributes for garage
        garage = r.get("attributes", {}).get("BusinessParking")
        # garage is usually a dict string
        if garage is None:
            count_missing_garage += 1
        elif "'garage': True" in str(garage):
             count_garage += 1

print(f"Ambience Intimate: {count_intimate}")
print(f"Garage True: {count_garage}")
print(f"Garage Missing: {count_missing_garage}")
