import json

# Load Data
restaurants = []
with open("data/philly_cafes/restaurants.jsonl", "r") as f:
    for line in f:
        restaurants.append(json.loads(line))

requests = []
with open("data/philly_cafes/g04_new_requests.jsonl", "r") as f:
    for line in f:
        requests.append(json.loads(line))

# Helper to check conditions
def check_condition(rest, condition):
    val = rest
    path = condition["evidence"]["path"]
    
    for p in path:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return False

    if "contains" in condition["evidence"]:
        return condition["evidence"]["contains"] in str(val)
    if "true" in condition["evidence"]:
        return str(val) == condition["evidence"]["true"]
    return False

# Verify R38, R39, R40
target_ids = ["R38", "R39", "R40"]
results = {}

for req in requests:
    if req["id"] not in target_ids:
        continue
    
    matches = []
    print(f"Verifying {req['id']} ({req['gold_restaurant']})...")
    
    conditions = req["structure"]["args"]
    
    for r in restaurants:
        match = True
        for cond in conditions:
            if not check_condition(r, cond):
                match = False
                break
        if match:
            matches.append(r["business_id"])
            
    results[req["id"]] = matches
    print(f"  Matches: {len(matches)} -> {matches}")
    
    if len(matches) == 1 and matches[0] == req["gold_restaurant"]:
        print("  [PASS] Unique match found.")
    else:
        print("  [FAIL] Match count or ID mismatch.")

