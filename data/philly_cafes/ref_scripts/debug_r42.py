import json
from data.validate import validate_request

# Load Tria
tria = None
with open("data/philly_cafes/restaurants.jsonl", "r") as f:
    for line in f:
        r = json.loads(line)
        if r["business_id"] == "eJaeTZlIdM3HWCq__Ve4Wg":
            tria = r
            break

print("Loaded Tria:", tria["name"])

# Load Request R42
# (Copying from what I expect or reading file)
req_str = """
{
  "id": "R42",
  "group": "G05",
  "scenario": "Progressive Search",
  "text": "Looking for a place that has classy ambience and either has no HappyHour, price range 3, or price range 4",
  "shorthand": "AND(ambience_classy, OR(no_HappyHour, price_3, price_4))",
  "structure": {
    "op": "AND",
    "args": [
      {
        "aspect": "ambience_classy",
        "evidence": {
          "kind": "item_meta",
          "path": ["attributes", "Ambience"],
          "contains": "'classy': True"
        }
      },
      {
        "op": "OR",
        "args": [
          {
            "aspect": "no_HappyHour",
            "evidence": {
              "kind": "item_meta",
              "path": ["attributes", "HappyHour"],
              "true": "False"
            }
          },
          {
            "aspect": "price_3",
            "evidence": {
              "kind": "item_meta",
              "path": ["attributes", "RestaurantsPriceRange2"],
              "true": "3"
            }
          },
          {
            "aspect": "price_4",
            "evidence": {
              "kind": "item_meta",
              "path": ["attributes", "RestaurantsPriceRange2"],
              "true": "4"
            }
          }
        ]
      }
    ]
  },
  "gold_restaurant": "eJaeTZlIdM3HWCq__Ve4Wg"
}
"""
# Note: I need the ACTUAL R42 from the file to be sure
# I will overwrite this variable with file content
with open("data/philly_cafes/requests.jsonl", "r") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] == "R42":
            req = r
            break

print("Loaded Request R42:", req["text"])

# Validate
# validate_request(request, restaurants, reviews_by_id)
# We can pass [tria] as restaurants list
# Need reviews? Not for G05 (item_meta only)
res = validate_request(req, [tria], {})
print("Result:", res)

# Debug deeply if failed
from data.validate import evaluate_structure
print("Detailed Check:")
print("Structure result:", evaluate_structure(tria, req["structure"]))

# Check individual args
print("Checking Args:")
for arg in req["structure"]["args"]:
    from data.validate import evaluate_condition, evaluate_structure
    if "op" in arg:
        print(f"Arg OP {arg['op']} => {evaluate_structure(tria, arg)}")
        if arg["op"] == "OR":
            for sub in arg["args"]:
                print(f"  Sub {sub.get('aspect')} => {evaluate_condition(tria, sub)}")
    else:
        print(f"Arg {arg['aspect']} => {evaluate_condition(tria, arg)}")
