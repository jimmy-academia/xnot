#!/usr/bin/env python3
"""Add more conditions to requests to make them harder."""

import json
import random
from pathlib import Path

# Candidate attributes to add (with human-readable aspect names)
# Format: (aspect_name, path, expected_value, [match_type])
CANDIDATE_ATTRS = [
    ("wifi_free", ["attributes", "WiFi"], "u'free'"),
    ("wifi_none", ["attributes", "WiFi"], "u'no'"),
    ("good_for_kids", ["attributes", "GoodForKids"], "True"),
    ("not_for_kids", ["attributes", "GoodForKids"], "False"),
    ("bike_parking", ["attributes", "BikeParking"], "True"),
    ("delivery", ["attributes", "RestaurantsDelivery"], "True"),
    ("takeout", ["attributes", "RestaurantsTakeOut"], "True"),
    ("has_tv", ["attributes", "HasTV"], "True"),
    ("no_tv", ["attributes", "HasTV"], "False"),
    ("caters", ["attributes", "Caters"], "True"),
    ("wheelchair", ["attributes", "WheelchairAccessible"], "True"),
    ("reservations", ["attributes", "RestaurantsReservations"], "True"),
    ("no_reservations", ["attributes", "RestaurantsReservations"], "False"),
    ("good_for_groups", ["attributes", "RestaurantsGoodForGroups"], "True"),
    ("credit_cards", ["attributes", "BusinessAcceptsCreditCards"], "True"),
    ("price_cheap", ["attributes", "RestaurantsPriceRange2"], "1"),
    ("price_mid", ["attributes", "RestaurantsPriceRange2"], "2"),
    ("casual_attire", ["attributes", "RestaurantsAttire"], "u'casual'"),
    ("hipster_vibe", ["attributes", "Ambience"], "'hipster': True", "contains"),
    ("classy_vibe", ["attributes", "Ambience"], "'classy': True", "contains"),
    ("casual_vibe", ["attributes", "Ambience"], "'casual': True", "contains"),
    ("brunch", ["attributes", "GoodForMeal"], "'brunch': True", "contains"),
    ("breakfast", ["attributes", "GoodForMeal"], "'breakfast': True", "contains"),
    ("lunch", ["attributes", "GoodForMeal"], "'lunch': True", "contains"),
]


def get_restaurant_attrs(restaurant):
    """Extract all attribute values from a restaurant."""
    return restaurant.get("attributes", {})


def find_matching_candidates(restaurant, existing_aspects):
    """Find candidates that match the restaurant and aren't already used."""
    attrs = get_restaurant_attrs(restaurant)
    matches = []

    for candidate in CANDIDATE_ATTRS:
        aspect_name = candidate[0]
        path = candidate[1]
        expected = candidate[2]
        match_type = candidate[3] if len(candidate) > 3 else "exact"

        if aspect_name in existing_aspects:
            continue

        attr_key = path[-1]
        attr_val = attrs.get(attr_key, "")

        if match_type == "contains":
            if expected in str(attr_val):
                matches.append(candidate)
        else:  # exact
            if str(attr_val) == expected:
                matches.append(candidate)

    return matches


def build_condition(candidate):
    """Build a condition dict from a candidate."""
    aspect_name = candidate[0]
    path = candidate[1]
    expected = candidate[2]
    match_type = candidate[3] if len(candidate) > 3 else "exact"

    evidence = {"kind": "item_meta", "path": path}
    if match_type == "contains":
        evidence["contains"] = expected
    else:
        evidence["true"] = expected

    return {"aspect": aspect_name, "evidence": evidence}


def collect_aspects(args):
    """Recursively collect all aspect names from args (handles nested OR/AND)."""
    aspects = set()
    for arg in args:
        if "aspect" in arg:
            aspects.add(arg["aspect"])
        elif "op" in arg and "args" in arg:
            aspects.update(collect_aspects(arg["args"]))
    return aspects


def process_request(request, restaurants_by_id):
    """Add 2-4 more conditions to a request."""
    gold_id = request["gold_restaurant"]
    restaurant = restaurants_by_id.get(gold_id)
    if not restaurant:
        return request

    # Get existing aspects (recursively)
    existing_aspects = collect_aspects(request["structure"]["args"])

    # Find matching candidates
    matches = find_matching_candidates(restaurant, existing_aspects)

    # Determine how many to add (aim for 3-5 total)
    current_count = len(request["structure"]["args"])
    target_total = random.randint(3, 5)
    to_add = max(0, min(len(matches), target_total - current_count))

    if to_add > 0:
        selected = random.sample(matches, to_add)
        for candidate in selected:
            request["structure"]["args"].append(build_condition(candidate))

    return request


def main():
    data_dir = Path(__file__).parent.parent / "philly_cafes"

    # Load restaurants
    restaurants = {}
    with open(data_dir / "restaurants.jsonl") as f:
        for line in f:
            r = json.loads(line)
            restaurants[r["business_id"]] = r

    # Process requests
    requests = []
    with open(data_dir / "requests.jsonl") as f:
        for line in f:
            req = json.loads(line)
            req = process_request(req, restaurants)
            requests.append(req)

    # Write back
    with open(data_dir / "requests.jsonl", "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Print summary
    total_conditions = 0
    for req in requests:
        n = len(req["structure"]["args"])
        total_conditions += n
        print(f"{req['id']}: {n} conditions")

    print(f"\nTotal: {len(requests)} requests, avg {total_conditions/len(requests):.1f} conditions each")


if __name__ == "__main__":
    main()
