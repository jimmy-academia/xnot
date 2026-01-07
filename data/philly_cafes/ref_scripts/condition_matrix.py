#!/usr/bin/env python3
"""
Condition Matrix Generator for Philly Cafes Dataset

This script builds a comprehensive condition satisfaction matrix showing
which restaurants satisfy which conditions. This enables systematic
design of requests with guaranteed unique matches.

Evidence Types Supported:
1. item_meta - Restaurant attributes (true, contains, not_true)
2. item_meta_hours - Operating hours conditions
3. review_text - Review text pattern matching
4. review_meta - Weighted review conditions (G04 style)

Usage:
    python data/scripts/condition_matrix.py [--output json|csv|summary]

Output:
    - data/philly_cafes/condition_matrix.json (full matrix)
    - data/philly_cafes/condition_summary.md (human-readable summary)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

# Paths
DATA_DIR = Path(__file__).parent.parent / "philly_cafes"
RESTAURANTS_FILE = DATA_DIR / "restaurants.jsonl"
REVIEWS_FILE = DATA_DIR / "reviews.jsonl"
OUTPUT_MATRIX = DATA_DIR / "condition_matrix.json"
OUTPUT_SUMMARY = DATA_DIR / "condition_summary.md"


def load_data() -> Tuple[List[dict], List[dict]]:
    """Load restaurants and reviews."""
    restaurants = []
    with open(RESTAURANTS_FILE) as f:
        for line in f:
            restaurants.append(json.loads(line))

    reviews = []
    with open(REVIEWS_FILE) as f:
        for line in f:
            reviews.append(json.loads(line))

    return restaurants, reviews


def build_review_index(reviews: List[dict]) -> Dict[str, List[dict]]:
    """Index reviews by business_id."""
    index = defaultdict(list)
    for r in reviews:
        index[r['business_id']].append(r)
    return index


# =============================================================================
# CONDITION DEFINITIONS
# =============================================================================

def define_item_meta_conditions() -> List[dict]:
    """
    Define all item_meta conditions to check.

    Each condition has:
    - name: Human-readable name
    - path: JSON path to attribute
    - check_type: 'true' (exact), 'contains' (substring), 'not_true' (negation)
    - value: Value to match
    - natural_phrase: How to express this naturally in a request
    - negative_justification: For negative conditions, why someone might want this
    """
    conditions = []

    # === Price Range ===
    conditions.extend([
        {"name": "price_budget", "path": ["attributes", "RestaurantsPriceRange2"],
         "check_type": "true", "value": "1",
         "natural_phrase": "budget-friendly", "category": "price"},
        {"name": "price_mid", "path": ["attributes", "RestaurantsPriceRange2"],
         "check_type": "true", "value": "2",
         "natural_phrase": "mid-priced", "category": "price"},
        {"name": "price_upscale", "path": ["attributes", "RestaurantsPriceRange2"],
         "check_type": "true", "value": "3",
         "natural_phrase": "upscale", "category": "price"},
    ])

    # === Noise Level ===
    conditions.extend([
        {"name": "noise_quiet", "path": ["attributes", "NoiseLevel"],
         "check_type": "contains", "value": "quiet",
         "natural_phrase": "quiet", "category": "ambience"},
        {"name": "noise_average", "path": ["attributes", "NoiseLevel"],
         "check_type": "contains", "value": "average",
         "natural_phrase": "with moderate noise", "category": "ambience"},
        {"name": "noise_loud", "path": ["attributes", "NoiseLevel"],
         "check_type": "contains", "value": "loud",
         "natural_phrase": "lively", "category": "ambience"},
    ])

    # === WiFi ===
    conditions.extend([
        {"name": "wifi_free", "path": ["attributes", "WiFi"],
         "check_type": "contains", "value": "free",
         "natural_phrase": "with free WiFi", "category": "amenities"},
        {"name": "wifi_paid", "path": ["attributes", "WiFi"],
         "check_type": "contains", "value": "paid",
         "natural_phrase": "with paid WiFi", "category": "amenities"},
        {"name": "wifi_none", "path": ["attributes", "WiFi"],
         "check_type": "contains", "value": "no",
         "natural_phrase": "without WiFi",
         "negative_justification": "digital detox, focused conversation", "category": "amenities"},
    ])

    # === Alcohol ===
    conditions.extend([
        {"name": "alcohol_full_bar", "path": ["attributes", "Alcohol"],
         "check_type": "contains", "value": "full_bar",
         "natural_phrase": "with a full bar", "category": "drinks"},
        {"name": "alcohol_beer_wine", "path": ["attributes", "Alcohol"],
         "check_type": "contains", "value": "beer_and_wine",
         "natural_phrase": "with beer and wine", "category": "drinks"},
        {"name": "alcohol_none", "path": ["attributes", "Alcohol"],
         "check_type": "contains", "value": "none",
         "natural_phrase": "without alcohol",
         "negative_justification": "family-friendly, sober meetup", "category": "drinks"},
    ])

    # === Kids ===
    conditions.extend([
        {"name": "kids_yes", "path": ["attributes", "GoodForKids"],
         "check_type": "true", "value": "True",
         "natural_phrase": "kid-friendly", "category": "audience"},
        {"name": "kids_no", "path": ["attributes", "GoodForKids"],
         "check_type": "true", "value": "False",
         "natural_phrase": "adult-oriented",
         "negative_justification": "quiet work environment, date night", "category": "audience"},
    ])

    # === Dogs ===
    conditions.extend([
        {"name": "dogs_yes", "path": ["attributes", "DogsAllowed"],
         "check_type": "true", "value": "True",
         "natural_phrase": "dog-friendly", "category": "pets"},
        {"name": "dogs_no", "path": ["attributes", "DogsAllowed"],
         "check_type": "true", "value": "False",
         "natural_phrase": "without dogs",
         "negative_justification": "allergies, quiet atmosphere", "category": "pets"},
    ])

    # === Outdoor Seating ===
    conditions.extend([
        {"name": "outdoor_yes", "path": ["attributes", "OutdoorSeating"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with outdoor seating", "category": "seating"},
        {"name": "outdoor_no", "path": ["attributes", "OutdoorSeating"],
         "check_type": "true", "value": "False",
         "natural_phrase": "indoor-only",
         "negative_justification": "weather concerns, privacy", "category": "seating"},
    ])

    # === Services ===
    conditions.extend([
        {"name": "takeout_yes", "path": ["attributes", "RestaurantsTakeOut"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with takeout", "category": "service"},
        {"name": "takeout_no", "path": ["attributes", "RestaurantsTakeOut"],
         "check_type": "true", "value": "False",
         "natural_phrase": "dine-in only",
         "negative_justification": "full dining experience", "category": "service"},
        {"name": "delivery_yes", "path": ["attributes", "RestaurantsDelivery"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with delivery", "category": "service"},
        {"name": "reservations_yes", "path": ["attributes", "RestaurantsReservations"],
         "check_type": "true", "value": "True",
         "natural_phrase": "that takes reservations", "category": "service"},
        {"name": "reservations_no", "path": ["attributes", "RestaurantsReservations"],
         "check_type": "true", "value": "False",
         "natural_phrase": "walk-in only",
         "negative_justification": "spontaneous visit, casual vibe", "category": "service"},
    ])

    # === Special Features ===
    conditions.extend([
        {"name": "drive_thru", "path": ["attributes", "DriveThru"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with a drive-thru", "category": "special"},
        {"name": "coat_check", "path": ["attributes", "CoatCheck"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with coat check", "category": "special"},
        {"name": "happy_hour", "path": ["attributes", "HappyHour"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with happy hour", "category": "special"},
        {"name": "has_tv", "path": ["attributes", "HasTV"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with TVs", "category": "special"},
        {"name": "no_tv", "path": ["attributes", "HasTV"],
         "check_type": "true", "value": "False",
         "natural_phrase": "without TVs",
         "negative_justification": "focused work, quiet conversation", "category": "special"},
        {"name": "byob", "path": ["attributes", "BYOB"],
         "check_type": "true", "value": "True",
         "natural_phrase": "that's BYOB", "category": "special"},
        {"name": "byob_corkage_free", "path": ["attributes", "BYOBCorkage"],
         "check_type": "contains", "value": "yes_free",
         "natural_phrase": "with free corkage", "category": "special"},
        {"name": "byob_corkage_no", "path": ["attributes", "BYOBCorkage"],
         "check_type": "contains", "value": "'no'",
         "natural_phrase": "without BYOB option", "category": "special"},
        {"name": "bike_parking", "path": ["attributes", "BikeParking"],
         "check_type": "true", "value": "True",
         "natural_phrase": "with bike parking", "category": "special"},
        {"name": "wheelchair", "path": ["attributes", "WheelchairAccessible"],
         "check_type": "true", "value": "True",
         "natural_phrase": "wheelchair accessible", "category": "special"},
        {"name": "credit_cards", "path": ["attributes", "BusinessAcceptsCreditCards"],
         "check_type": "true", "value": "True",
         "natural_phrase": "that accepts credit cards", "category": "special"},
    ])

    # === Ambience (contains checks) ===
    ambience_types = [
        ("hipster", "with a hipster vibe"),
        ("trendy", "trendy"),
        ("classy", "classy"),
        ("casual", "with a casual atmosphere"),
        ("romantic", "romantic"),
        ("intimate", "intimate"),
        ("divey", "divey"),
        ("touristy", "touristy"),
        ("upscale", "upscale"),
    ]
    for amb_type, phrase in ambience_types:
        conditions.append({
            "name": f"ambience_{amb_type}",
            "path": ["attributes", "Ambience"],
            "check_type": "contains", "value": f"'{amb_type}': True",
            "natural_phrase": phrase, "category": "ambience"
        })

    # === Good For Meal ===
    meal_types = [
        ("breakfast", "good for breakfast"),
        ("brunch", "good for brunch"),
        ("lunch", "good for lunch"),
        ("dinner", "good for dinner"),
        ("dessert", "good for dessert"),
        ("latenight", "good for late night"),
    ]
    for meal_type, phrase in meal_types:
        conditions.append({
            "name": f"meal_{meal_type}",
            "path": ["attributes", "GoodForMeal"],
            "check_type": "contains", "value": f"'{meal_type}': True",
            "natural_phrase": phrase, "category": "meal"
        })

    return conditions


def define_hours_conditions() -> List[dict]:
    """
    Define item_meta_hours conditions.

    Format: day + time range
    """
    conditions = []

    # Common time slots
    time_slots = [
        ("early_morning", "7:0-9:0", "early morning (7-9 AM)"),
        ("morning", "9:0-11:0", "morning (9-11 AM)"),
        ("lunch", "12:0-14:0", "lunch time (12-2 PM)"),
        ("afternoon", "14:0-17:0", "afternoon (2-5 PM)"),
        ("evening", "18:0-21:0", "evening (6-9 PM)"),
        ("late_night", "21:0-23:0", "late night (9-11 PM)"),
    ]

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Generate a subset of useful combinations
    key_combinations = [
        ("Monday", "early_morning", "7:0-9:0", "open Monday early morning"),
        ("Monday", "afternoon", "14:0-17:0", "open Monday afternoon"),
        ("Friday", "evening", "18:0-21:0", "open Friday evening"),
        ("Friday", "late_night", "21:0-23:0", "open Friday late night"),
        ("Saturday", "morning", "9:0-11:0", "open Saturday morning"),
        ("Saturday", "evening", "18:0-21:0", "open Saturday evening"),
        ("Sunday", "morning", "8:0-10:0", "open Sunday morning"),
        ("Sunday", "afternoon", "12:0-17:0", "open Sunday afternoon"),
    ]

    for day, slot_name, time_range, phrase in key_combinations:
        conditions.append({
            "name": f"hours_{day.lower()}_{slot_name}",
            "path": ["hours", day],
            "time_range": time_range,
            "natural_phrase": phrase,
            "category": "hours"
        })

    return conditions


def define_review_text_conditions() -> List[dict]:
    """
    Define review_text pattern conditions.

    These check if any review mentions a specific pattern.
    """
    patterns = [
        # Atmosphere/Vibe
        ("cozy", "cozy", "described as cozy"),
        ("quiet_reviews", "quiet", "praised for being quiet"),
        ("loud_reviews", "loud|noisy", "noted for being loud"),
        ("romantic_reviews", "romantic|date", "good for dates"),

        # Food & Drink
        ("coffee_reviews", "coffee", "praised for coffee"),
        ("espresso_reviews", "espresso", "known for espresso"),
        ("latte_reviews", "latte", "latte mentioned"),
        ("tea_reviews", "tea", "tea mentioned"),
        ("pastry_reviews", "pastry|croissant|muffin", "pastries mentioned"),
        ("brunch_reviews", "brunch", "brunch mentioned"),
        ("breakfast_reviews", "breakfast", "breakfast mentioned"),
        ("sandwich_reviews", "sandwich", "sandwiches mentioned"),
        ("wine_reviews", "wine", "wine mentioned"),
        ("cocktail_reviews", "cocktail", "cocktails mentioned"),
        ("beer_reviews", "beer", "beer mentioned"),

        # Experience
        ("study_reviews", "study|studying", "good for studying"),
        ("work_reviews", "work|working|laptop", "good for working"),
        ("meeting_reviews", "meeting", "good for meetings"),
        ("friendly_reviews", "friendly", "friendly service"),
        ("fast_reviews", "fast|quick", "fast service"),
        ("slow_reviews", "slow", "slow service noted"),

        # Specific mentions
        ("wifi_reviews", "wifi|wi-fi", "WiFi mentioned"),
        ("outdoor_reviews", "outdoor|patio|terrace", "outdoor seating mentioned"),
        ("music_reviews", "music", "music mentioned"),
        ("art_reviews", "art", "art mentioned"),
        ("books_reviews", "book", "books mentioned"),

        # Quality
        ("best_reviews", "best", "called 'the best'"),
        ("love_reviews", "love", "'love' mentioned"),
        ("favorite_reviews", "favorite", "called a favorite"),
        ("hidden_gem_reviews", "hidden gem", "called a hidden gem"),
        ("recommend_reviews", "recommend", "recommended"),

        # Unique patterns
        ("vegan_reviews", "vegan", "vegan options mentioned"),
        ("gluten_reviews", "gluten", "gluten-free mentioned"),
        ("organic_reviews", "organic", "organic mentioned"),
    ]

    conditions = []
    for name, pattern, phrase in patterns:
        conditions.append({
            "name": name,
            "pattern": pattern,
            "natural_phrase": f"where reviews mention '{pattern.split('|')[0]}'",
            "category": "review_text"
        })

    return conditions


def define_review_meta_conditions() -> List[dict]:
    """
    Define review_meta conditions (weighted reviews - G04 style).

    These use credibility-count evaluation: "At least N credible reviewers
    (above percentile) mention pattern"

    Defaults:
        credibility_percentile: 50 (above median)
        min_credible_matches: 2 (at least 2 agree)
    """
    # Weight fields available
    weight_fields = [
        (["user", "review_count"], "experienced reviewers", "reviewer_experience"),
        (["user", "fans"], "popular reviewers", "reviewer_popularity"),
        (["user", "elite"], "elite reviewers", "elite_status"),
        (["useful"], "helpful reviews", "review_helpfulness"),
    ]

    # Patterns to combine with weights
    patterns = ["coffee", "love", "best", "recommend", "cozy", "work"]

    conditions = []
    for field, field_phrase, field_name in weight_fields:
        for pattern in patterns:
            conditions.append({
                "name": f"review_meta_{field_name}_{pattern}",
                "pattern": pattern,
                "weight_by": {"field": field},
                # Uses defaults: credibility_percentile=50, min_credible_matches=2
                "natural_phrase": f"where {field_phrase} mention '{pattern}'",
                "category": "review_meta"
            })

    return conditions


# =============================================================================
# CONDITION EVALUATION
# =============================================================================

def check_item_meta(restaurant: dict, condition: dict) -> bool:
    """Check if restaurant satisfies an item_meta condition."""
    # Navigate path
    value = restaurant
    for key in condition["path"]:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = None
            break

    if value is None:
        return False

    value_str = str(value)
    check_type = condition["check_type"]
    target = condition["value"]

    if check_type == "true":
        return target in value_str or value_str == target
    elif check_type == "contains":
        return target in value_str
    elif check_type == "not_true":
        return target not in value_str

    return False


def check_hours(restaurant: dict, condition: dict) -> bool:
    """Check if restaurant is open during specified hours."""
    hours = restaurant.get("hours", {})
    day = condition["path"][1]
    day_hours = hours.get(day)

    if not day_hours:
        return False

    # Parse time range from condition
    time_range = condition["time_range"]
    req_start, req_end = time_range.split("-")
    req_start_h = int(req_start.split(":")[0])
    req_end_h = int(req_end.split(":")[0])

    # Parse restaurant hours (format: "7:0-22:0")
    try:
        open_time, close_time = day_hours.split("-")
        open_h = int(open_time.split(":")[0])
        close_h = int(close_time.split(":")[0])

        # Check if requested range is within open hours
        return open_h <= req_start_h and close_h >= req_end_h
    except:
        return False


def check_review_text(reviews: List[dict], condition: dict) -> bool:
    """Check if any review matches the pattern."""
    pattern = condition["pattern"]
    regex = re.compile(pattern, re.IGNORECASE)

    for review in reviews:
        text = review.get("text", "")
        if regex.search(text):
            return True
    return False


def check_review_meta(reviews: List[dict], condition: dict) -> bool:
    """Check credibility-count review condition (G04 style).

    Semantics: "At least N credible reviewers (above percentile) mention pattern"

    Uses defaults:
        credibility_percentile: 50 (above median)
        min_credible_matches: 2 (at least 2 agree)
    """
    pattern = condition["pattern"]
    field_path = condition["weight_by"]["field"]
    credibility_percentile = condition.get("credibility_percentile", 50)
    min_credible_matches = condition.get("min_credible_matches", 2)

    if not reviews:
        return False

    regex = re.compile(pattern, re.IGNORECASE)

    # Extract field values for each review
    values = []
    for review in reviews:
        v = review
        for key in field_path:
            if isinstance(v, dict):
                v = v.get(key, 0)
            else:
                v = 0
                break

        # Handle elite (list of years)
        if isinstance(v, list):
            v = len([e for e in v if e and e != 'None'])
        elif isinstance(v, str):
            try:
                v = float(v)
            except:
                v = 0

        values.append(float(v) if v else 0.0)

    # Compute credibility threshold (percentile of non-zero values)
    nonzero = sorted([v for v in values if v > 0])
    if len(nonzero) < min_credible_matches:
        return False  # Not enough credible reviewers exist

    threshold_idx = int(len(nonzero) * credibility_percentile / 100)
    cred_threshold = nonzero[min(threshold_idx, len(nonzero) - 1)]

    # Count credible reviewers who mention pattern
    credible_matches = sum(
        1 for review, v in zip(reviews, values)
        if v >= cred_threshold and regex.search(review.get("text", ""))
    )

    return credible_matches >= min_credible_matches


# =============================================================================
# MATRIX BUILDING
# =============================================================================

def build_condition_matrix(restaurants: List[dict], reviews_by_biz: Dict[str, List[dict]]) -> dict:
    """
    Build the complete condition satisfaction matrix.

    Returns:
        {
            "conditions": {
                "condition_name": {
                    "type": "item_meta|item_meta_hours|review_text|review_meta",
                    "definition": {...},
                    "satisfying_restaurants": [0, 3, 5, ...],
                    "count": 3
                },
                ...
            },
            "restaurants": {
                "0": {
                    "name": "...",
                    "business_id": "...",
                    "satisfying_conditions": ["cond1", "cond2", ...]
                },
                ...
            },
            "unique_conditions": ["cond1", "cond2", ...],  # conditions with count=1
            "rare_conditions": ["cond1", "cond2", ...],    # conditions with count<=3
        }
    """
    # Get all condition definitions
    item_meta_conds = define_item_meta_conditions()
    hours_conds = define_hours_conditions()
    review_text_conds = define_review_text_conditions()
    review_meta_conds = define_review_meta_conditions()

    matrix = {
        "conditions": {},
        "restaurants": {},
        "unique_conditions": [],
        "rare_conditions": [],
    }

    # Initialize restaurant entries
    for i, rest in enumerate(restaurants):
        matrix["restaurants"][str(i)] = {
            "name": rest["name"],
            "business_id": rest["business_id"],
            "satisfying_conditions": []
        }

    # Check item_meta conditions
    for cond in item_meta_conds:
        satisfying = []
        for i, rest in enumerate(restaurants):
            if check_item_meta(rest, cond):
                satisfying.append(i)
                matrix["restaurants"][str(i)]["satisfying_conditions"].append(cond["name"])

        matrix["conditions"][cond["name"]] = {
            "type": "item_meta",
            "definition": cond,
            "satisfying_restaurants": satisfying,
            "count": len(satisfying)
        }

    # Check hours conditions
    for cond in hours_conds:
        satisfying = []
        for i, rest in enumerate(restaurants):
            if check_hours(rest, cond):
                satisfying.append(i)
                matrix["restaurants"][str(i)]["satisfying_conditions"].append(cond["name"])

        matrix["conditions"][cond["name"]] = {
            "type": "item_meta_hours",
            "definition": cond,
            "satisfying_restaurants": satisfying,
            "count": len(satisfying)
        }

    # Check review_text conditions
    for cond in review_text_conds:
        satisfying = []
        for i, rest in enumerate(restaurants):
            biz_reviews = reviews_by_biz.get(rest["business_id"], [])
            if check_review_text(biz_reviews, cond):
                satisfying.append(i)
                matrix["restaurants"][str(i)]["satisfying_conditions"].append(cond["name"])

        matrix["conditions"][cond["name"]] = {
            "type": "review_text",
            "definition": cond,
            "satisfying_restaurants": satisfying,
            "count": len(satisfying)
        }

    # Check review_meta conditions
    for cond in review_meta_conds:
        satisfying = []
        for i, rest in enumerate(restaurants):
            biz_reviews = reviews_by_biz.get(rest["business_id"], [])
            if check_review_meta(biz_reviews, cond):
                satisfying.append(i)
                matrix["restaurants"][str(i)]["satisfying_conditions"].append(cond["name"])

        matrix["conditions"][cond["name"]] = {
            "type": "review_meta",
            "definition": cond,
            "satisfying_restaurants": satisfying,
            "count": len(satisfying)
        }

    # Identify unique and rare conditions
    for name, data in matrix["conditions"].items():
        if data["count"] == 1:
            matrix["unique_conditions"].append(name)
        if 1 <= data["count"] <= 3:
            matrix["rare_conditions"].append(name)

    # Sort by count
    matrix["unique_conditions"].sort()
    matrix["rare_conditions"].sort(key=lambda x: matrix["conditions"][x]["count"])

    return matrix


def find_unique_combinations(matrix: dict, target_idx: int, max_conditions: int = 3) -> List[List[str]]:
    """
    Find minimal condition combinations that uniquely identify a restaurant.

    Returns list of condition combinations, sorted by size then by condition rarity.
    """
    target_conds = set(matrix["restaurants"][str(target_idx)]["satisfying_conditions"])

    # First check if any single condition is unique
    for cond_name in target_conds:
        if matrix["conditions"][cond_name]["count"] == 1:
            return [[cond_name]]

    # Try combinations of increasing size
    from itertools import combinations

    results = []
    target_conds_list = list(target_conds)

    for size in range(2, min(max_conditions + 1, len(target_conds_list) + 1)):
        for combo in combinations(target_conds_list, size):
            # Find intersection of all satisfying restaurants
            satisfying = None
            for cond_name in combo:
                cond_satisfying = set(matrix["conditions"][cond_name]["satisfying_restaurants"])
                if satisfying is None:
                    satisfying = cond_satisfying
                else:
                    satisfying = satisfying & cond_satisfying

            if satisfying == {target_idx}:
                results.append(list(combo))

        if results:  # Found combinations at this size
            break

    return results


def generate_summary(matrix: dict, restaurants: List[dict]) -> str:
    """Generate human-readable summary."""
    lines = [
        "# Condition Matrix Summary",
        "",
        f"Generated for {len(restaurants)} restaurants",
        "",
        "## Statistics",
        "",
        f"- Total conditions defined: {len(matrix['conditions'])}",
        f"- Unique conditions (count=1): {len(matrix['unique_conditions'])}",
        f"- Rare conditions (count<=3): {len(matrix['rare_conditions'])}",
        "",
        "## Condition Types",
        "",
    ]

    # Count by type
    type_counts = defaultdict(int)
    for cond_data in matrix["conditions"].values():
        type_counts[cond_data["type"]] += 1

    for ctype, count in sorted(type_counts.items()):
        lines.append(f"- {ctype}: {count} conditions")

    lines.extend([
        "",
        "## Unique Conditions (can uniquely identify a restaurant)",
        "",
    ])

    for cond_name in matrix["unique_conditions"]:
        cond = matrix["conditions"][cond_name]
        rest_idx = cond["satisfying_restaurants"][0]
        rest_name = restaurants[rest_idx]["name"]
        lines.append(f"- **{cond_name}**: [{rest_idx}] {rest_name}")

    lines.extend([
        "",
        "## Rare Conditions (count <= 3)",
        "",
        "| Condition | Count | Restaurants |",
        "|-----------|-------|-------------|",
    ])

    for cond_name in matrix["rare_conditions"]:
        cond = matrix["conditions"][cond_name]
        rest_names = [f"[{i}]" for i in cond["satisfying_restaurants"]]
        lines.append(f"| {cond_name} | {cond['count']} | {', '.join(rest_names)} |")

    lines.extend([
        "",
        "## Restaurant Coverage",
        "",
        "Restaurants with unique identifiers:",
        "",
    ])

    for i, rest in enumerate(restaurants):
        combos = find_unique_combinations(matrix, i, max_conditions=3)
        if combos:
            best = combos[0]
            lines.append(f"- [{i}] {rest['name']}: `{' + '.join(best)}`")
        else:
            lines.append(f"- [{i}] {rest['name']}: *(needs 4+ conditions)*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate condition satisfaction matrix")
    parser.add_argument("--output", choices=["json", "summary", "both"], default="both")
    args = parser.parse_args()

    print("Loading data...")
    restaurants, reviews = load_data()
    reviews_by_biz = build_review_index(reviews)

    print(f"Loaded {len(restaurants)} restaurants, {len(reviews)} reviews")

    print("Building condition matrix...")
    matrix = build_condition_matrix(restaurants, reviews_by_biz)

    print(f"Defined {len(matrix['conditions'])} conditions")
    print(f"Found {len(matrix['unique_conditions'])} unique conditions")
    print(f"Found {len(matrix['rare_conditions'])} rare conditions")

    if args.output in ["json", "both"]:
        print(f"Writing {OUTPUT_MATRIX}...")
        with open(OUTPUT_MATRIX, "w") as f:
            json.dump(matrix, f, indent=2)

    if args.output in ["summary", "both"]:
        print(f"Writing {OUTPUT_SUMMARY}...")
        summary = generate_summary(matrix, restaurants)
        with open(OUTPUT_SUMMARY, "w") as f:
            f.write(summary)

    print("Done!")


if __name__ == "__main__":
    main()
