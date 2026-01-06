#!/usr/bin/env python3
"""Generate G05-G08 requests with nested condition structures.

G05: OR(A, B, C) - Triple OR
G06: OR(AND(A,B), AND(C,D)) - Nested OR with AND branches
G07: AND(OR(a,b), OR(c,d)) - Chained double OR
G08: AND(A, OR(B, AND(C,D))) - Unbalanced nesting

Usage:
    python data/scripts/generate_g05_g08.py --group G05 --dry-run
    python data/scripts/generate_g05_g08.py --group all
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

DATA_DIR = Path(__file__).parent.parent / "philly_cafes"

# ===== Condition Building Helpers =====

def make_item_meta(aspect: str, attr: str, value: Any) -> dict:
    """Create item_meta condition."""
    return {
        "aspect": aspect,
        "evidence": {
            "kind": "item_meta",
            "path": ["attributes", attr],
            "true": str(value)
        }
    }

def make_item_meta_contains(aspect: str, attr: str, contains: str) -> dict:
    """Create item_meta condition with contains."""
    return {
        "aspect": aspect,
        "evidence": {
            "kind": "item_meta",
            "path": ["attributes", attr],
            "contains": contains
        }
    }

def make_review_text(aspect: str, pattern: str, weight_by: Optional[dict] = None) -> dict:
    """Create review_text condition."""
    cond = {
        "aspect": aspect,
        "evidence": {
            "kind": "review_text",
            "pattern": pattern
        }
    }
    if weight_by:
        cond["evidence"]["weight_by"] = weight_by
    return cond

def make_hours(aspect: str, day: str, time_check: str) -> dict:
    """Create item_meta_hours condition."""
    return {
        "aspect": aspect,
        "evidence": {
            "kind": "item_meta_hours",
            "path": ["hours", day],
            "true": time_check
        }
    }

def make_or(*args) -> dict:
    """Create OR condition."""
    return {"op": "OR", "args": list(args)}

def make_and(*args) -> dict:
    """Create AND condition."""
    return {"op": "AND", "args": list(args)}


# ===== Condition Library =====

# Boolean attributes (for varied conditions)
BOOL_ATTRS = [
    ("good_for_kids", "GoodForKids", "True"),
    ("not_good_for_kids", "GoodForKids", "False"),
    ("has_tv", "HasTV", "True"),
    ("no_tv", "HasTV", "False"),
    ("dog_friendly", "DogsAllowed", "True"),
    ("bike_parking", "BikeParking", "True"),
    ("outdoor_seating", "OutdoorSeating", "True"),
    ("indoor_only", "OutdoorSeating", "False"),
    ("takes_reservations", "RestaurantsReservations", "True"),
    ("good_for_groups", "RestaurantsGoodForGroups", "True"),
    ("has_takeout", "RestaurantsTakeOut", "True"),
    ("has_delivery", "RestaurantsDelivery", "True"),
    ("coat_check", "CoatCheck", "True"),
    ("wheelchair", "WheelchairAccessible", "True"),
    ("accepts_cc", "BusinessAcceptsCreditCards", "True"),
]

# Alcohol options
ALCOHOL_ATTRS = [
    ("full_bar", "Alcohol", "u'full_bar'"),
    ("beer_wine", "Alcohol", "u'beer_and_wine'"),
    ("no_alcohol", "Alcohol", "u'none'"),
]

# WiFi options
WIFI_ATTRS = [
    ("free_wifi", "WiFi", "u'free'"),
    ("no_wifi", "WiFi", "u'no'"),
]

# Noise levels
NOISE_ATTRS = [
    ("quiet", "NoiseLevel", "u'quiet'"),
    ("moderate_noise", "NoiseLevel", "u'average'"),
    ("lively", "NoiseLevel", "u'loud'"),
]

# Price ranges
PRICE_ATTRS = [
    ("budget", "RestaurantsPriceRange2", "1"),
    ("mid_price", "RestaurantsPriceRange2", "2"),
    ("upscale", "RestaurantsPriceRange2", "3"),
]

# Review text patterns (distinct concepts, not synonyms)
REVIEW_PATTERNS = [
    ("study_spot", "study"),
    ("work_friendly", "laptop"),
    ("meeting_spot", "meeting"),
    ("brunch_spot", "brunch"),
    ("wine_mention", "wine"),
    ("cocktail_mention", "cocktail"),
    ("coffee_quality", "coffee"),
    ("espresso", "espresso"),
    ("pastry", "pastry"),
    ("breakfast", "breakfast"),
    ("lunch", "lunch"),
    ("quiet_mention", "quiet"),
    ("cozy", "cozy"),
    ("friendly_staff", "friendly"),
]

# Ambience options (via contains)
AMBIENCE_OPTIONS = [
    ("hipster_vibe", "Ambience", "'hipster': True"),
    ("trendy", "Ambience", "'trendy': True"),
    ("casual_vibe", "Ambience", "'casual': True"),
    ("romantic", "Ambience", "'romantic': True"),
]

# Hours options (using time range format for validator compatibility)
HOURS_OPTIONS = [
    ("friday_evening", "Friday", "21:0-22:0"),
    ("saturday_morning", "Saturday", "10:0-11:0"),
    ("sunday_brunch", "Sunday", "11:0-12:0"),
    ("weekday_late", "Wednesday", "21:0-22:0"),
    ("early_morning", "Monday", "7:0-8:0"),
]

# Weighted review options
WEIGHTED_REVIEW_OPTIONS = [
    ("experienced_reviewers", {"field": ["user", "review_count"]}),
    ("popular_reviewers", {"field": ["user", "fans"]}),
    ("elite_reviewers", {"field": ["user", "elite"]}),
]


# ===== Evaluation Helpers =====

def load_data():
    """Load restaurants and reviews."""
    with open(DATA_DIR / "restaurants.jsonl") as f:
        restaurants = [json.loads(l) for l in f]

    with open(DATA_DIR / "reviews.jsonl") as f:
        reviews_list = [json.loads(l) for l in f]

    # Group reviews by business
    reviews_by_biz = {}
    for r in reviews_list:
        bid = r["business_id"]
        if bid not in reviews_by_biz:
            reviews_by_biz[bid] = []
        reviews_by_biz[bid].append(r)

    return restaurants, reviews_by_biz


def check_condition(restaurant: dict, reviews: List[dict], condition: dict) -> bool:
    """Check if restaurant satisfies a single condition."""
    ev = condition.get("evidence", {})
    kind = ev.get("kind")

    if kind == "item_meta":
        path = ev.get("path", [])
        if len(path) < 2:
            return False
        attr = path[1]
        attrs = restaurant.get("attributes", {})
        actual = str(attrs.get(attr, ""))

        if "true" in ev:
            return actual == ev["true"]
        if "not_true" in ev:
            return actual != ev["not_true"]
        if "contains" in ev:
            return ev["contains"] in actual
        return False

    elif kind == "review_text":
        pattern = ev.get("pattern", "").lower()
        for r in reviews:
            if pattern in r.get("text", "").lower():
                return True
        return False

    elif kind == "item_meta_hours":
        path = ev.get("path", [])
        day = path[1] if len(path) > 1 else ""
        hours = restaurant.get("hours", {})
        day_hours = hours.get(day, "")
        required = ev.get("true", "")

        # Time range format: "21:0-22:0" means restaurant must be open during this range
        if not day_hours or "-" not in day_hours:
            return False
        if not required or "-" not in required:
            return False
        try:
            # Parse restaurant hours
            open_str, close_str = day_hours.split("-")
            def to_mins(t):
                h, m = t.split(":")
                return int(h) * 60 + int(m)
            item_start = to_mins(open_str)
            item_end = to_mins(close_str)
            # Handle overnight hours
            if item_end < item_start:
                item_end += 24 * 60

            # Parse required hours
            req_open, req_close = required.split("-")
            req_start = to_mins(req_open)
            req_end = to_mins(req_close)

            # Check if restaurant hours cover the required range
            return item_start <= req_start and item_end >= req_end
        except:
            return False

    return False


def evaluate_structure(restaurant: dict, reviews: List[dict], structure: dict) -> bool:
    """Evaluate if restaurant satisfies the structure (recursively handles AND/OR)."""
    op = structure.get("op")

    if op == "AND":
        args = structure.get("args", [])
        return all(evaluate_structure(restaurant, reviews, arg) for arg in args)

    elif op == "OR":
        args = structure.get("args", [])
        return any(evaluate_structure(restaurant, reviews, arg) for arg in args)

    else:
        # Leaf condition
        return check_condition(restaurant, reviews, structure)


def find_matching_restaurants(restaurants: List[dict], reviews_by_biz: dict, structure: dict) -> List[int]:
    """Find all restaurants matching the structure."""
    matching = []
    for i, r in enumerate(restaurants):
        reviews = reviews_by_biz.get(r["business_id"], [])
        if evaluate_structure(r, reviews, structure):
            matching.append(i)
    return matching


# ===== Text Rendering =====

# Import existing renderer helpers
from fix_g01_requests import CONDITION_TEXT, AMBIENCE_TEXT, negate_phrase, format_time_range


def leaf_to_short_phrase(cond: dict) -> str:
    """Convert leaf condition to short natural phrase (for use in OR lists)."""
    ev = cond.get("evidence", {})
    kind = ev.get("kind")

    if kind == "item_meta":
        path = tuple(ev.get("path", []))
        if "true" in ev:
            key = (*path, ev["true"])
            if key in CONDITION_TEXT:
                return CONDITION_TEXT[key]
            return f"{path[-1]}={ev['true']}"
        if "contains" in ev:
            contains = ev["contains"]
            if "Ambience" in path and contains in AMBIENCE_TEXT:
                return AMBIENCE_TEXT[contains]
            return f"has {contains}"
        if "not_true" in ev:
            key = (*path, ev["not_true"])
            if key in CONDITION_TEXT:
                return negate_phrase(CONDITION_TEXT[key])
            return f"not {path[-1]}"

    elif kind == "review_text":
        pattern = ev.get("pattern", "")
        weight_by = ev.get("weight_by", {})
        field = weight_by.get("field", [])

        # Convert review patterns to descriptive phrases
        pattern_phrases = {
            "study": "good for studying",
            "laptop": "laptop-friendly",
            "meeting": "good for meetings",
            "brunch": "great for brunch",
            "wine": "has wine mentioned in reviews",
            "cocktail": "known for cocktails",
            "coffee": "praised for coffee",
            "espresso": "known for espresso",
            "breakfast": "good for breakfast",
            "lunch": "good for lunch",
            "dinner": "good for dinner",
            "quiet": "described as quiet",
            "cozy": "cozy atmosphere",
            "friendly": "friendly service",
            "pastry": "good pastries",
            "romantic": "romantic atmosphere",
            "latte": "good lattes",
            "quick": "quick service",
        }
        phrase = pattern_phrases.get(pattern, f"reviews mention '{pattern}'")

        if field == ["user", "review_count"]:
            return f"experienced reviewers say {phrase}"
        elif field == ["user", "fans"]:
            return f"popular reviewers mention {phrase}"
        elif field == ["user", "elite"]:
            return f"elite reviewers recommend for {pattern}"

        return phrase

    elif kind == "item_meta_hours":
        path = ev.get("path", [])
        day = path[1] if len(path) > 1 else ""
        time_range = ev.get("true", "")
        # Parse time range (e.g., "21:0-22:0")
        try:
            start_time = int(time_range.split(":")[0])
        except:
            start_time = 12  # default to noon
        if start_time >= 20:
            return f"open {day} evening"
        elif start_time <= 8:
            return f"open early {day}"
        elif start_time <= 12:
            return f"open {day} morning"
        return f"open on {day}"

    return "[unknown]"


def render_or_list(args: list) -> str:
    """Render a list of OR conditions naturally."""
    phrases = []
    for arg in args:
        if arg.get("op") == "AND":
            # Nested AND inside OR
            and_phrases = [leaf_to_short_phrase(a) for a in arg.get("args", [])]
            phrases.append(" and ".join(and_phrases))
        else:
            phrases.append(leaf_to_short_phrase(arg))

    if len(phrases) == 2:
        return f"{phrases[0]}, or {phrases[1]}"
    elif len(phrases) == 3:
        return f"{phrases[0]}, {phrases[1]}, or {phrases[2]}"
    else:
        return ", ".join(phrases[:-1]) + f", or {phrases[-1]}"


def render_structure_text(structure: dict) -> str:
    """Render top-level structure to natural text."""
    op = structure.get("op")
    args = structure.get("args", [])

    if op == "OR":
        # Top-level OR: "Looking for a cafe that's either A, B, or C"
        or_text = render_or_list(args)
        return f"Looking for a cafe that's either {or_text}"

    elif op == "AND":
        # Collect parts
        parts = []
        for arg in args:
            if arg.get("op") == "OR":
                or_text = render_or_list(arg.get("args", []))
                parts.append(f"either {or_text}")
            elif arg.get("op") == "AND":
                and_phrases = [leaf_to_short_phrase(a) for a in arg.get("args", [])]
                parts.append(" and ".join(and_phrases))
            else:
                parts.append(leaf_to_short_phrase(arg))

        if len(parts) == 1:
            return f"Looking for a cafe that's {parts[0]}"
        elif len(parts) == 2:
            return f"Looking for a cafe that's {parts[0]}, and {parts[1]}"
        else:
            main = ", ".join(parts[:-1])
            return f"Looking for a cafe that's {main}, and {parts[-1]}"

    return "Looking for a cafe"


# ===== Structure Generators =====

def generate_g05_structures(seed: int = 42) -> List[dict]:
    """Generate G05: OR(A, B, C) - Triple OR structures."""
    random.seed(seed)
    structures = []

    # Combinations to try (varied evidence types)
    combos = [
        # All review_text
        [make_review_text("study", "study"),
         make_review_text("work", "laptop"),
         make_review_text("meeting", "meeting")],
        [make_review_text("brunch", "brunch"),
         make_review_text("coffee", "coffee"),
         make_review_text("breakfast", "breakfast")],
        [make_review_text("wine", "wine"),
         make_review_text("cocktail", "cocktail"),
         make_review_text("espresso", "espresso")],
        # Mixed: item_meta + review_text
        [make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
         make_item_meta("outdoor", "OutdoorSeating", "True"),
         make_review_text("cozy", "cozy")],
        [make_item_meta("full_bar", "Alcohol", "u'full_bar'"),
         make_item_meta("beer_wine", "Alcohol", "u'beer_and_wine'"),
         make_review_text("wine", "wine")],
        # item_meta only
        [make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
         make_item_meta("moderate", "NoiseLevel", "u'average'"),
         make_item_meta("lively", "NoiseLevel", "u'loud'")],
        [make_item_meta("budget", "RestaurantsPriceRange2", "1"),
         make_item_meta("mid", "RestaurantsPriceRange2", "2"),
         make_item_meta("upscale", "RestaurantsPriceRange2", "3")],
        # hours + review
        [make_hours("friday", "Friday", "21:0-22:0"),
         make_hours("saturday", "Saturday", "10:0-11:0"),
         make_review_text("brunch", "brunch")],
        # Mixed all types
        [make_item_meta("dog_friendly", "DogsAllowed", "True"),
         make_review_text("friendly", "friendly"),
         make_hours("sunday", "Sunday", "11:0-12:0")],
        [make_item_meta_contains("hipster", "Ambience", "'hipster': True"),
         make_review_text("coffee", "coffee"),
         make_item_meta("free_wifi", "WiFi", "u'free'")],
    ]

    for combo in combos:
        structures.append(make_or(*combo))

    return structures


def generate_g06_structures(seed: int = 42) -> List[dict]:
    """Generate G06: OR(AND(A,B), AND(C,D)) - Nested OR with AND branches."""
    random.seed(seed)
    structures = []

    # Different branch combinations
    combos = [
        # Both branches item_meta
        (make_and(make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
                  make_item_meta("outdoor", "OutdoorSeating", "True")),
         make_and(make_item_meta("lively", "NoiseLevel", "u'loud'"),
                  make_item_meta("full_bar", "Alcohol", "u'full_bar'"))),
        # Branch 1: item_meta, Branch 2: review
        (make_and(make_item_meta("budget", "RestaurantsPriceRange2", "1"),
                  make_item_meta("free_wifi", "WiFi", "u'free'")),
         make_and(make_review_text("study", "study"),
                  make_review_text("quiet", "quiet"))),
        # Both branches mixed
        (make_and(make_review_text("brunch", "brunch"),
                  make_hours("saturday", "Saturday", "10:0-11:0")),
         make_and(make_review_text("dinner", "dinner"),
                  make_hours("friday", "Friday", "21:0-22:0"))),
        (make_and(make_item_meta("dog_friendly", "DogsAllowed", "True"),
                  make_item_meta("outdoor", "OutdoorSeating", "True")),
         make_and(make_item_meta("good_for_kids", "GoodForKids", "True"),
                  make_item_meta("has_tv", "HasTV", "True"))),
        (make_and(make_item_meta_contains("romantic", "Ambience", "'romantic': True"),
                  make_review_text("wine", "wine")),
         make_and(make_item_meta_contains("casual", "Ambience", "'casual': True"),
                  make_review_text("coffee", "coffee"))),
        # Price + coffee focus (removed weight_by for simpler validation)
        (make_and(make_review_text("coffee", "coffee"),
                  make_item_meta("upscale", "RestaurantsPriceRange2", "3")),
         make_and(make_review_text("espresso", "espresso"),
                  make_item_meta("mid", "RestaurantsPriceRange2", "2"))),
        # Hours focus
        (make_and(make_hours("monday", "Monday", "7:0-8:0"),
                  make_review_text("breakfast", "breakfast")),
         make_and(make_hours("sunday", "Sunday", "11:0-12:0"),
                  make_review_text("brunch", "brunch"))),
        (make_and(make_item_meta("takes_reservations", "RestaurantsReservations", "True"),
                  make_item_meta("good_for_groups", "RestaurantsGoodForGroups", "True")),
         make_and(make_item_meta("has_takeout", "RestaurantsTakeOut", "True"),
                  make_item_meta("quick", "NoiseLevel", "u'average'"))),
        (make_and(make_item_meta("wheelchair", "WheelchairAccessible", "True"),
                  make_review_text("friendly", "friendly")),
         make_and(make_item_meta("bike_parking", "BikeParking", "True"),
                  make_review_text("coffee", "coffee"))),
        (make_and(make_item_meta_contains("trendy", "Ambience", "'trendy': True"),
                  make_review_text("cocktail", "cocktail")),
         make_and(make_item_meta_contains("hipster", "Ambience", "'hipster': True"),
                  make_review_text("espresso", "espresso"))),
    ]

    for branch1, branch2 in combos:
        structures.append(make_or(branch1, branch2))

    return structures


def generate_g07_structures(seed: int = 42) -> List[dict]:
    """Generate G07: AND(OR(a,b), OR(c,d)) - Chained double OR."""
    random.seed(seed)
    structures = []

    # Different OR pair combinations
    combos = [
        # Both ORs item_meta
        (make_or(make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
                 make_item_meta("moderate", "NoiseLevel", "u'average'")),
         make_or(make_item_meta("outdoor", "OutdoorSeating", "True"),
                 make_item_meta("free_wifi", "WiFi", "u'free'"))),
        # OR1 item_meta, OR2 review
        (make_or(make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
                 make_review_text("romantic", "romantic")),
         make_or(make_review_text("brunch", "brunch"),
                 make_review_text("wine", "wine"))),
        # Both ORs review
        (make_or(make_review_text("study", "study"),
                 make_review_text("work", "laptop")),
         make_or(make_review_text("coffee", "coffee"),
                 make_review_text("quiet", "quiet"))),
        # OR with hours
        (make_or(make_hours("friday", "Friday", "21:0-22:0"),
                 make_hours("saturday", "Saturday", "21:0-22:0")),
         make_or(make_review_text("cocktail", "cocktail"),
                 make_review_text("wine", "wine"))),
        # Alcohol options
        (make_or(make_item_meta("full_bar", "Alcohol", "u'full_bar'"),
                 make_item_meta("beer_wine", "Alcohol", "u'beer_and_wine'")),
         make_or(make_review_text("dinner", "dinner"),
                 make_hours("friday", "Friday", "21:0-22:0"))),
        # Price flexibility
        (make_or(make_item_meta("budget", "RestaurantsPriceRange2", "1"),
                 make_item_meta("mid", "RestaurantsPriceRange2", "2")),
         make_or(make_review_text("coffee", "coffee"),
                 make_review_text("pastry", "pastry"))),
        # Ambience + reviews
        (make_or(make_item_meta_contains("hipster", "Ambience", "'hipster': True"),
                 make_item_meta_contains("trendy", "Ambience", "'trendy': True")),
         make_or(make_review_text("espresso", "espresso"),
                 make_review_text("latte", "latte"))),
        # Accessibility
        (make_or(make_item_meta("wheelchair", "WheelchairAccessible", "True"),
                 make_item_meta("dog_friendly", "DogsAllowed", "True")),
         make_or(make_review_text("friendly", "friendly"),
                 make_item_meta("outdoor", "OutdoorSeating", "True"))),
        # Mixed everything
        (make_or(make_item_meta("good_for_kids", "GoodForKids", "True"),
                 make_item_meta("has_tv", "HasTV", "True")),
         make_or(make_review_text("breakfast", "breakfast"),
                 make_hours("saturday", "Saturday", "10:0-11:0"))),
        (make_or(make_item_meta("has_delivery", "RestaurantsDelivery", "True"),
                 make_item_meta("has_takeout", "RestaurantsTakeOut", "True")),
         make_or(make_review_text("lunch", "lunch"),
                 make_review_text("quick", "quick"))),
    ]

    for or1, or2 in combos:
        structures.append(make_and(or1, or2))

    return structures


def generate_g08_structures(seed: int = 42) -> List[dict]:
    """Generate G08: AND(A, OR(B, AND(C,D))) - Unbalanced nesting."""
    random.seed(seed)
    structures = []

    # Different unbalanced combinations
    combos = [
        # A=item_meta, B=review, AND(C,D)=item_meta+hours
        (make_item_meta("upscale", "RestaurantsPriceRange2", "3"),
         make_review_text("wine", "wine"),
         make_and(make_review_text("brunch", "brunch"),
                  make_hours("saturday", "Saturday", "10:0-11:0"))),
        # A=review, B=item_meta, AND(C,D)=mixed
        (make_review_text("coffee", "coffee"),
         make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
         make_and(make_item_meta("free_wifi", "WiFi", "u'free'"),
                  make_review_text("study", "study"))),
        # A=hours, B=review, AND(C,D)=item_meta
        (make_hours("friday", "Friday", "21:0-22:0"),
         make_review_text("cocktail", "cocktail"),
         make_and(make_item_meta("full_bar", "Alcohol", "u'full_bar'"),
                  make_item_meta("lively", "NoiseLevel", "u'loud'"))),
        # A=ambience, B=hours, AND(C,D)=review
        (make_item_meta_contains("hipster", "Ambience", "'hipster': True"),
         make_hours("saturday", "Saturday", "21:0-22:0"),
         make_and(make_review_text("espresso", "espresso"),
                  make_review_text("coffee", "coffee"))),
        # Various mixes
        (make_item_meta("budget", "RestaurantsPriceRange2", "1"),
         make_item_meta("free_wifi", "WiFi", "u'free'"),
         make_and(make_review_text("study", "study"),
                  make_review_text("quiet", "quiet"))),
        (make_item_meta("outdoor", "OutdoorSeating", "True"),
         make_review_text("brunch", "brunch"),
         make_and(make_hours("sunday", "Sunday", "11:0-12:0"),
                  make_item_meta("dog_friendly", "DogsAllowed", "True"))),
        (make_item_meta("good_for_groups", "RestaurantsGoodForGroups", "True"),
         make_item_meta("takes_reservations", "RestaurantsReservations", "True"),
         make_and(make_review_text("dinner", "dinner"),
                  make_hours("friday", "Friday", "21:0-22:0"))),
        # Note: removed romantic ambience (no restaurants have it)
        (make_item_meta_contains("casual", "Ambience", "'casual': True"),
         make_review_text("wine", "wine"),
         make_and(make_item_meta("quiet", "NoiseLevel", "u'quiet'"),
                  make_review_text("dinner", "dinner"))),
        (make_item_meta("bike_parking", "BikeParking", "True"),
         make_review_text("coffee", "coffee"),
         make_and(make_item_meta("free_wifi", "WiFi", "u'free'"),
                  make_hours("monday", "Monday", "7:0-8:0"))),
        (make_item_meta("wheelchair", "WheelchairAccessible", "True"),
         make_item_meta("good_for_kids", "GoodForKids", "True"),
         make_and(make_review_text("breakfast", "breakfast"),
                  make_review_text("friendly", "friendly"))),
    ]

    for a, b, cd in combos:
        # AND(A, OR(B, AND(C,D)))
        inner_or = make_or(b, cd)
        structures.append(make_and(a, inner_or))

    return structures


# ===== Main Generation =====

def generate_requests(group: str, restaurants: List[dict], reviews_by_biz: dict) -> List[dict]:
    """Generate requests for a group."""
    generators = {
        "G05": (generate_g05_structures, 40, "Triple OR"),
        "G06": (generate_g06_structures, 50, "Nested OR+AND"),
        "G07": (generate_g07_structures, 60, "Chained OR"),
        "G08": (generate_g08_structures, 70, "Unbalanced"),
    }

    if group not in generators:
        return []

    gen_func, start_id, desc = generators[group]
    structures = gen_func()

    requests = []
    for i, structure in enumerate(structures):
        rid = f"R{start_id + i:02d}"

        # Find matching restaurants
        matching = find_matching_restaurants(restaurants, reviews_by_biz, structure)

        if not matching:
            print(f"  {rid}: NO MATCHES - skipping")
            continue

        # Pick gold restaurant (first match for determinism)
        gold_idx = matching[0]
        gold_id = restaurants[gold_idx]["business_id"]
        gold_name = restaurants[gold_idx].get("name", "Unknown")

        # Render text
        text = render_structure_text(structure)

        req = {
            "id": rid,
            "group": group,
            "scenario": desc,
            "text": text,
            "structure": structure,
            "gold_restaurant": gold_id
        }

        requests.append(req)
        print(f"  {rid}: {len(matching)} matches -> {gold_name} ({gold_id[:8]}...)")

    return requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="G05", help="Group to generate: G05, G06, G07, G08, or all")
    parser.add_argument("--dry-run", action="store_true", help="Show output without writing")
    args = parser.parse_args()

    print("Loading data...")
    restaurants, reviews_by_biz = load_data()
    print(f"  {len(restaurants)} restaurants, {len(reviews_by_biz)} with reviews")

    groups = ["G05", "G06", "G07", "G08"] if args.group.lower() == "all" else [args.group.upper()]

    # Load existing requests (keep G01-G04)
    requests_file = DATA_DIR / "requests.jsonl"
    with open(requests_file) as f:
        existing = [json.loads(l) for l in f if l.strip()]

    # Keep only G01-G04 (R00-R39)
    kept = [r for r in existing if int(r["id"][1:]) < 40]
    print(f"\nKeeping {len(kept)} existing requests (R00-R39)")

    all_new = []
    for group in groups:
        print(f"\n=== Generating {group} ===")
        new_requests = generate_requests(group, restaurants, reviews_by_biz)
        all_new.extend(new_requests)
        print(f"  Generated {len(new_requests)} requests")

    # Combine
    final = kept + all_new
    final.sort(key=lambda r: int(r["id"][1:]))

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(final)} requests")
        print("\nSample new requests:")
        for r in all_new[:3]:
            print(f"  {r['id']}: {r['text'][:60]}...")
        return

    # Write
    with open(requests_file, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(final)} requests to {requests_file}")


if __name__ == "__main__":
    main()
