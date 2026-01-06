#!/usr/bin/env python3
"""Fix request text issues for G01-G05.

Usage:
    python data/scripts/fix_g01_requests.py --group G01 --dry-run
    python data/scripts/fix_g01_requests.py --group G02
    python data/scripts/fix_g01_requests.py --group all
"""

import json
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "philly_cafes"

# Mapping from condition to natural text
CONDITION_TEXT = {
    # Boolean attributes
    ("attributes", "DriveThru", "True"): "with a drive-thru",
    ("attributes", "DriveThru", "False"): "without a drive-thru",
    ("attributes", "GoodForKids", "True"): "kid-friendly",
    ("attributes", "GoodForKids", "False"): "not aimed at kids",
    ("attributes", "HasTV", "True"): "with TVs",
    ("attributes", "HasTV", "False"): "without TVs",
    ("attributes", "DogsAllowed", "True"): "dog-friendly",
    ("attributes", "DogsAllowed", "False"): "no dogs allowed",
    ("attributes", "BikeParking", "True"): "with bike parking",
    ("attributes", "BikeParking", "False"): "without bike parking",
    ("attributes", "OutdoorSeating", "True"): "with outdoor seating",
    ("attributes", "OutdoorSeating", "False"): "indoor-only",
    ("attributes", "RestaurantsReservations", "True"): "takes reservations",
    ("attributes", "RestaurantsReservations", "False"): "no reservations needed",
    ("attributes", "RestaurantsGoodForGroups", "True"): "good for groups",
    ("attributes", "RestaurantsGoodForGroups", "False"): "not suited for groups",
    ("attributes", "RestaurantsTakeOut", "True"): "with takeout",
    ("attributes", "RestaurantsTakeOut", "False"): "no takeout",
    ("attributes", "RestaurantsDelivery", "True"): "offers delivery",
    ("attributes", "RestaurantsDelivery", "False"): "no delivery",
    ("attributes", "CoatCheck", "True"): "with coat check",
    ("attributes", "CoatCheck", "False"): "no coat check",
    ("attributes", "BYOB", "True"): "BYOB",
    ("attributes", "BYOB", "False"): "not BYOB",
    ("attributes", "HappyHour", "True"): "with happy hour",
    ("attributes", "HappyHour", "False"): "no happy hour",
    ("attributes", "BusinessAcceptsCreditCards", "True"): "accepts credit cards",
    ("attributes", "BusinessAcceptsCreditCards", "False"): "cash only",
    ("attributes", "WheelchairAccessible", "True"): "wheelchair accessible",
    ("attributes", "WheelchairAccessible", "False"): "not wheelchair accessible",
    # WiFi
    ("attributes", "WiFi", "u'free'"): "with free WiFi",
    ("attributes", "WiFi", "u'no'"): "without WiFi",
    ("attributes", "WiFi", "u'paid'"): "with paid WiFi",
    # Noise
    ("attributes", "NoiseLevel", "u'quiet'"): "quiet",
    ("attributes", "NoiseLevel", "u'average'"): "with moderate noise",
    ("attributes", "NoiseLevel", "u'loud'"): "lively",
    ("attributes", "NoiseLevel", "u'very_loud'"): "very lively",
    # Alcohol (handle both u'...' and '...' formats)
    ("attributes", "Alcohol", "u'full_bar'"): "with a full bar",
    ("attributes", "Alcohol", "'full_bar'"): "with a full bar",
    ("attributes", "Alcohol", "u'beer_and_wine'"): "with beer and wine",
    ("attributes", "Alcohol", "'beer_and_wine'"): "with beer and wine",
    ("attributes", "Alcohol", "u'none'"): "no alcohol",
    ("attributes", "Alcohol", "'none'"): "no alcohol",
    # Price
    ("attributes", "RestaurantsPriceRange2", "1"): "budget-friendly",
    ("attributes", "RestaurantsPriceRange2", "2"): "mid-priced",
    ("attributes", "RestaurantsPriceRange2", "3"): "upscale",
    ("attributes", "RestaurantsPriceRange2", "4"): "high-end",
    # Attire (handle both u'...' and '...' formats)
    ("attributes", "RestaurantsAttire", "u'casual'"): "with casual dress code",
    ("attributes", "RestaurantsAttire", "'casual'"): "with casual dress code",
    ("attributes", "RestaurantsAttire", "u'dressy'"): "with dressy attire",
    ("attributes", "RestaurantsAttire", "'dressy'"): "with dressy attire",
    ("attributes", "RestaurantsAttire", "u'formal'"): "with formal attire",
    ("attributes", "RestaurantsAttire", "'formal'"): "with formal attire",
}

# Ambience patterns
AMBIENCE_TEXT = {
    "'hipster': True": "with a hipster vibe",
    "'trendy': True": "trendy",
    "'casual': True": "with a casual atmosphere",
    "'romantic': True": "with a romantic atmosphere",
    "'intimate': True": "with an intimate setting",
    "'classy': True": "with a classy atmosphere",
    "'upscale': True": "upscale",
    "'divey': True": "with a divey atmosphere",
    "'touristy': True": "touristy",
}

# Meal patterns
MEAL_TEXT = {
    "'breakfast': True": "good for breakfast",
    "'brunch': True": "good for brunch",
    "'lunch': True": "good for lunch",
    "'dinner': True": "good for dinner",
    "'latenight': True": "good for late night",
    "'dessert': True": "good for dessert",
}


def negate_phrase(phrase: str) -> str:
    """Negate a positive phrase."""
    # Handle specific cases
    negations = {
        "with outdoor seating": "without outdoor seating",
        "with a drive-thru": "without a drive-thru",
        "kid-friendly": "not kid-friendly",
        "dog-friendly": "no dogs allowed",
        "with TVs": "without TVs",
        "with bike parking": "without bike parking",
        "with coat check": "without coat check",
        "with takeout": "without takeout",
        "offers delivery": "no delivery",
        "takes reservations": "no reservations",
        "good for groups": "not good for groups",
        "BYOB": "not BYOB",
        "with free WiFi": "without WiFi",
        "with a full bar": "no alcohol",
        "with beer and wine": "no alcohol",
        "wheelchair accessible": "not wheelchair accessible",
        "accepts credit cards": "cash only",
    }
    if phrase in negations:
        return negations[phrase]
    # Generic negation
    if phrase.startswith("with "):
        return "without " + phrase[5:]
    return f"not {phrase}"


def format_time_range(time_range: str) -> str:
    """Format time range like '7:0-8:0' to 'from 7:00 AM to 8:00 AM'."""
    if not time_range or '-' not in time_range:
        return time_range

    def format_hour(h_str: str) -> str:
        """Format single time like '7:0' to '7:00 AM'."""
        parts = h_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0

        if hour == 0:
            return f"12:{minute:02d} AM"
        elif hour < 12:
            return f"{hour}:{minute:02d} AM"
        elif hour == 12:
            return f"12:{minute:02d} PM"
        else:
            return f"{hour-12}:{minute:02d} PM"

    try:
        start, end = time_range.split('-')
        return f"from {format_hour(start)} to {format_hour(end)}"
    except:
        return time_range


def condition_to_phrase(cond: dict) -> tuple:
    """Convert condition to natural phrase.

    Returns (phrase, phrase_type) where phrase_type is:
    - 'adj': adjective (use "that's X")
    - 'with': starts with "with"
    - 'where': review-based condition (use "where X")
    - 'other': other phrases
    """
    # Handle OR conditions at the top level
    if cond.get("op") == "OR":
        args = cond.get("args", [])
        if not args:
            return ("[empty OR]", 'other')

        # Check if all branches are review_text patterns
        all_review_text = all(
            arg.get("evidence", {}).get("kind") == "review_text"
            for arg in args
        )

        if all_review_text:
            # Compact form: "reviews mention 'X' or 'Y'"
            patterns = [f"'{arg['evidence']['pattern']}'" for arg in args]
            if len(patterns) == 2:
                joined = f"reviews mention {patterns[0]} or {patterns[1]}"
            else:
                joined = "reviews mention " + ", ".join(patterns[:-1]) + f", or {patterns[-1]}"
            return (joined, 'where')

        # Mixed types - get phrases for each branch
        or_phrases = []
        has_review = False
        for arg in args:
            phrase, ptype = condition_to_phrase(arg)
            if ptype == 'where':
                has_review = True
                # For mixed OR, prefix review conditions
                phrase = f"reviews mention '{arg['evidence']['pattern']}'"
            or_phrases.append(phrase)

        # Join with "or"
        if len(or_phrases) == 2:
            joined = f"{or_phrases[0]} or {or_phrases[1]}"
        else:
            joined = ", ".join(or_phrases[:-1]) + f", or {or_phrases[-1]}"

        # Mixed OR goes in 'other' category
        return (joined, 'other')

    ev = cond.get("evidence", {})
    kind = ev.get("kind")

    if kind == "item_meta":
        path = tuple(ev.get("path", []))

        if "true" in ev:
            key = (*path, ev["true"])
            if key in CONDITION_TEXT:
                phrase = CONDITION_TEXT[key]
                ptype = 'with' if phrase.startswith('with ') else 'adj' if not any(phrase.startswith(x) for x in ['no ', 'without ', 'indoor', 'offers', 'takes', 'good for', 'accepts']) else 'other'
                return (phrase, ptype)
            # Fallback
            return (f"{path[-1]}={ev['true']}", 'other')

        # Handle "not_true" - negated conditions
        if "not_true" in ev:
            # Look up what the positive condition would be, then negate it
            positive_key = (*path, ev["not_true"])
            if positive_key in CONDITION_TEXT:
                positive_phrase = CONDITION_TEXT[positive_key]
                # Negate the phrase
                negated = negate_phrase(positive_phrase)
                ptype = 'other'  # negations go in "other" category
                return (negated, ptype)
            # Fallback for unknown negated conditions
            attr = path[-1] if path else "unknown"
            return (f"without {attr}", 'other')

        if "contains" in ev:
            contains = ev["contains"]
            if "Ambience" in path and contains in AMBIENCE_TEXT:
                phrase = AMBIENCE_TEXT[contains]
                ptype = 'with' if phrase.startswith('with ') else 'adj'
                return (phrase, ptype)
            if "GoodForMeal" in path and contains in MEAL_TEXT:
                return (MEAL_TEXT[contains], 'other')
            return (f"{path[-1]} contains {contains}", 'other')

    elif kind == "review_text":
        pattern = ev.get("pattern", "")
        weight_by = ev.get("weight_by", {})

        # Build reviewer description based on weight_by
        if weight_by:
            field = weight_by.get("field", [])
            if field == ["user", "review_count"]:
                return (f"experienced reviewers mention '{pattern}'", 'where')
            elif field == ["user", "fans"]:
                return (f"popular reviewers mention '{pattern}'", 'where')
            elif field == ["user", "elite"]:
                return (f"elite reviewers mention '{pattern}'", 'where')
            elif field == ["useful"]:
                return (f"helpful reviews mention '{pattern}'", 'where')
            elif field == ["user", "friends"]:
                return (f"well-connected reviewers mention '{pattern}'", 'where')

        return (f"reviews mention '{pattern}'", 'where')

    elif kind == "item_meta_hours":
        # Day is in path[1], time range is in "true"
        path = ev.get("path", [])
        day = path[1] if len(path) > 1 else ""
        time_range = ev.get("true", "")
        # Parse time range like "7:0-8:0" -> "7:00 AM - 8:00 AM"
        time_str = format_time_range(time_range)
        return (f"open on {day} {time_str}", 'other')

    return (f"[unknown: {cond}]", 'other')


def remove_duplicates(args: list) -> list:
    """Remove duplicate conditions from args list."""
    seen = {}
    result = []
    for arg in args:
        # Handle OR conditions (no "evidence" key)
        if arg.get("op") == "OR":
            # Use string representation as key for OR conditions
            key = ("OR", str(arg.get("args", [])))
        else:
            ev = arg.get("evidence", {})
            # Key by kind + path + value
            kind = ev.get("kind", "")
            if kind == "review_text":
                key = ("review_text", ev.get("pattern", ""), str(ev.get("weight_by", "")))
            else:
                key = (kind, tuple(ev.get("path", [])), ev.get("true", ev.get("contains", ev.get("not_true", ""))))

        if key not in seen:
            seen[key] = arg.get("aspect", "OR")
            result.append(arg)
    return result


def render_natural_text(conditions: list) -> str:
    """Render conditions as natural text."""
    # Get (phrase, type) tuples
    phrase_data = [condition_to_phrase(c) for c in conditions]

    if not phrase_data:
        return "Looking for a cafe"

    # Classify by phrase type
    adjectives = []     # "that's X"
    with_phrases = []   # "with X"
    where_phrases = []  # "where reviews mention X"
    other = []          # "good for X", "takes reservations", etc.

    for phrase, ptype in phrase_data:
        if ptype == 'adj':
            adjectives.append(phrase)
        elif ptype == 'with':
            with_phrases.append(phrase)
        elif ptype == 'where':
            where_phrases.append(phrase)
        else:
            other.append(phrase)

    # Build sentence parts
    parts = []

    # Adjectives first (comma-separated, use "that's")
    if adjectives:
        if len(adjectives) == 1:
            parts.append(f"that's {adjectives[0]}")
        else:
            parts.append("that's " + ", ".join(adjectives))

    # With-phrases
    for w in with_phrases:
        parts.append(w)

    # Where-phrases (review mentions)
    for w in where_phrases:
        parts.append(f"where {w}")

    # Other phrases
    for o in other:
        parts.append(o)

    # Join all parts
    if len(parts) == 0:
        return "Looking for a cafe"
    elif len(parts) == 1:
        return f"Looking for a cafe {parts[0]}"
    elif len(parts) == 2:
        return f"Looking for a cafe {parts[0]}, {parts[1]}"
    else:
        # Join with commas and "and" before last
        main = ", ".join(parts[:-1])
        return f"Looking for a cafe {main}, and {parts[-1]}"


def fix_request(req: dict) -> dict:
    """Fix a single request."""
    structure = req.get("structure", {})
    args = structure.get("args", [])

    # Remove duplicates
    args = remove_duplicates(args)
    structure["args"] = args

    # Render new text
    new_text = render_natural_text(args)

    new_req = req.copy()
    new_req["structure"] = structure
    new_req["text"] = new_text

    return new_req


GROUP_RANGES = {
    "G01": (0, 10),
    "G02": (10, 20),
    "G03": (20, 30),
    "G04": (30, 40),
    "G05": (40, 50),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--group", default="G01", help="Group to fix: G01, G02, G03, G04, G05, or all")
    args = parser.parse_args()

    # Load requests
    requests_file = DATA_DIR / "requests.jsonl"
    with open(requests_file) as f:
        requests = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(requests)} requests")

    # Determine range
    if args.group.lower() == "all":
        start, end = 0, 50
        print(f"Fixing all groups (R00-R49)...\n")
    else:
        group = args.group.upper()
        if group not in GROUP_RANGES:
            print(f"Unknown group: {group}. Use G01-G05 or 'all'")
            return
        start, end = GROUP_RANGES[group]
        print(f"Fixing {group} (R{start:02d}-R{end-1:02d})...\n")

    # Process specified range
    for i in range(start, end):
        old_text = requests[i]["text"]
        requests[i] = fix_request(requests[i])
        new_text = requests[i]["text"]

        if old_text != new_text:
            print(f"R{i:02d}:")
            print(f"  OLD: {old_text}")
            print(f"  NEW: {new_text}")
            print()

    if args.dry_run:
        print("[DRY RUN] No changes written")
        return

    # Write back
    with open(requests_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Wrote {len(requests)} requests to {requests_file}")


if __name__ == "__main__":
    main()
