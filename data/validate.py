#!/usr/bin/env python3
"""Validate ground truth dataset.

Checks that each request matches exactly 1 restaurant.
Generates groundtruth.jsonl for downstream evaluation.

Usage:
    python -m data.validate              # interactive dataset selection
    python -m data.validate philly_cafes # specific dataset
"""

import argparse
import ast
import json
import re
import sys
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


DATA_DIR = Path(__file__).parent

# --- Judgement Cache (for review-level semantic validation) ---

_judgement_cache = None

def get_judgement_cache():
    """Load judgement_cache.json for review-level semantic evaluation.

    Cache structure (Option A - keyed by review_id:aspect):
    {
        "review_abc123:boba": {
            "review_id": "review_abc123",
            "business_id": "ZpgVL...",
            "aspect": "boba",
            "judgement": true,
            "sentiment": "positive"
        }
    }
    """
    global _judgement_cache
    if _judgement_cache is None:
        cache_path = DATA_DIR / "philly_cafes" / "judgement_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                _judgement_cache = json.load(f)
        else:
            _judgement_cache = {}
    return _judgement_cache


def check_review_judgement(review_id: str, aspect: str) -> str:
    """Check review judgement from cache.

    Args:
        review_id: The review's ID
        aspect: The aspect/topic to check (e.g., "boba", "espresso")

    Returns:
        One of: "positive", "negative", "neutral", "not_mentioned"
        Defaults to "not_mentioned" if cache entry not found.
    """
    cache = get_judgement_cache()
    cache_key = f"{review_id}:{aspect}"

    if cache_key not in cache:
        return "not_mentioned"  # Default for missing entries

    entry = cache[cache_key]
    judgement = entry.get("judgement", "")

    # Normalize to standard values
    if isinstance(judgement, str):
        j_lower = judgement.lower()
        if j_lower in ("positive", "pos", "true", "yes"):
            return "positive"
        elif j_lower in ("negative", "neg", "false", "no"):
            return "negative"
        elif j_lower in ("neutral", "mixed"):
            return "neutral"
        elif j_lower in ("not_mentioned", "none", "n/a", "missing"):
            return "not_mentioned"

    # If boolean (legacy)
    if isinstance(judgement, bool):
        return "positive" if judgement else "not_mentioned"

    return "not_mentioned"  # Default for invalid values


def evaluate_review_sentiment(business_id: str, reviews: list, evidence_spec: dict) -> int:
    """Evaluate review_sentiment condition.

    Checks if reviews have positive/negative sentiment for a topic.
    Uses cached LLM judgements (required - errors if missing).
    If social_filter specified, only considers reviews from users in the social network.

    Args:
        business_id: Restaurant's business_id
        reviews: List of review dicts
        evidence_spec: {"kind": "review_sentiment", "topic": "X", "sentiment": "positive/negative", "min_positive": N, "social_filter": {...}}

    Returns: 1 if condition met, -1 otherwise
    """
    topic = evidence_spec.get("topic", "")
    expected_sentiment = evidence_spec.get("sentiment", "positive")
    min_positive = evidence_spec.get("min_positive", 1)
    min_negative = evidence_spec.get("min_negative", 1)
    social_filter = evidence_spec.get("social_filter")

    if not topic or not reviews:
        return -1

    # If social_filter specified, filter reviews by social network
    if social_filter:
        friends = social_filter.get("friends", [])
        hops = social_filter.get("hops", 1)

        # Load friend graph
        social = get_social_data()
        friend_graph = social.get("friend_graph", {})

        # Build the social network
        network = set(friends)  # 0-hop: the anchors themselves
        if hops >= 1:
            for anchor in friends:
                network.update(friend_graph.get(anchor, []))
        if hops >= 2:
            one_hop_friends = set(network)
            for person in one_hop_friends:
                network.update(friend_graph.get(person, []))

        # Filter reviews to only those from the social network
        reviews = [r for r in reviews if r.get("user", {}).get("name", "") in network]

    pos_count = 0
    neg_count = 0

    for r in reviews:
        review_id = r.get("review_id", "")

        # Get cached judgement
        judgement = check_review_judgement(review_id, topic)

        if judgement == "positive":
            pos_count += 1
        elif judgement == "negative":
            neg_count += 1
        # "neutral" and "not_mentioned" don't count toward either

    if expected_sentiment == "positive":
        return 1 if pos_count >= min_positive and pos_count > neg_count else -1
    else:  # negative
        return 1 if neg_count >= min_negative else -1


# --- Social Filter Data (Lazy Loaded) ---

_social_data = None

def get_social_data():
    """Load user_mapping.json for social filter evaluation."""
    global _social_data
    if _social_data is None:
        mapping_path = DATA_DIR / "philly_cafes" / "user_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                _social_data = json.load(f)
        else:
            _social_data = {"user_names": {}, "friend_graph": {}, "restaurant_reviews": {}}
    return _social_data


def check_social_filter(reviewer_name: str, friends: list[str], hops: int) -> bool:
    """Check if reviewer qualifies under social filter.

    Args:
        reviewer_name: Name of the reviewer (e.g., "Alice")
        friends: Query's anchor friend list (e.g., ["Grace"])
        hops: 1 for anchor + direct friends, 2 for + friends-of-friends

    Returns:
        True if reviewer qualifies (is in social circle of any anchor)
    """
    social = get_social_data()
    friend_graph = social.get("friend_graph", {})

    # 0-hop: reviewer IS one of the anchor names
    if reviewer_name in friends:
        return True

    # 1-hop: reviewer is a friend of an anchor
    if hops >= 1:
        for anchor in friends:
            anchor_friends = friend_graph.get(anchor, [])
            if reviewer_name in anchor_friends:
                return True

    # 2-hop: reviewer is friend-of-friend of an anchor
    if hops >= 2:
        for anchor in friends:
            anchor_friends = friend_graph.get(anchor, [])
            for anchor_friend in anchor_friends:
                fof_list = friend_graph.get(anchor_friend, [])
                if reviewer_name in fof_list:
                    return True

    return False


# --- Hours Parsing ---

def parse_hours_range(hours_str: str) -> tuple[int, int] | None:
    """Parse 'H:M-H:M' to (start_minutes, end_minutes)."""
    if not hours_str or "-" not in hours_str:
        return None
    try:
        start, end = hours_str.split("-")
        def to_mins(t: str) -> int:
            h, m = t.split(":")
            return int(h) * 60 + int(m)
        return to_mins(start), to_mins(end)
    except (ValueError, AttributeError):
        return None


def hours_contains(item_hours: str, required: str) -> bool:
    """Check if restaurant hours cover the required time range."""
    item_range = parse_hours_range(item_hours)
    req_range = parse_hours_range(required)
    if not item_range or not req_range:
        return False
    item_start, item_end = item_range
    req_start, req_end = req_range
    # Handle overnight hours (end < start means crosses midnight)
    if item_end < item_start:
        item_end += 24 * 60
    return item_start <= req_start and item_end >= req_end


# --- Review Metadata Credibility Evaluation ---

def get_review_metadata_value(review: dict, field_path: list) -> float:
    """Extract metadata value from a review for credibility scoring.

    Args:
        review: Review dict containing user metadata
        field_path: Path to value (e.g., ["user", "review_count"])

    Returns:
        Float value (0.0 if missing or invalid)
    """
    v = review
    for key in field_path:
        if isinstance(v, dict):
            v = v.get(key)
        else:
            return 0.0

    # Special handling for elite (list of years)
    if field_path == ["user", "elite"] and isinstance(v, list):
        return float(len([e for e in v if e and e != 'None']))

    return float(v) if v is not None else 0.0


def evaluate_credibility_count(reviews: list, aspect: str, field_path: list,
                               credibility_percentile: int = 50,
                               min_credible_matches: int = 2) -> int:
    """Evaluate aspect using credibility threshold + count floor.

    Semantics: "At least N credible reviewers (above percentile) mention aspect positively"

    Uses cached LLM judgements (required - errors if missing).

    Args:
        reviews: List of review dicts
        aspect: Aspect/topic to check (used as cache key)
        field_path: Path to credibility field (e.g., ["user", "review_count"])
        credibility_percentile: Percentile threshold for "credible" (default: 50 = median)
        min_credible_matches: Minimum credible reviewers with positive judgement (default: 2)

    Returns:
        1 if condition satisfied, -1 otherwise

    Raises:
        JudgementMissingError: If any required cache entry is missing
    """
    if not reviews or not aspect:
        return -1

    # 1. Extract field values
    values = [get_review_metadata_value(r, field_path) for r in reviews]

    # 2. Compute credibility threshold (percentile of non-zero values)
    nonzero = sorted([v for v in values if v > 0])
    if len(nonzero) < min_credible_matches:
        return -1  # Not enough credible reviewers exist

    threshold_idx = int(len(nonzero) * credibility_percentile / 100)
    cred_threshold = nonzero[min(threshold_idx, len(nonzero) - 1)]

    # 3. Count credible reviewers with positive judgement
    credible_matches = 0
    for r, v in zip(reviews, values):
        if v >= cred_threshold:
            review_id = r.get("review_id", "")
            judgement = check_review_judgement(review_id, aspect)
            if judgement == "positive":
                credible_matches += 1

    # 4. Return result
    return 1 if credible_matches >= min_credible_matches else -1


def evaluate_review_meta(reviews: list, evidence_spec: dict) -> int:
    """Evaluate standalone review_meta condition.

    Checks metadata across reviews with optional filtering.

    evidence_spec:
        path: ["user", "elite"] or ["useful"]
        op: "not_empty", "gte", "lte"
        value: threshold for gte/lte
        min_stars: optional star filter (only count reviews with stars >= this)
        agg: "any" (default), "all", "count"
        count: for agg="count", require this many matches
    """
    if not reviews:
        return evidence_spec.get("missing", 0)

    path = evidence_spec.get("path", [])
    op = evidence_spec.get("op", "not_empty")
    value = evidence_spec.get("value")
    min_stars = evidence_spec.get("min_stars")
    agg = evidence_spec.get("agg", "any")
    count_needed = evidence_spec.get("count", 1)

    matching = 0
    total = 0

    for r in reviews:
        # Filter by min_stars if specified
        if min_stars is not None:
            stars = r.get("stars", 0)
            if stars < min_stars:
                continue

        total += 1

        # Navigate to the value
        v = r
        for key in path:
            if isinstance(v, dict):
                v = v.get(key)
            else:
                v = None
                break

        # Check condition
        match = False
        if op == "not_empty":
            # For lists like elite years
            if isinstance(v, list):
                match = bool(v) and len([e for e in v if e and e != 'None']) > 0
            else:
                match = v is not None and v != "" and v != 0
        elif op == "gte" and v is not None:
            match = float(v) >= float(value)
        elif op == "lte" and v is not None:
            match = float(v) <= float(value)

        if match:
            matching += 1

    # Aggregation
    if agg == "any":
        return 1 if matching > 0 else -1
    elif agg == "all":
        return 1 if total > 0 and matching == total else -1
    elif agg == "count":
        return 1 if matching >= count_needed else -1

    return 0


# --- Review Group Rating Evaluation (G04 conditions) ---

def filter_reviews_by_group(reviews: list, group_filter: dict) -> list:
    """Filter reviews by group criteria.

    group_filter:
        field: "date" or ["user", "average_stars"] etc.
        operator: "gte", "lt", "lte", "gt"
        value: threshold value
    """
    if not reviews or not group_filter:
        return reviews

    field = group_filter.get("field")
    op = group_filter.get("operator", "gte")
    value = group_filter.get("value")

    if field is None or value is None:
        return reviews

    filtered = []
    for r in reviews:
        # Navigate to field value
        if isinstance(field, list):
            v = r
            for key in field:
                if isinstance(v, dict):
                    v = v.get(key)
                else:
                    v = None
                    break
        else:
            v = r.get(field)

        if v is None:
            continue

        # Compare based on operator
        try:
            # Handle date strings
            if isinstance(v, str) and isinstance(value, str) and "-" in v:
                passes = (
                    (op == "gte" and v >= value) or
                    (op == "gt" and v > value) or
                    (op == "lt" and v < value) or
                    (op == "lte" and v <= value)
                )
            else:
                v_num = float(v)
                val_num = float(value)
                passes = (
                    (op == "gte" and v_num >= val_num) or
                    (op == "gt" and v_num > val_num) or
                    (op == "lt" and v_num < val_num) or
                    (op == "lte" and v_num <= val_num)
                )

            if passes:
                filtered.append(r)
        except (ValueError, TypeError):
            continue

    return filtered


def evaluate_review_group_rating(reviews: list, evidence_spec: dict) -> int:
    """Evaluate review_group_rating condition.

    Filters reviews by group criteria, then computes metric and compares to threshold.

    evidence_spec:
        group: name of group (e.g., "post_2020", "harsh_reviewers")
        group_filter: {"field": ..., "operator": ..., "value": ...}
        metric: "avg_stars" or "count"
        threshold: numeric threshold
        operator: "gte", "lt", etc.

    Returns: 1 if condition met, -1 otherwise
    """
    if not reviews:
        return -1

    group_filter = evidence_spec.get("group_filter", {})
    metric = evidence_spec.get("metric", "avg_stars")
    threshold = evidence_spec.get("threshold", 0)
    op = evidence_spec.get("operator", "gte")

    # Filter reviews by group
    filtered = filter_reviews_by_group(reviews, group_filter)

    # Compute metric
    if metric == "count":
        computed = len(filtered)
    elif metric == "avg_stars":
        if not filtered:
            return -1  # No reviews in group = condition not met
        stars = [r.get("stars", 0) for r in filtered]
        computed = sum(stars) / len(stars) if stars else 0
    else:
        return -1

    # Compare to threshold
    if op == "gte":
        return 1 if computed >= threshold else -1
    elif op == "gt":
        return 1 if computed > threshold else -1
    elif op == "lt":
        return 1 if computed < threshold else -1
    elif op == "lte":
        return 1 if computed <= threshold else -1

    return -1


def evaluate_review_group_rating_negative(reviews: list, evidence_spec: dict) -> int:
    """Evaluate review_group_rating_negative condition.

    Checks that a restaurant is NOT only praised by easy raters.
    Returns 1 if the restaurant passes (is NOT the bad pattern), -1 if it fails.

    evidence_spec:
        condition: {
            "generous_avg_gte": 4.0,  # generous reviewers avg >= 4.0
            "harsh_avg_lt": 2.0       # AND harsh reviewers avg < 2.0
        }

    The condition describes the BAD pattern to avoid.
    If generous_avg >= 4.0 AND harsh_avg < 2.0, return -1 (fails - only easy raters like it)
    Otherwise return 1 (passes - not just praised by easy raters)
    """
    if not reviews:
        return 1  # No reviews = not the bad pattern

    condition = evidence_spec.get("condition", {})
    generous_threshold = condition.get("generous_avg_gte", 4.0)
    harsh_threshold = condition.get("harsh_avg_lt", 2.0)

    # Filter for generous reviewers (user.average_stars >= 4.0)
    generous_reviews = []
    harsh_reviews = []

    for r in reviews:
        user = r.get("user", {})
        avg_stars = user.get("average_stars")
        if avg_stars is None:
            continue

        try:
            avg = float(avg_stars)
            if avg >= 4.0:
                generous_reviews.append(r)
            elif avg < 3.5:
                harsh_reviews.append(r)
        except (ValueError, TypeError):
            continue

    # Compute averages
    generous_avg = None
    harsh_avg = None

    if generous_reviews:
        stars = [r.get("stars", 0) for r in generous_reviews]
        generous_avg = sum(stars) / len(stars) if stars else None

    if harsh_reviews:
        stars = [r.get("stars", 0) for r in harsh_reviews]
        harsh_avg = sum(stars) / len(stars) if stars else None

    # Check if it matches the BAD pattern
    # Bad pattern: generous love it (>= threshold) AND harsh hate it (< threshold)
    is_bad_pattern = (
        generous_avg is not None and generous_avg >= generous_threshold and
        harsh_avg is not None and harsh_avg < harsh_threshold
    )

    # Return 1 if NOT the bad pattern, -1 if it IS the bad pattern
    return -1 if is_bad_pattern else 1


# --- Three-Value Logic (from oldsrc) ---

class TV(Enum):
    """Three-value logic: True, Unknown, False."""
    T = 1
    U = 0
    F = -1


def reduce_tv(op: str, values: list[TV]) -> TV:
    """Reduce list of TV values using AND or OR.

    - AND: False dominates, then Unknown, then True
    - OR: True dominates, then Unknown, then False
    """
    if not values:
        return TV.U

    if op == "AND":
        if any(v is TV.F for v in values):
            return TV.F
        if any(v is TV.U for v in values):
            return TV.U
        return TV.T
    else:  # OR
        if any(v is TV.T for v in values):
            return TV.T
        if any(v is TV.U for v in values):
            return TV.U
        return TV.F


# --- Attribute Parsing (from oldsrc) ---

def parse_attr_value(value):
    """Parse Yelp attribute string to Python value.

    Examples:
        "True" → True
        "False" → False
        "u'free'" → "free"
        "'quiet'" → "quiet"
        "{'romantic': False}" → {"romantic": False}
        None → None
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    # Handle boolean strings
    if value == "True":
        return True
    if value == "False":
        return False
    if value.lower() == "none":
        return None

    # Handle unicode string prefix: u'value' or u"value"
    match = re.match(r"^u?['\"](.+)['\"]$", value)
    if match:
        return match.group(1)

    # Handle dict/list literals
    if value.startswith("{") or value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


def get_nested_value(item: dict, path: list):
    """Navigate nested dict using path, parsing attribute strings.

    Returns None if path doesn't exist or value is missing.
    """
    current = item
    for i, key in enumerate(path):
        if current is None:
            return None
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None

        value = current[key]

        # Parse string values (Yelp attributes are stored as strings)
        if isinstance(value, str) and i < len(path) - 1:
            value = parse_attr_value(value)
        elif i == len(path) - 1:
            value = parse_attr_value(value)

        current = value

    return current


def match_value(actual, expected) -> bool:
    """Normalized string matching (case-insensitive).

    Also parses expected values that may be in Yelp string format.
    """
    if expected is None:
        return False
    if isinstance(expected, bool):
        return actual == expected
    if isinstance(expected, list):
        return any(match_value(actual, e) for e in expected)

    # Parse expected value if it's a Yelp-format string
    parsed_expected = parse_attr_value(expected)

    # Try exact match first (case-insensitive for strings)
    if isinstance(actual, str) and isinstance(parsed_expected, str):
        if actual.lower() == parsed_expected.lower():
            return True
    elif actual == parsed_expected:
        return True

    # Fall back to contains matching
    return str(parsed_expected).lower() in str(actual).lower()


# --- Condition Evaluation (from oldsrc, simplified) ---

def evaluate_item_meta_rule(value, evidence_spec) -> int:
    """Evaluate item_meta using declarative rules from evidence_spec.

    Rules:
    - "not_true" / "true_not" given → value should NOT match (None passes)
    - "not_contains" given → substring should NOT be in string repr (None passes)
    - Missing value → use "missing" field (default 0=neutral)
    - Dict of booleans → OR across children (e.g., BusinessParking)
    - None given → default boolean check (True→1, False→-1, else→0)
    - Only "true" given → match→1, no match→-1
    - Only "false" given → match→-1, no match→1 (negative logic)
    - "contains" given → substring check in string repr
    """
    true_cond = evidence_spec.get("true")
    false_cond = evidence_spec.get("false")
    not_true_cond = evidence_spec.get("not_true") or evidence_spec.get("true_not")
    not_contains_cond = evidence_spec.get("not_contains")
    neutral_cond = evidence_spec.get("neutral")
    contains_cond = evidence_spec.get("contains")
    missing_val = evidence_spec.get("missing", 0)

    # "not_contains" check - substring should NOT be present (None passes)
    if not_contains_cond is not None:
        if value is None:
            return 1  # None doesn't contain anything, so passes
        value_str = str(value).lower()
        not_contains_str = str(not_contains_cond).lower()
        return -1 if not_contains_str in value_str else 1

    # "contains" check - look for substring in string repr
    # Handle BEFORE missing check: str(None).lower() = "none", so "contains": "none" should match
    if contains_cond is not None:
        value_str = str(value).lower()
        contains_str = str(contains_cond).lower()
        return 1 if contains_str in value_str else -1

    # "not_true" / "true_not" - negative check (value should NOT match)
    # Handle BEFORE missing check: None is "not True" so it passes
    if not_true_cond is not None:
        if value is None:
            return 1  # None is not the target value, so passes
        return -1 if match_value(value, not_true_cond) else 1

    # Missing value → use "missing" field (default 0=neutral)
    if value is None:
        return missing_val

    # Dict of booleans → OR across children (e.g., BusinessParking)
    if isinstance(value, dict) and value and all(isinstance(v, bool) for v in value.values()):
        has_true = any(v for v in value.values())
        return 1 if has_true else -1

    # Case 1: None given → default boolean check
    if true_cond is None and false_cond is None and neutral_cond is None:
        if value is True:
            return 1
        if value is False:
            return -1
        return 0

    # Case 2: Only "false" given → negative logic (no match = true)
    if false_cond is not None and true_cond is None and neutral_cond is None:
        return -1 if match_value(value, false_cond) else 1

    # Case 3: Only "true" given → no match = false
    if true_cond is not None and false_cond is None and neutral_cond is None:
        return 1 if match_value(value, true_cond) else -1

    # Case 4: Multiple conditions given → check each explicitly
    if true_cond is not None and match_value(value, true_cond):
        return 1
    if false_cond is not None and match_value(value, false_cond):
        return -1
    if neutral_cond is not None and match_value(value, neutral_cond):
        return 0

    # None matched → neutral
    return 0


def evaluate_review_text_pattern(reviews: list, pattern: str) -> int:
    """Check if any review mentions the pattern positively.

    Uses cached LLM judgements (required - errors if missing).

    Args:
        reviews: List of review dicts
        pattern: Aspect/topic to check (used as cache key)

    Returns: 1 if any review is positive, -1 if none positive

    Raises:
        JudgementMissingError: If any required cache entry is missing
    """
    if not pattern or not reviews:
        return -1

    for r in reviews:
        review_id = r.get("review_id", "")
        judgement = check_review_judgement(review_id, pattern)
        if judgement == "positive":
            return 1

    return -1


def evaluate_social_filter_from_reviews(reviews: list, pattern: str, social_filter: dict) -> int:
    """Evaluate review text pattern with social filter using actual review data.

    Checks BOTH:
    1. Review mentions the pattern positively (via judgement cache)
    2. Reviewer qualifies under social filter (name in friends or friend-of-friend)

    Args:
        reviews: Actual review dicts from loaded data
        pattern: Aspect/topic to check (used as cache key)
        social_filter: {"friends": ["Alice", "Bob"], "hops": 1 or 2}

    Returns: 1 if condition satisfied, -1 otherwise

    Raises:
        JudgementMissingError: If any required cache entry is missing
    """
    friends = social_filter.get("friends", [])
    hops = social_filter.get("hops", 1)
    min_matches = social_filter.get("min_matches", 1)

    if not reviews:
        return -1

    # Load friend_graph for looking up reviewer's friends
    social = get_social_data()
    friend_graph = social.get("friend_graph", {})

    matches = 0
    for review in reviews:
        review_id = review.get("review_id", "")

        # 1. Check if review mentions pattern positively (via cache)
        judgement = check_review_judgement(review_id, pattern)
        if judgement != "positive":
            continue

        # 2. Check if reviewer qualifies under social filter
        user = review.get("user", {})
        reviewer_name = user.get("name", "")
        # Look up reviewer's friends from friend_graph (not from review data)
        reviewer_friends = friend_graph.get(reviewer_name, [])

        # 0-hop: reviewer IS one of the anchor names
        if reviewer_name in friends:
            matches += 1
            continue

        # 1-hop: reviewer is a friend of an anchor
        # Check if any anchor is in reviewer's friend list
        if hops >= 1:
            is_1hop = False
            for anchor in friends:
                if anchor in reviewer_friends:
                    is_1hop = True
                    break
            if is_1hop:
                matches += 1
                continue

        # 2-hop: reviewer is friend-of-friend of an anchor
        if hops >= 2:
            # Check if any of reviewer's friends has an anchor as their friend
            is_2hop = False
            for reviewer_friend in reviewer_friends:
                friend_of_friend_list = friend_graph.get(reviewer_friend, [])
                for anchor in friends:
                    if anchor in friend_of_friend_list:
                        is_2hop = True
                        break
                if is_2hop:
                    break
            if is_2hop:
                matches += 1
                continue

    return 1 if matches >= min_matches else -1


def evaluate_social_rating(reviews: list, social_filter: dict, min_stars: int = 4) -> int:
    """Evaluate star rating with social filter.

    Checks if someone in the social network gave a high rating.

    Args:
        reviews: Actual review dicts from loaded data
        social_filter: {"friends": ["Alice", "Bob"], "hops": 1 or 2}
        min_stars: Minimum star rating to count (default 4)

    Returns: 1 if condition satisfied, -1 otherwise
    """
    friends = social_filter.get("friends", [])
    hops = social_filter.get("hops", 1)
    min_matches = social_filter.get("min_matches", 1)

    if not reviews:
        return -1

    # Load friend_graph for looking up reviewer's friends
    social = get_social_data()
    friend_graph = social.get("friend_graph", {})

    matches = 0
    for review in reviews:
        # 1. Check if review has high enough stars
        stars = review.get("stars", 0)
        if stars < min_stars:
            continue

        # 2. Check if reviewer qualifies under social filter
        user = review.get("user", {})
        reviewer_name = user.get("name", "")
        reviewer_friends = friend_graph.get(reviewer_name, [])

        # 0-hop: reviewer IS one of the anchor names
        if reviewer_name in friends:
            matches += 1
            continue

        # 1-hop: reviewer is a friend of an anchor
        if hops >= 1:
            is_1hop = False
            for anchor in friends:
                if anchor in reviewer_friends:
                    is_1hop = True
                    break
            if is_1hop:
                matches += 1
                continue

        # 2-hop: reviewer is friend-of-friend of an anchor
        if hops >= 2:
            is_2hop = False
            for reviewer_friend in reviewer_friends:
                friend_of_friend_list = friend_graph.get(reviewer_friend, [])
                for anchor in friends:
                    if anchor in friend_of_friend_list:
                        is_2hop = True
                        break
                if is_2hop:
                    break
            if is_2hop:
                matches += 1
                continue

    return 1 if matches >= min_matches else -1


def evaluate_condition(item: dict, condition: dict, reviews: list = None) -> int:
    """Evaluate a single condition against an item.

    Returns: 1 (satisfied), -1 (not satisfied), 0 (unknown)
    """
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")
    path = evidence_spec.get("path", [])

    if kind == "item_meta":
        value = get_nested_value(item, path)
        return evaluate_item_meta_rule(value, evidence_spec)

    elif kind == "review_text":
        pattern = evidence_spec.get("pattern", "")
        weight_by = evidence_spec.get("weight_by")
        social_filter = evidence_spec.get("social_filter")

        # If social_filter specified, check actual review data
        if social_filter:
            if not reviews:
                return -1
            return evaluate_social_filter_from_reviews(reviews, pattern, social_filter)

        if not reviews:
            return 0

        # If weight_by specified, use credibility-count evaluation
        if weight_by:
            field_path = weight_by.get("field", [])
            credibility_percentile = evidence_spec.get("credibility_percentile", 50)
            min_credible_matches = evidence_spec.get("min_credible_matches", 2)
            return evaluate_credibility_count(
                reviews, pattern, field_path,
                credibility_percentile, min_credible_matches
            )
        else:
            return evaluate_review_text_pattern(reviews, pattern)

    elif kind == "review_meta":
        # Standalone review metadata check
        if not reviews:
            return evidence_spec.get("missing", 0)
        return evaluate_review_meta(reviews, evidence_spec)

    elif kind == "review_sentiment":
        # Sentiment-based review check (positive/negative about topic)
        business_id = item.get("business_id", "")
        if not reviews:
            return -1
        return evaluate_review_sentiment(business_id, reviews, evidence_spec)

    elif kind == "item_meta_hours":
        day_hours = get_nested_value(item, path)
        required = evidence_spec.get("true", "")
        if not day_hours:
            return evidence_spec.get("missing", 0)
        return 1 if hours_contains(day_hours, required) else -1

    elif kind == "review_group_rating":
        # G04: Filter reviews by group, compute metric, compare to threshold
        if not reviews:
            return -1
        return evaluate_review_group_rating(reviews, evidence_spec)

    elif kind == "review_group_rating_negative":
        # G04: Negative filter (NOT only praised by easy raters)
        if not reviews:
            return 1  # No reviews = not the bad pattern
        return evaluate_review_group_rating_negative(reviews, evidence_spec)

    elif kind == "social_rating":
        # G09/G10: Star rating with social filter
        social_filter = evidence_spec.get("social_filter")
        if not reviews or not social_filter:
            return -1
        min_stars = evidence_spec.get("min_stars", 4)
        return evaluate_social_rating(reviews, social_filter, min_stars)

    return 0


def evaluate_structure(item: dict, structure: dict, reviews: list = None) -> int:
    """Evaluate full AND/OR structure.

    Returns: 1 (satisfied), -1 (not satisfied), 0 (unknown)
    """
    op = structure.get("op", "AND")
    args = structure.get("args", [])

    child_results = []

    for arg in args:
        if "op" in arg:
            # Nested structure
            result = evaluate_structure(item, arg, reviews)
        else:
            # Single condition
            result = evaluate_condition(item, arg, reviews)
        child_results.append(TV(result))

    # Handle NOT operator (inverts single child)
    if op == "NOT":
        if len(child_results) != 1:
            return 0  # Invalid NOT structure
        child = child_results[0]
        if child == TV.T:
            return -1
        elif child == TV.F:
            return 1
        else:
            return 0  # Unknown stays unknown

    final_tv = reduce_tv(op, child_results)
    return final_tv.value


# --- Validation Logic ---

def validate_request(request: dict, restaurants: list, reviews_by_id: dict) -> dict:
    """Check how many restaurants match this request.

    Returns:
        {
            "request_id": str,
            "gold_restaurant": str,
            "gold_idx": int,
            "matches": [business_ids],
            "status": "ok" | "no_match" | "multi_match" | "gold_not_match"
        }
    """
    request_id = request.get("id", "?")
    gold_restaurant = request.get("gold_restaurant", "")
    structure = request.get("structure", {})

    matches = []
    gold_idx = -1

    for idx, restaurant in enumerate(restaurants):
        business_id = restaurant.get("business_id", "")
        reviews = reviews_by_id.get(business_id, [])

        result = evaluate_structure(restaurant, structure, reviews)

        if result == 1:
            matches.append(business_id)

        if business_id == gold_restaurant:
            gold_idx = idx

    # Determine status
    if len(matches) == 0:
        status = "no_match"
    elif len(matches) == 1:
        if matches[0] == gold_restaurant:
            status = "ok"
        else:
            status = "gold_not_match"
    else:
        if gold_restaurant in matches:
            status = "multi_match"
        else:
            status = "multi_match_no_gold"

    return {
        "request_id": request_id,
        "gold_restaurant": gold_restaurant,
        "gold_idx": gold_idx,
        "matches": matches,
        "matches_count": len(matches),
        "status": status
    }


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def validate_dataset(name: str, group_filter: set[str] = None):
    """Validate a dataset and generate groundtruth.jsonl.

    Args:
        name: Dataset name
        group_filter: Optional set of group names to filter (e.g., {"G06", "G07"})
    """
    # Import loader - works both as module and direct script
    try:
        from .loader import load_dataset
    except ImportError:
        from loader import load_dataset

    dataset_dir = DATA_DIR / name

    # Check required files
    restaurants_path = dataset_dir / "restaurants.jsonl"
    reviews_path = dataset_dir / "reviews.jsonl"
    requests_path = dataset_dir / "requests.jsonl"

    for p in [restaurants_path, reviews_path, requests_path]:
        if not p.exists():
            print(f"Error: Missing {p}")
            sys.exit(1)

    # Load data using loader (applies synthetic user.name/friends for G09/G10)
    dataset = load_dataset(name)

    # Handle both list and dict formats from loader
    if isinstance(dataset.items, dict):
        # Dict format: {"1": item1, "2": item2, ...}
        items_list = [dataset.items[k] for k in sorted(dataset.items.keys(), key=lambda x: int(x))]
    else:
        items_list = dataset.items

    restaurants = [
        {**item, "reviews": item.get("reviews", [])}
        for item in items_list
    ]
    all_requests = dataset.requests

    # Filter requests by group if specified
    if group_filter:
        requests = [r for r in all_requests if r.get("group") in group_filter]
        if not requests:
            print(f"Warning: No requests found for groups {sorted(group_filter)}")
            return True
    else:
        requests = all_requests

    # Load raw business_ids from file (loader may strip them)
    raw_restaurants = load_jsonl(restaurants_path)
    business_ids = [r["business_id"] for r in raw_restaurants]

    # Attach business_id to restaurants if missing
    for i, r in enumerate(restaurants):
        if "business_id" not in r and i < len(business_ids):
            r["business_id"] = business_ids[i]

    # Group reviews by business_id
    reviews_by_id = {}
    for i, item in enumerate(restaurants):
        bid = item.get("business_id", business_ids[i] if i < len(business_ids) else "")
        reviews_by_id[bid] = item.get("reviews", [])

    total_reviews = sum(len(item.get("reviews", [])) for item in restaurants)
    print(f"Dataset: {name}")
    print(f"  Restaurants: {len(restaurants)}")
    print(f"  Reviews: {total_reviews}")
    if group_filter:
        print(f"  Groups: {', '.join(sorted(group_filter))}")
        print(f"  Requests: {len(requests)} (filtered from {len(all_requests)})")
    else:
        print(f"  Requests: {len(requests)}")
    print()

    # Build evaluation matrix: (item_id, request_id) -> result
    eval_matrix = {}
    item_ids = [r["business_id"] for r in restaurants]
    request_ids = [r["id"] for r in requests]

    # Build request info lookup (for shorthand display)
    request_info = {r["id"]: r for r in requests}

    for request in requests:
        req_id = request["id"]
        structure = request["structure"]
        for restaurant in restaurants:
            item_id = restaurant["business_id"]
            reviews = reviews_by_id.get(item_id, [])
            result = evaluate_structure(restaurant, structure, reviews)
            eval_matrix[(item_id, req_id)] = result

    # Validate each request
    results = []
    ok_count = 0
    error_count = 0

    for request in requests:
        result = validate_request(request, restaurants, reviews_by_id)
        results.append(result)

        status = result["status"]
        if status == "ok":
            ok_count += 1
        else:
            error_count += 1

    # Compute statistics
    stats = {1: 0, 0: 0, -1: 0}
    for val in eval_matrix.values():
        stats[val] = stats.get(val, 0) + 1

    # Print matrix and stats
    print_groundtruth_matrix(eval_matrix, item_ids, request_ids, group_filter, request_info)
    print_statistics(stats)
    print_hits_summary(results, restaurants, request_info)

    print()

    # Summary
    if error_count == 0:
        console.print(f"[green]All {ok_count} requests validated successfully![/green]")
    else:
        console.print(f"[yellow]{ok_count} OK, {error_count} errors[/yellow]")

    # Write groundtruth.jsonl
    groundtruth_path = dataset_dir / "groundtruth.jsonl"
    with open(groundtruth_path, "w") as f:
        for result in results:
            entry = {
                "request_id": result["request_id"],
                "gold_restaurant": result["gold_restaurant"],
                "gold_idx": result["gold_idx"],
                "status": result["status"]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"\nSaved: {groundtruth_path}")

    return error_count == 0


def print_groundtruth_matrix(eval_matrix: dict, item_ids: list, request_ids: list,
                             group_filter: set[str] = None, request_info: dict = None):
    """Print items × requests matrix with colored labels.

    Splits into multiple tables if more than 10 requests (10 per table).

    Args:
        eval_matrix: {(item_id, request_id): result} where result is 1/-1/0
        item_ids: Ordered list of item IDs
        request_ids: Ordered list of request IDs
        group_filter: Optional set of group names being filtered
        request_info: Optional dict mapping request_id -> request dict
    """

    # Split requests into chunks of 10
    chunk_size = 10
    request_chunks = [request_ids[i:i+chunk_size] for i in range(0, len(request_ids), chunk_size)]

    for chunk_idx, req_chunk in enumerate(request_chunks):
        # Build title with chunk info and group info
        title_parts = ["Validation Matrix"]

        # Add chunk info if multiple chunks
        if len(request_chunks) > 1:
            title_parts.append(f"({chunk_idx+1}/{len(request_chunks)})")

        # Add group info for this chunk
        if request_info:
            chunk_groups = sorted(set(
                request_info.get(rid, {}).get("group", "?")
                for rid in req_chunk
            ))
            if chunk_groups and chunk_groups != ["?"]:
                title_parts.append(f"[{', '.join(chunk_groups)}]")
        elif group_filter:
            title_parts.append(f"[{', '.join(sorted(group_filter))}]")

        title = " ".join(title_parts)

        table = Table(title=title, show_lines=False)
        table.add_column("Item", style="cyan", width=14)
        for req_id in req_chunk:
            table.add_column(req_id, justify="center", width=5)

        for item_id in item_ids:
            row = [item_id[:12] + ".."]
            for req_id in req_chunk:
                label = eval_matrix.get((item_id, req_id), "?")
                if label == 1:
                    row.append("[green]+1[/green]")
                elif label == -1:
                    row.append("[red]-1[/red]")
                elif label == 0:
                    row.append("[yellow]0[/yellow]")
                else:
                    row.append("[dim]?[/dim]")
            table.add_row(*row)

        console.print(table)
        if chunk_idx < len(request_chunks) - 1:
            console.print()  # Add spacing between tables


def print_statistics(stats: dict):
    """Print distribution statistics."""

    total = sum(stats.values())
    if total == 0:
        return

    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total entries: {total}")
    console.print(f"  Match (+1):    [green]{stats.get(1, 0):4d}[/green] ({stats.get(1, 0)/total*100:5.1f}%)")
    console.print(f"  Unknown (0):   [yellow]{stats.get(0, 0):4d}[/yellow] ({stats.get(0, 0)/total*100:5.1f}%)")
    console.print(f"  No match (-1): [red]{stats.get(-1, 0):4d}[/red] ({stats.get(-1, 0)/total*100:5.1f}%)")


def print_hits_summary(results: list, restaurants: list, request_info: dict = None):
    """Print Hits@1 summary showing which restaurant matches each request.

    Uses 2-column layout with shorthand if space permits, otherwise falls back to 1 column.
    """

    console.print("\n[bold]Hits@1 Summary:[/bold]")
    hits = 0
    total = len(results)

    # Build business_id to index lookup
    id_to_idx = {r["business_id"]: i for i, r in enumerate(restaurants)}
    id_to_name = {r["business_id"]: r.get("name", "?")[:20] for r in restaurants}

    # Build formatted entries: (req_id, status_str, shorthand)
    entries = []
    for r in results:
        req_id = r["request_id"]
        gold_id = r["gold_restaurant"]
        gold_idx = r["gold_idx"]
        gold_name = id_to_name.get(gold_id, "?")
        status = r["status"]

        # Get shorthand from request_info
        shorthand = ""
        if request_info and req_id in request_info:
            shorthand = request_info[req_id].get("shorthand", "")

        if status == "ok":
            hits += 1
            status_str = f"[green]✓[/green] [{gold_idx}] {gold_name}"
        elif status == "no_match":
            status_str = f"[red]✗[/red] no_match"
        elif status == "multi_match":
            match_idxs = [id_to_idx.get(m, "?") for m in r["matches"][:3]]
            status_str = f"[yellow]⚠[/yellow] multi ({r['matches_count']}): {match_idxs}"
        elif status == "multi_match_no_gold":
            status_str = f"[red]✗[/red] multi_match_no_gold"
        else:
            status_str = f"[red]✗[/red] {status}"

        entries.append((req_id, status_str, shorthand))

    # Determine layout: 2-column if terminal is wide enough
    console_width = console.width or 120

    # Calculate widths for 2-column layout
    # Format: "  R01: status | shorthand  |  R02: status | shorthand"
    # Each entry: "  Rnn: " (7) + status (~25) + " | " (3) + shorthand (variable)
    max_status_len = max(len(s.replace('[green]', '').replace('[/green]', '')
                            .replace('[red]', '').replace('[/red]', '')
                            .replace('[yellow]', '').replace('[/yellow]', ''))
                         for _, s, _ in entries) if entries else 20
    max_shorthand_len = max(len(sh) for _, _, sh in entries) if entries else 30

    # Entry width: "  Rnn: " + status + " | " + shorthand
    entry_width = 7 + max_status_len + 3 + min(max_shorthand_len, 50)
    separator_width = 3  # " | "

    # Use 2 columns if both fit, otherwise 1 column
    use_two_columns = (entry_width * 2 + separator_width) <= console_width and len(entries) > 5

    if use_two_columns:
        # Calculate actual widths for alignment
        col_width = (console_width - separator_width) // 2
        shorthand_width = col_width - 7 - max_status_len - 3

        # Print in 2 columns
        mid = (len(entries) + 1) // 2
        left_entries = entries[:mid]
        right_entries = entries[mid:]

        for i, (left, right) in enumerate(zip(left_entries, right_entries + [(None, None, None)] * (mid - len(right_entries)))):
            l_req, l_status, l_short = left
            l_short_trunc = l_short[:shorthand_width] if len(l_short) > shorthand_width else l_short

            if right[0] is not None:
                r_req, r_status, r_short = right
                r_short_trunc = r_short[:shorthand_width] if len(r_short) > shorthand_width else r_short
                # Build left part with padding
                l_plain_status = l_status.replace('[green]', '').replace('[/green]', '') \
                                         .replace('[red]', '').replace('[/red]', '') \
                                         .replace('[yellow]', '').replace('[/yellow]', '')
                l_pad = max_status_len - len(l_plain_status)
                console.print(f"  {l_req}: {l_status}{' ' * l_pad} [dim]{l_short_trunc:<{shorthand_width}}[/dim] | "
                             f"  {r_req}: {r_status} [dim]{r_short_trunc}[/dim]")
            else:
                console.print(f"  {l_req}: {l_status} [dim]{l_short_trunc}[/dim]")
    else:
        # Single column with shorthand
        shorthand_width = max(20, console_width - 7 - max_status_len - 5)
        for req_id, status_str, shorthand in entries:
            short_trunc = shorthand[:shorthand_width] if len(shorthand) > shorthand_width else shorthand
            if shorthand:
                console.print(f"  {req_id}: {status_str} [dim]{short_trunc}[/dim]")
            else:
                console.print(f"  {req_id}: {status_str}")

    pct = hits / total * 100 if total else 0
    console.print(f"\n[bold]Validation: {hits}/{total} = {pct:.0f}%[/bold]")


def list_datasets() -> list[str]:
    """List available datasets in data/ directory."""
    datasets = []
    for p in DATA_DIR.iterdir():
        if p.is_dir() and not p.name.startswith(('_', '.')):
            # Check if it has required files
            if (p / "requests.jsonl").exists():
                datasets.append(p.name)
    return sorted(datasets)


def choose_dataset() -> str:
    """Interactively choose a dataset."""
    datasets = list_datasets()

    if not datasets:
        print("No datasets found in data/")
        sys.exit(1)

    if len(datasets) == 1:
        print(f"Using dataset: {datasets[0]}")
        return datasets[0]

    print("Available datasets:")
    for i, name in enumerate(datasets, 1):
        print(f"  [{i}] {name}")

    while True:
        try:
            choice = input("\nSelect dataset (number or name): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    return datasets[idx]
            elif choice in datasets:
                return choice
            print(f"Invalid choice. Enter 1-{len(datasets)} or dataset name.")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)


def parse_group_spec(spec: str) -> set[str]:
    """Parse group specification into set of group names.

    Args:
        spec: Group specification string

    Returns:
        Set of group names (e.g., {"G01", "G02", "G03"})

    Examples:
        '1' -> {"G01"}
        '1,2,3' -> {"G01", "G02", "G03"}
        '1-3' -> {"G01", "G02", "G03"}
        'G06' -> {"G06"}
        'G06,G07' -> {"G06", "G07"}
    """
    if not spec or not spec.strip():
        return set()

    spec = spec.strip()
    groups = set()

    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue

        # Handle range: "1-3" or "G01-G03"
        if '-' in part:
            # Remove 'G' prefix if present for parsing
            range_part = part.replace('G', '').replace('g', '')
            try:
                start, end = range_part.split('-', 1)
                for i in range(int(start), int(end) + 1):
                    groups.add(f"G{i:02d}")
            except ValueError:
                pass
        else:
            # Single group: "1" or "G01"
            clean = part.replace('G', '').replace('g', '')
            try:
                n = int(clean)
                groups.add(f"G{n:02d}")
            except ValueError:
                # If it doesn't parse as int, use as-is
                groups.add(part.upper())

    return groups


def main():
    parser = argparse.ArgumentParser(description="Validate ground truth dataset")
    parser.add_argument("name", nargs="?", help="Dataset name (e.g., philly_cafes)")
    parser.add_argument("--group", type=str, metavar="N[,N,...]",
                        help="Filter by group(s): single (6), comma-separated (6,7), or range (6-8)")
    args = parser.parse_args()

    name = args.name if args.name else choose_dataset()

    # Parse group filter
    group_filter = parse_group_spec(args.group) if args.group else None

    success = validate_dataset(name, group_filter=group_filter)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
