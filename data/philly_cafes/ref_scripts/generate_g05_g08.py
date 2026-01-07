#!/usr/bin/env python3
"""
Generate G05-G08 Requests Using Condition Matrix

This script generates requests for groups G05-G08 using the condition matrix
to ensure unique restaurant matches. Each request uses the anchor-first design:
anchor conditions guarantee uniqueness, OR conditions add complexity.

Evidence Type Distribution Target:
- 50-60% item_meta
- 10-15% item_meta_hours
- 20-25% review_text
- 5-10% review_meta

Usage:
    python data/scripts/generate_g05_g08_v2.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

# Paths
DATA_DIR = Path(__file__).parent.parent / "philly_cafes"
MATRIX_FILE = DATA_DIR / "condition_matrix.json"
RESTAURANTS_FILE = DATA_DIR / "restaurants.jsonl"
REQUESTS_FILE = DATA_DIR / "requests.jsonl"

# Load condition matrix
def load_matrix() -> dict:
    with open(MATRIX_FILE) as f:
        return json.load(f)

def load_restaurants() -> List[dict]:
    restaurants = []
    with open(RESTAURANTS_FILE) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants

def load_existing_requests() -> List[dict]:
    requests = []
    with open(REQUESTS_FILE) as f:
        for line in f:
            requests.append(json.loads(line))
    return requests


# =============================================================================
# CONDITION TO EVIDENCE CONVERSION
# =============================================================================

def condition_to_evidence(cond_name: str, matrix: dict) -> dict:
    """Convert a condition name to its evidence structure."""
    cond = matrix["conditions"][cond_name]
    cond_def = cond["definition"]
    cond_type = cond["type"]

    if cond_type == "item_meta":
        evidence = {
            "kind": "item_meta",
            "path": cond_def["path"],
        }
        check_type = cond_def["check_type"]
        if check_type == "true":
            evidence["true"] = cond_def["value"]
        elif check_type == "contains":
            evidence["contains"] = cond_def["value"]
        elif check_type == "not_true":
            evidence["not_true"] = cond_def["value"]
        return evidence

    elif cond_type == "item_meta_hours":
        return {
            "kind": "item_meta_hours",
            "path": cond_def["path"],
            "true": cond_def["time_range"]
        }

    elif cond_type == "review_text":
        return {
            "kind": "review_text",
            "pattern": cond_def["pattern"]
        }

    elif cond_type == "review_meta":
        return {
            "kind": "review_text",
            "pattern": cond_def["pattern"],
            "weight_by": cond_def["weight_by"],
            "threshold": cond_def["threshold"]
        }

    raise ValueError(f"Unknown condition type: {cond_type}")


def get_natural_phrase(cond_name: str, matrix: dict) -> str:
    """Get natural language phrase for a condition."""
    cond = matrix["conditions"][cond_name]
    return cond["definition"].get("natural_phrase", cond_name)


# =============================================================================
# UNIQUE IDENTIFIER FINDER
# =============================================================================

def find_unique_identifier(target_idx: int, matrix: dict, max_conditions: int = 4) -> List[str]:
    """
    Find minimal condition combination that uniquely identifies a restaurant.
    Returns list of condition names.
    """
    target_conds = set(matrix["restaurants"][str(target_idx)]["satisfying_conditions"])

    # First check single conditions
    for cond_name in target_conds:
        if matrix["conditions"][cond_name]["count"] == 1:
            return [cond_name]

    # Try combinations of increasing size
    from itertools import combinations
    target_conds_list = list(target_conds)

    for size in range(2, min(max_conditions + 1, len(target_conds_list) + 1)):
        for combo in combinations(target_conds_list, size):
            # Find intersection
            satisfying = None
            for cond_name in combo:
                cond_sats = set(matrix["conditions"][cond_name]["satisfying_restaurants"])
                if satisfying is None:
                    satisfying = cond_sats
                else:
                    satisfying = satisfying & cond_sats

            if satisfying == {target_idx}:
                return list(combo)

    return []  # No unique identifier found


def categorize_conditions(conds: List[str], matrix: dict) -> Dict[str, List[str]]:
    """Categorize conditions by type."""
    categories = {
        "item_meta": [],
        "item_meta_hours": [],
        "review_text": [],
        "review_meta": []
    }
    for cond in conds:
        cond_type = matrix["conditions"][cond]["type"]
        categories[cond_type].append(cond)
    return categories


# =============================================================================
# REQUEST GENERATION
# =============================================================================

def generate_or_options(target_idx: int, matrix: dict, count: int = 3,
                        prefer_types: List[str] = None) -> List[str]:
    """
    Generate OR options that the target restaurant satisfies.
    Prefers review_text conditions for natural variety.
    """
    if prefer_types is None:
        prefer_types = ["review_text", "review_meta"]

    target_conds = matrix["restaurants"][str(target_idx)]["satisfying_conditions"]

    # Filter by preferred types
    candidates = []
    for cond in target_conds:
        cond_type = matrix["conditions"][cond]["type"]
        if cond_type in prefer_types:
            # Prefer broader conditions for OR (not too rare)
            cond_count = matrix["conditions"][cond]["count"]
            if cond_count >= 5:  # Broader conditions
                candidates.append(cond)

    if len(candidates) < count:
        # Add more conditions
        for cond in target_conds:
            cond_type = matrix["conditions"][cond]["type"]
            if cond_type in prefer_types and cond not in candidates:
                candidates.append(cond)

    # Random sample
    if len(candidates) >= count:
        return random.sample(candidates, count)
    return candidates


def build_request_structure(anchor_conds: List[str], or_conds: List[str],
                           matrix: dict, pattern: str = "G05") -> dict:
    """
    Build request structure based on pattern.

    Patterns:
    - G05: AND(anchors..., OR(a, b, c))
    - G06: AND(anchor, OR(AND(a,b), AND(c,d)))
    - G07: AND(anchor, OR(a,b), OR(c,d))
    - G08: AND(anchor, simple, OR(opt, AND(c,d)))
    """

    if pattern == "G05":
        # Simple: anchors + triple OR
        args = []
        for cond in anchor_conds:
            args.append({
                "aspect": cond,
                "evidence": condition_to_evidence(cond, matrix)
            })

        or_args = []
        for cond in or_conds:
            or_args.append({
                "aspect": cond,
                "evidence": condition_to_evidence(cond, matrix)
            })

        args.append({"op": "OR", "args": or_args})
        return {"op": "AND", "args": args}

    elif pattern == "G06":
        # Nested: anchor + OR(AND(...), AND(...))
        args = []
        for cond in anchor_conds:
            args.append({
                "aspect": cond,
                "evidence": condition_to_evidence(cond, matrix)
            })

        # Split or_conds into two AND groups
        mid = len(or_conds) // 2
        and1_conds = or_conds[:mid] if mid > 0 else or_conds[:1]
        and2_conds = or_conds[mid:] if mid > 0 else or_conds[1:]

        and1_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in and1_conds]
        and2_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in and2_conds]

        or_branch = {
            "op": "OR",
            "args": [
                {"op": "AND", "args": and1_args} if len(and1_args) > 1 else and1_args[0],
                {"op": "AND", "args": and2_args} if len(and2_args) > 1 else and2_args[0]
            ]
        }
        args.append(or_branch)
        return {"op": "AND", "args": args}

    elif pattern == "G07":
        # Chained: anchor + OR(...) + OR(...)
        args = []
        for cond in anchor_conds:
            args.append({
                "aspect": cond,
                "evidence": condition_to_evidence(cond, matrix)
            })

        # Split into two ORs
        mid = len(or_conds) // 2
        or1_conds = or_conds[:mid] if mid > 0 else or_conds[:2]
        or2_conds = or_conds[mid:] if mid > 0 else or_conds[2:]

        or1_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in or1_conds]
        or2_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in or2_conds]

        if or1_args:
            args.append({"op": "OR", "args": or1_args})
        if or2_args:
            args.append({"op": "OR", "args": or2_args})

        return {"op": "AND", "args": args}

    elif pattern == "G08":
        # Unbalanced: anchor + simple + OR(opt, AND(...))
        args = []

        # Use first anchor condition
        if anchor_conds:
            args.append({
                "aspect": anchor_conds[0],
                "evidence": condition_to_evidence(anchor_conds[0], matrix)
            })

        # Use second anchor as simple condition
        if len(anchor_conds) > 1:
            args.append({
                "aspect": anchor_conds[1],
                "evidence": condition_to_evidence(anchor_conds[1], matrix)
            })

        # Build unbalanced OR
        if len(or_conds) >= 3:
            simple_opt = {"aspect": or_conds[0], "evidence": condition_to_evidence(or_conds[0], matrix)}
            and_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in or_conds[1:3]]
            or_branch = {
                "op": "OR",
                "args": [simple_opt, {"op": "AND", "args": and_args}]
            }
            args.append(or_branch)
        elif or_conds:
            # Fallback
            or_args = [{"aspect": c, "evidence": condition_to_evidence(c, matrix)} for c in or_conds]
            args.append({"op": "OR", "args": or_args})

        return {"op": "AND", "args": args}

    raise ValueError(f"Unknown pattern: {pattern}")


def generate_natural_text(anchor_conds: List[str], or_conds: List[str],
                         matrix: dict, pattern: str) -> str:
    """Generate natural language text for the request."""
    anchor_phrases = [get_natural_phrase(c, matrix) for c in anchor_conds]
    or_phrases = [get_natural_phrase(c, matrix) for c in or_conds]

    # Build text
    parts = ["Looking for a cafe"]

    # Add anchor phrases
    if anchor_phrases:
        adj_phrases = []
        with_phrases = []
        for p in anchor_phrases:
            if p.startswith("with ") or p.startswith("that "):
                with_phrases.append(p)
            else:
                adj_phrases.append(p)

        if adj_phrases:
            parts.append(f"that's {', '.join(adj_phrases)}")
        for wp in with_phrases:
            parts.append(wp)

    # Add OR phrases based on pattern
    if pattern == "G05":
        if or_phrases:
            parts.append(f"that either {', '.join(or_phrases[:-1])}, or {or_phrases[-1]}"
                        if len(or_phrases) > 1 else or_phrases[0])
    elif pattern == "G06":
        mid = len(or_phrases) // 2
        if mid > 0:
            group1 = " and ".join(or_phrases[:mid])
            group2 = " and ".join(or_phrases[mid:])
            parts.append(f"that's either ({group1}) or ({group2})")
    elif pattern == "G07":
        mid = len(or_phrases) // 2
        if mid > 0:
            or1 = " or ".join(or_phrases[:mid])
            or2 = " or ".join(or_phrases[mid:])
            parts.append(f"that's either {or1}, and either {or2}")
    elif pattern == "G08":
        if len(or_phrases) >= 3:
            parts.append(f"that's either {or_phrases[0]}, or both {or_phrases[1]} and {or_phrases[2]}")

    return " ".join(parts)


# =============================================================================
# GROUP GENERATORS
# =============================================================================

def assign_restaurants_to_groups(matrix: dict) -> Dict[str, List[int]]:
    """
    Assign restaurants to groups G05-G08, ensuring diversity.

    Each group gets 10 restaurants (except G08 which gets R70-R79).
    We aim for variety in restaurant types and anchor conditions.
    """
    # Get restaurants with their unique identifiers
    rest_ids = []
    for i in range(50):
        uid = find_unique_identifier(i, matrix)
        rest_ids.append((i, uid, len(uid)))

    # Sort by identifier complexity (simpler first)
    rest_ids.sort(key=lambda x: x[2])

    # Assign to groups - prioritize diversity
    used = set()
    assignments = {
        "G05": [],  # R40-R49: Simple anchors
        "G06": [],  # R50-R59: Medium complexity
        "G07": [],  # R60-R69: Chained ORs
        "G08": []   # R70-R79: Unbalanced
    }

    # G05: Use restaurants with single unique anchors first
    for i, uid, complexity in rest_ids:
        if complexity == 1 and len(assignments["G05"]) < 10:
            assignments["G05"].append(i)
            used.add(i)

    # Fill remaining G05 with simple 2-condition identifiers
    for i, uid, complexity in rest_ids:
        if i not in used and complexity == 2 and len(assignments["G05"]) < 10:
            assignments["G05"].append(i)
            used.add(i)

    # G08 needs R70 special (already unique with price_upscale)
    # Use restaurants with medium complexity for G08
    for i, uid, complexity in rest_ids:
        if i not in used and complexity >= 2 and len(assignments["G08"]) < 10:
            assignments["G08"].append(i)
            used.add(i)

    # G06 and G07: distribute remaining
    remaining = [i for i in range(50) if i not in used]
    random.shuffle(remaining)

    for i in remaining:
        if len(assignments["G06"]) < 10:
            assignments["G06"].append(i)
        elif len(assignments["G07"]) < 10:
            assignments["G07"].append(i)

    return assignments


def generate_group_requests(group: str, restaurant_indices: List[int],
                           matrix: dict, restaurants: List[dict]) -> List[dict]:
    """Generate requests for a group."""
    requests = []

    base_id = {"G05": 40, "G06": 50, "G07": 60, "G08": 70}[group]

    for offset, rest_idx in enumerate(restaurant_indices):
        request_id = f"R{base_id + offset:02d}"

        # Find unique identifier
        anchor_conds = find_unique_identifier(rest_idx, matrix)
        if not anchor_conds:
            print(f"Warning: No unique identifier for [{rest_idx}] {restaurants[rest_idx]['name']}")
            continue

        # Generate OR options
        or_conds = generate_or_options(rest_idx, matrix, count=4)

        # Ensure we have enough OR conditions
        if len(or_conds) < 2:
            # Use item_meta conditions as fallback
            target_conds = matrix["restaurants"][str(rest_idx)]["satisfying_conditions"]
            for cond in target_conds:
                if cond not in or_conds and cond not in anchor_conds:
                    or_conds.append(cond)
                    if len(or_conds) >= 4:
                        break

        # Build structure
        structure = build_request_structure(anchor_conds, or_conds, matrix, pattern=group)

        # Generate text
        text = generate_natural_text(anchor_conds, or_conds, matrix, pattern=group)

        request = {
            "id": request_id,
            "group": group,
            "scenario": f"{group} pattern",
            "text": text,
            "structure": structure,
            "gold_restaurant": restaurants[rest_idx]["business_id"]
        }

        requests.append(request)

    return requests


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")
    matrix = load_matrix()
    restaurants = load_restaurants()
    existing_requests = load_existing_requests()

    print(f"Loaded {len(restaurants)} restaurants, {len(matrix['conditions'])} conditions")

    # Assign restaurants to groups
    print("\nAssigning restaurants to groups...")
    assignments = assign_restaurants_to_groups(matrix)

    for group, indices in assignments.items():
        print(f"  {group}: {len(indices)} restaurants - {indices[:5]}...")

    # Generate requests
    print("\nGenerating requests...")
    all_new_requests = []

    for group in ["G05", "G06", "G07", "G08"]:
        group_requests = generate_group_requests(group, assignments[group], matrix, restaurants)
        all_new_requests.extend(group_requests)
        print(f"  {group}: {len(group_requests)} requests generated")

    # Merge with existing requests (keep G01-G04, replace G05-G08)
    final_requests = []
    for req in existing_requests:
        if req["group"] in ["G01", "G02", "G03", "G04"]:
            final_requests.append(req)

    final_requests.extend(all_new_requests)

    # Sort by request ID
    final_requests.sort(key=lambda x: int(x["id"][1:]))

    # Write output
    print(f"\nWriting {len(final_requests)} requests to {REQUESTS_FILE}...")
    with open(REQUESTS_FILE, 'w') as f:
        for req in final_requests:
            f.write(json.dumps(req) + '\n')

    print("Done!")

    # Show sample
    print("\n=== Sample G05 Request ===")
    for req in all_new_requests:
        if req["group"] == "G05":
            print(json.dumps(req, indent=2))
            break


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    main()
