#!/usr/bin/env python3
"""
Redistribute G06 and G07 requests to use top 20 restaurants.

This script generates new requests for R50-R68 (19 requests) using
restaurants that are already frequently used as gold answers, ensuring
100% coverage at N=20 scaling.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "philly_cafes"


def load_data():
    """Load condition matrix, restaurants, and existing requests."""
    with open(DATA_DIR / "condition_matrix.json") as f:
        matrix = json.load(f)

    with open(DATA_DIR / "restaurants.jsonl") as f:
        restaurants = [json.loads(line) for line in f]

    with open(DATA_DIR / "requests.jsonl") as f:
        requests = [json.loads(line) for line in f]

    with open(DATA_DIR / "groundtruth.jsonl") as f:
        groundtruth = [json.loads(line) for line in f]

    return matrix, restaurants, requests, groundtruth


def get_restaurant_conditions(matrix, idx, cond_type=None):
    """Get conditions satisfied by a restaurant."""
    conditions = matrix['conditions']
    satisfied = []
    for cond_name, cond_data in conditions.items():
        if idx in cond_data['satisfying_restaurants']:
            if cond_type is None or cond_data['type'] == cond_type:
                satisfied.append((cond_name, cond_data))
    return satisfied


def get_review_text_conditions(matrix, idx):
    """Get simple review_text conditions (no weight_by)."""
    all_conds = get_restaurant_conditions(matrix, idx, 'review_text')
    return [
        (n, d) for n, d in all_conds
        if 'weight_by' not in d.get('definition', {})
    ]


def build_evidence_spec(cond_name, cond_data):
    """Build evidence spec from condition definition."""
    defn = cond_data['definition']
    ctype = cond_data['type']

    if ctype == 'item_meta':
        spec = {
            "kind": "item_meta",
            "path": defn['path'],
        }
        if defn.get('check_type') == 'true':
            spec['true'] = defn['value']
        elif defn.get('check_type') == 'contains':
            spec['contains'] = defn['value']
        return spec

    elif ctype == 'review_text':
        spec = {
            "kind": "review_text",
            "pattern": defn['pattern'],
        }
        if 'weight_by' in defn:
            spec['weight_by'] = defn['weight_by']
        return spec

    elif ctype == 'review_meta':
        # review_meta is similar to review_text but always has weight_by
        spec = {
            "kind": "review_text",
            "pattern": defn['pattern'],
            "weight_by": defn['weight_by'],
        }
        return spec

    elif ctype == 'item_meta_hours':
        return {
            "kind": "item_meta_hours",
            "path": defn['path'],
            "true": defn['value'],
        }

    return None


def natural_phrase(cond_name, cond_data):
    """Get natural language phrase for condition."""
    defn = cond_data.get('definition', {})
    if 'natural_phrase' in defn:
        return defn['natural_phrase']

    # Fallback for review conditions
    if cond_data['type'] == 'review_text':
        pattern = defn.get('pattern', cond_name.replace('_reviews', ''))
        return f"where reviews mention '{pattern.split('|')[0]}'"

    return cond_name


# Assignments - using anchors that are unique within ALL 50 restaurants
G06_ASSIGNMENTS = [
    # (request_id, gold_idx, anchor_conditions)
    ("R50", 32, ["takeout_no"]),                        # unique depth=1
    ("R51", 25, ["wifi_paid"]),                         # unique depth=1
    ("R52", 21, ["ambience_intimate"]),                 # unique depth=1
    ("R53", 18, ["dogs_yes", "outdoor_no"]),            # unique depth=2
    ("R54", 19, ["dogs_yes", "meeting_reviews"]),       # unique depth=2
    ("R55", 14, ["kids_no", "outdoor_no"]),             # unique depth=2
    ("R56", 12, ["byob", "music_reviews"]),             # unique depth=2
    ("R57", 11, ["byob", "ambience_classy"]),           # unique depth=2
    ("R58", 5, ["price_budget", "byob"]),               # unique depth=2
    ("R59", 17, ["price_mid", "dogs_yes", "ambience_trendy"]),  # unique depth=3
]

G07_ASSIGNMENTS = [
    # (request_id, gold_idx, anchor_conditions)
    ("R60", 32, ["takeout_no"]),
    ("R61", 25, ["wifi_paid"]),
    ("R62", 21, ["ambience_intimate"]),
    ("R63", 18, ["dogs_yes", "outdoor_no"]),
    ("R64", 19, ["dogs_yes", "meeting_reviews"]),
    ("R65", 14, ["kids_no", "outdoor_no"]),
    ("R66", 12, ["byob", "music_reviews"]),
    ("R67", 11, ["byob", "ambience_classy"]),
    ("R68", 5, ["price_budget", "byob"]),
    # R69 stays unchanged (gold_idx=17, already in top 20)
]


def generate_g06_request(req_id, gold_idx, anchor_conds, matrix, restaurants):
    """
    Generate a G06 request: AND(anchors, OR(AND(a,b), AND(c,d)))
    """
    conditions = matrix['conditions']

    # Build anchor evidence specs
    anchor_args = []
    anchor_phrases = []
    for cond_name in anchor_conds:
        cond_data = conditions[cond_name]
        anchor_args.append({
            "aspect": cond_name,
            "evidence": build_evidence_spec(cond_name, cond_data)
        })
        anchor_phrases.append(natural_phrase(cond_name, cond_data))

    # Get review conditions for OR blocks
    review_conds = get_review_text_conditions(matrix, gold_idx)
    # Pick 4 different review conditions for the OR(AND, AND) structure
    or_conds = []
    for n, d in review_conds:
        if n not in anchor_conds and len(or_conds) < 4:
            or_conds.append((n, d))

    # Ensure we have 4 conditions
    while len(or_conds) < 4:
        # Add common review conditions that all restaurants satisfy
        common = ["coffee_reviews", "friendly_reviews", "great_reviews", "good_reviews"]
        for c in common:
            if c in conditions and c not in [n for n, _ in or_conds] and c not in anchor_conds:
                or_conds.append((c, conditions[c]))
                if len(or_conds) >= 4:
                    break

    # Build OR block: OR(AND(c1, c2), AND(c3, c4))
    or_block = {
        "op": "OR",
        "args": [
            {
                "op": "AND",
                "args": [
                    {"aspect": or_conds[0][0], "evidence": build_evidence_spec(*or_conds[0])},
                    {"aspect": or_conds[1][0], "evidence": build_evidence_spec(*or_conds[1])},
                ]
            },
            {
                "op": "AND",
                "args": [
                    {"aspect": or_conds[2][0], "evidence": build_evidence_spec(*or_conds[2])},
                    {"aspect": or_conds[3][0], "evidence": build_evidence_spec(*or_conds[3])},
                ]
            }
        ]
    }

    # Build full structure
    structure = {
        "op": "AND",
        "args": anchor_args + [or_block]
    }

    # Build natural language text
    or_phrases = [
        f"({natural_phrase(*or_conds[0])} and {natural_phrase(*or_conds[1])})",
        f"({natural_phrase(*or_conds[2])} and {natural_phrase(*or_conds[3])})"
    ]
    text = f"Looking for a cafe that's {' and '.join(anchor_phrases)} that's either {or_phrases[0]} or {or_phrases[1]}"

    return {
        "id": req_id,
        "group": "G06",
        "scenario": "G06 pattern",
        "text": text,
        "structure": structure,
        "gold_restaurant": restaurants[gold_idx]['business_id']
    }


def generate_g07_request(req_id, gold_idx, anchor_conds, matrix, restaurants):
    """
    Generate a G07 request: AND(anchors, OR(a,b), OR(c,d))
    """
    conditions = matrix['conditions']

    # Build anchor evidence specs
    anchor_args = []
    anchor_phrases = []
    for cond_name in anchor_conds:
        cond_data = conditions[cond_name]
        anchor_args.append({
            "aspect": cond_name,
            "evidence": build_evidence_spec(cond_name, cond_data)
        })
        anchor_phrases.append(natural_phrase(cond_name, cond_data))

    # Get review conditions for OR blocks
    review_conds = get_review_text_conditions(matrix, gold_idx)
    # Pick 4 different review conditions for two separate OR blocks
    or_conds = []
    for n, d in review_conds:
        if n not in anchor_conds and len(or_conds) < 4:
            or_conds.append((n, d))

    # Ensure we have 4 conditions
    while len(or_conds) < 4:
        common = ["coffee_reviews", "friendly_reviews", "great_reviews", "good_reviews"]
        for c in common:
            if c in conditions and c not in [n for n, _ in or_conds] and c not in anchor_conds:
                or_conds.append((c, conditions[c]))
                if len(or_conds) >= 4:
                    break

    # Build two separate OR blocks
    or_block_1 = {
        "op": "OR",
        "args": [
            {"aspect": or_conds[0][0], "evidence": build_evidence_spec(*or_conds[0])},
            {"aspect": or_conds[1][0], "evidence": build_evidence_spec(*or_conds[1])},
        ]
    }
    or_block_2 = {
        "op": "OR",
        "args": [
            {"aspect": or_conds[2][0], "evidence": build_evidence_spec(*or_conds[2])},
            {"aspect": or_conds[3][0], "evidence": build_evidence_spec(*or_conds[3])},
        ]
    }

    # Build full structure
    structure = {
        "op": "AND",
        "args": anchor_args + [or_block_1, or_block_2]
    }

    # Build natural language text
    text = (
        f"Looking for a cafe that's {' and '.join(anchor_phrases)} "
        f"that's either {natural_phrase(*or_conds[0])} or {natural_phrase(*or_conds[1])}, "
        f"and either {natural_phrase(*or_conds[2])} or {natural_phrase(*or_conds[3])}"
    )

    return {
        "id": req_id,
        "group": "G07",
        "scenario": "G07 pattern",
        "text": text,
        "structure": structure,
        "gold_restaurant": restaurants[gold_idx]['business_id']
    }


def main():
    matrix, restaurants, requests, groundtruth = load_data()

    # Build new requests
    new_requests = []

    # Keep R00-R49 unchanged
    for r in requests:
        if r['id'] < 'R50':
            new_requests.append(r)

    # Generate G06 requests (R50-R59)
    print("Generating G06 requests (R50-R59)...")
    for req_id, gold_idx, anchors in G06_ASSIGNMENTS:
        req = generate_g06_request(req_id, gold_idx, anchors, matrix, restaurants)
        new_requests.append(req)
        print(f"  {req_id} -> [{gold_idx}] {restaurants[gold_idx]['name']}")

    # Generate G07 requests (R60-R68)
    print("\nGenerating G07 requests (R60-R68)...")
    for req_id, gold_idx, anchors in G07_ASSIGNMENTS:
        req = generate_g07_request(req_id, gold_idx, anchors, matrix, restaurants)
        new_requests.append(req)
        print(f"  {req_id} -> [{gold_idx}] {restaurants[gold_idx]['name']}")

    # Keep R69 unchanged
    for r in requests:
        if r['id'] == 'R69':
            new_requests.append(r)
            print(f"\n  R69 -> [17] {restaurants[17]['name']} (unchanged)")
            break

    # Keep R70-R79 unchanged
    for r in requests:
        if r['id'] >= 'R70':
            new_requests.append(r)

    # Sort by ID
    new_requests.sort(key=lambda r: r['id'])

    # Build new groundtruth
    new_groundtruth = []
    for req in new_requests:
        # Find gold_idx from business_id
        gold_id = req['gold_restaurant']
        gold_idx = next(i for i, r in enumerate(restaurants) if r['business_id'] == gold_id)
        new_groundtruth.append({
            "request_id": req['id'],
            "gold_restaurant": gold_id,
            "gold_idx": gold_idx,
            "status": "ok"
        })

    # Write output files
    print(f"\nWriting {len(new_requests)} requests to requests.jsonl...")
    with open(DATA_DIR / "requests.jsonl", 'w') as f:
        for req in new_requests:
            f.write(json.dumps(req) + '\n')

    print(f"Writing {len(new_groundtruth)} groundtruth entries to groundtruth.jsonl...")
    with open(DATA_DIR / "groundtruth.jsonl", 'w') as f:
        for gt in new_groundtruth:
            f.write(json.dumps(gt) + '\n')

    print("\nDone!")


if __name__ == "__main__":
    main()
