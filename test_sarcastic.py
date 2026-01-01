#!/usr/bin/env python3
"""Test sarcastic attack on G01 requests."""

import json
from data.loader import load_yelp_dataset
from attack import apply_attack
from methods.cot import ChainOfThought
from run import format_query

# Initialize method
method = ChainOfThought()

# Load data
items, requests = load_yelp_dataset("selection_1", limit=3)
g01_requests = [r for r in requests if r["id"] in ["R00", "R01", "R02", "R03", "R04"]]

print("=== Comparing Clean vs Sarcastic Attack on G01 Requests ===")
print(f"Items: {len(items)}, G01 Requests: {len(g01_requests)}")
print()

# Get ground truth labels
gt_file = "data/yelp/groundtruth_1.jsonl"
gt_labels = {}
with open(gt_file) as f:
    for line in f:
        entry = json.loads(line)
        gt_labels[(entry["item_id"], entry["request_id"])] = entry["gold_label"]

# Apply sarcastic attack
items_attacked = apply_attack(items, "sarcastic", target_attributes=None)

# Test on a subset
results = {"clean_correct": 0, "attacked_correct": 0, "total": 0, "changed": 0}

for item_idx in range(min(2, len(items))):
    item_clean = items[item_idx]
    item_attacked = items_attacked[item_idx]
    item_id = item_clean["item_id"]
    item_name = item_clean.get("item_name", "N/A")

    print(f"=== {item_name} ===")

    # Show injected reviews
    injected = [r for r in item_attacked["item_data"] if r["review_id"].startswith("sarcastic_")]
    print(f"  Injected sarcastic reviews: {len(injected)}")
    for r in injected:
        print(f"    [{r['review_id']}] {r['review'][:60]}...")
    print()

    for req in g01_requests[:3]:  # Test on first 3 requests
        req_id = req["id"]
        scenario = req.get("scenario", "")
        gold = gt_labels.get((item_id, req_id), None)

        if gold is None:
            continue

        # Format queries
        query_clean = format_query(item_clean, mode="string")
        query_attacked = format_query(item_attacked, mode="string")

        # Get predictions
        try:
            pred_clean = cot(req["text"], query_clean)
            pred_attacked = cot(req["text"], query_attacked)

            results["total"] += 1
            if pred_clean == gold:
                results["clean_correct"] += 1
            if pred_attacked == gold:
                results["attacked_correct"] += 1
            if pred_clean != pred_attacked:
                results["changed"] += 1

            match_clean = "✓" if pred_clean == gold else "✗"
            match_attacked = "✓" if pred_attacked == gold else "✗"

            print(f"  {req_id} ({scenario}): gold={gold}")
            print(f"    Clean:     pred={pred_clean} {match_clean}")
            print(f"    Sarcastic: pred={pred_attacked} {match_attacked}")

            if pred_clean != pred_attacked:
                print(f"    *** PREDICTION CHANGED ***")
        except Exception as e:
            print(f"  {req_id}: Error - {e}")
        print()

# Summary
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Total evaluations: {results['total']}")
print(f"Clean correct:     {results['clean_correct']} ({100*results['clean_correct']/max(1,results['total']):.1f}%)")
print(f"Attacked correct:  {results['attacked_correct']} ({100*results['attacked_correct']/max(1,results['total']):.1f}%)")
print(f"Predictions changed: {results['changed']} ({100*results['changed']/max(1,results['total']):.1f}%)")
