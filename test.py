#!/usr/bin/env python3
"""
LLM Evaluation Script for Yelp-style Restaurant Recommendation Dataset

This script evaluates an LLM method on a JSONL dataset where each restaurant
has reviews and ground truth recommendation labels for different user requests.

IMPORTANT: This script is designed to prevent answer leakage.
- final_answers and condition_satisfy fields are NEVER included in queries
- These fields are used ONLY for evaluation/comparison

Usage:
    python evaluate_llm.py --data path/to/data.jsonl --out results.jsonl
    python evaluate_llm.py --data data.jsonl --out results.jsonl --limit 100
    python evaluate_llm.py --data data.jsonl --out results.jsonl --print_examples 5
"""

import json
import argparse
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Any, Optional, Tuple


# =============================================================================
# USER REQUESTS DEFINITION
# =============================================================================
# These are the three user request contexts (R0, R1, R2) that the method will
# evaluate restaurants against. Each corresponds to a different user persona/need.

USER_REQUESTS = [
    # R0: Quiet dining with comfortable seating, budget-conscious
    (
        "I'm looking for a quiet restaurant with comfortable seating that won't "
        "break the bank. I want a peaceful dining experience where I can have a "
        "conversation without shouting."
    ),
    # R1: Allergy-conscious dining with clear labeling
    (
        "I have food allergies and need a restaurant that takes allergen care "
        "seriously. I need clear ingredient labeling and low cross-contamination risk."
    ),
    # R2: Authentic Chicago experience for tourists
    (
        "I'm visiting Chicago and want an authentic local experience with classic "
        "Chicago dishes. I'd like somewhere that's tourist-friendly but still has "
        "that genuine local vibe."
    ),
]


# =============================================================================
# PREDICTION NORMALIZATION
# =============================================================================

def normalize_pred(raw_pred: Any) -> int:
    """
    Normalize method output to {-1, 0, 1}.

    Handles various formats:
    - Integers: -1, 0, 1
    - Strings: "-1", "0", "1", "recommend", "not recommend", "neutral", etc.
    - Booleans: True -> 1, False -> -1

    Args:
        raw_pred: Raw prediction from the method (can be int, str, bool, etc.)

    Returns:
        int: Normalized prediction in {-1, 0, 1}

    Raises:
        ValueError: If the prediction cannot be normalized
    """
    # Handle None
    if raw_pred is None:
        raise ValueError("Prediction is None")

    # Handle integers directly
    if isinstance(raw_pred, int) and not isinstance(raw_pred, bool):
        if raw_pred in {-1, 0, 1}:
            return raw_pred
        raise ValueError(f"Integer prediction {raw_pred} not in {{-1, 0, 1}}")

    # Handle booleans (must check before int since bool is subclass of int)
    if isinstance(raw_pred, bool):
        return 1 if raw_pred else -1

    # Handle floats (round to nearest valid value)
    if isinstance(raw_pred, float):
        if raw_pred <= -0.5:
            return -1
        elif raw_pred >= 0.5:
            return 1
        else:
            return 0

    # Handle strings
    if isinstance(raw_pred, str):
        raw_str = raw_pred.strip().lower()

        # Direct numeric strings
        if raw_str in {"-1", "0", "1"}:
            return int(raw_str)

        # Check for negative indicators FIRST (order matters for "not recommend")
        negative_patterns = [
            "not recommend", "not recommended", "don't recommend",
            "do not recommend", "wouldn't recommend", "cannot recommend",
            "no", "negative", "bad", "unsuitable", "doesn't fit",
            "does not fit", "no match", "reject", "rejected", "avoid",
            "not suitable", "inappropriate", "poor fit", "poor match"
        ]
        for pattern in negative_patterns:
            if pattern in raw_str:
                return -1
        if raw_str == "-1" or raw_str == "false":
            return -1

        # Positive indicators -> 1
        positive_patterns = [
            "recommend", "recommended", "yes", "positive", "good",
            "suitable", "fits", "fit", "match", "matches", "approve",
            "approved", "great", "excellent", "definitely", "absolutely",
            "highly", "perfect"
        ]
        for pattern in positive_patterns:
            if pattern in raw_str:
                return 1
        if raw_str == "1" or raw_str == "true":
            return 1

        # Neutral indicators -> 0
        neutral_patterns = [
            "neutral", "uncertain", "maybe", "unsure", "unclear",
            "mixed", "ambiguous", "unknown", "neither", "undecided",
            "depends", "possibly", "potentially", "moderate"
        ]
        for pattern in neutral_patterns:
            if pattern in raw_str:
                return 0
        if raw_str == "0":
            return 0

        # Try to extract a number from the string
        for token in raw_str.replace(",", " ").replace(":", " ").split():
            token = token.strip("()[]{}.,;:")
            if token in {"-1", "0", "1"}:
                return int(token)

    raise ValueError(f"Cannot normalize prediction: {repr(raw_pred)}")


# =============================================================================
# QUERY FORMATTING (NO LEAKAGE)
# =============================================================================

def format_query(item: Dict[str, Any]) -> Tuple[str, int]:
    """
    Format a restaurant item into a query string for the method.

    =========================================================================
    CRITICAL LEAKAGE PREVENTION:
    This function must NOT include any of these fields in the output:
    - final_answers (ground truth recommendation labels)
    - condition_satisfy (intermediate condition evaluation labels)

    These fields exist ONLY for evaluation purposes and including them
    in the query would constitute answer leakage.
    =========================================================================

    Only includes:
    - Restaurant metadata (name, city, neighborhood, price_range, cuisine)
    - Review texts with their IDs

    Args:
        item: Restaurant item from the JSONL dataset

    Returns:
        Tuple of (formatted query string, number of reviews)
    """
    # Extract safe metadata fields only
    name = item.get("item_name", "Unknown")
    city = item.get("city", "Unknown")
    neighborhood = item.get("neighborhood", "Unknown")
    price_range = item.get("price_range", "Unknown")

    # Format cuisine list
    cuisines = item.get("cuisine", [])
    if isinstance(cuisines, list):
        cuisine_str = ", ".join(cuisines) if cuisines else "Unknown"
    else:
        cuisine_str = str(cuisines)

    # Build the header with deterministic template
    query_parts = [
        "Restaurant:",
        f"Name: {name}",
        f"City: {city}",
        f"Neighborhood: {neighborhood}",
        f"Price: {price_range}",
        f"Cuisine: {cuisine_str}",
        "",  # Empty line before reviews
        "Reviews:",
    ]

    # Add reviews - ONLY review_id and review text
    # =========================================================================
    # LEAKAGE CHECK: We intentionally DO NOT include 'condition_satisfy' here.
    # That field contains ground truth labels and must NEVER be in the query.
    # Only review_id and review text are safe to include.
    # =========================================================================
    item_data = item.get("item_data", [])
    num_reviews = 0
    for review_entry in item_data:
        review_id = review_entry.get("review_id", "unknown")
        review_text = review_entry.get("review", "")

        # SAFE: Only include review_id and review text
        # UNSAFE (not included): condition_satisfy
        query_parts.append(f"[{review_id}] {review_text}")
        num_reviews += 1

    return "\n".join(query_parts), num_reviews


# =============================================================================
# CONFUSION MATRIX UTILITIES
# =============================================================================

def create_empty_confusion_matrix() -> Dict[int, Dict[int, int]]:
    """Create an empty 3x3 confusion matrix for labels {-1, 0, 1}."""
    return {
        gold: {pred: 0 for pred in [-1, 0, 1]}
        for gold in [-1, 0, 1]
    }


def print_confusion_matrix(confusion: Dict[int, Dict[int, int]]) -> None:
    """Print a formatted 3x3 confusion matrix."""
    labels = [-1, 0, 1]

    print("\nConfusion Matrix (rows=gold, cols=pred):")
    print("           Pred")
    print("Gold     -1     0     1   | Total")
    print("-" * 38)

    for gold in labels:
        row = confusion.get(gold, {})
        counts = [row.get(pred, 0) for pred in labels]
        row_total = sum(counts)
        print(f"  {gold:2d}    {counts[0]:4d}  {counts[1]:4d}  {counts[2]:4d}   | {row_total:4d}")

    # Print column totals
    print("-" * 38)
    col_totals = [sum(confusion.get(g, {}).get(p, 0) for g in labels) for p in labels]
    grand_total = sum(col_totals)
    print(f"Total   {col_totals[0]:4d}  {col_totals[1]:4d}  {col_totals[2]:4d}   | {grand_total:4d}")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_dataset(
    data_path: str,
    method: Callable[[str, str], int],
    limit: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the method on the entire dataset.

    Args:
        data_path: Path to the JSONL data file
        method: Callable that takes (
        ) and returns recommendation
        limit: Optional limit on number of items to process
        verbose: Whether to print progress

    Returns:
        Dictionary containing evaluation results
    """
    results = []

    # Counters for accuracy
    total_correct = 0
    total_count = 0
    per_request_correct = defaultdict(int)
    per_request_count = defaultdict(int)

    # Confusion matrices: overall and per-request
    confusion = create_empty_confusion_matrix()
    per_request_confusion = {
        f"R{i}": create_empty_confusion_matrix() for i in range(3)
    }

    # Error counter
    error_count = 0

    # Load and process data
    items_processed = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if limit is not None and items_processed >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}",
                      file=sys.stderr)
                continue

            item_id = item.get("item_id", f"unknown_{line_num}")
            items_processed += 1

            if verbose and items_processed % 100 == 0:
                print(f"  Processed {items_processed} items...", file=sys.stderr)

            # Format the query (NO LEAKAGE - see format_query function)
            query, num_reviews = format_query(item)
            query_length = len(query)

            # ================================================================
            # CRITICAL: final_answers is ONLY used for evaluation below.
            # It is NEVER passed to the method - only query and context are.
            # ================================================================
            final_answers = item.get("final_answers", {})

            # Evaluate for each request
            for req_idx, user_request in enumerate(USER_REQUESTS):
                request_id = f"R{req_idx}"

                # Get gold label (ONLY for evaluation, not passed to method)
                gold = final_answers.get(request_id)
                if gold is None:
                    print(f"Warning: Missing gold label for {item_id}/{request_id}",
                          file=sys.stderr)
                    continue

                # Ensure gold is int
                gold = int(gold)

                # Call the method with ONLY query and context (no ground truth)
                error_msg = None
                try:
                    raw_pred = method(query, user_request)
                    pred = normalize_pred(raw_pred)
                except Exception as e:
                    # On failure, default to 0 (neutral) and log error
                    pred = 0
                    error_msg = str(e)
                    error_count += 1
                    print(f"Warning: Method failed for {item_id}/{request_id}: {e}",
                          file=sys.stderr)

                # Check correctness
                correct = (pred == gold)

                # Update counters
                total_count += 1
                if correct:
                    total_correct += 1
                per_request_count[request_id] += 1
                if correct:
                    per_request_correct[request_id] += 1

                # Update confusion matrices
                confusion[gold][pred] += 1
                per_request_confusion[request_id][gold][pred] += 1

                # Store result record
                result_record = {
                    "item_id": item_id,
                    "request_id": request_id,
                    "pred": pred,
                    "gold": gold,
                    "correct": correct,
                    "query_length": query_length,
                    "number_of_reviews": num_reviews,
                }
                if error_msg:
                    result_record["error"] = error_msg

                results.append(result_record)

    # Calculate accuracies
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    per_request_accuracy = {}
    for req_id in ["R0", "R1", "R2"]:
        count = per_request_count[req_id]
        correct = per_request_correct[req_id]
        per_request_accuracy[req_id] = correct / count if count > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
        "items_processed": items_processed,
        "error_count": error_count,
        "per_request_accuracy": per_request_accuracy,
        "per_request_correct": dict(per_request_correct),
        "per_request_counts": dict(per_request_count),
        "confusion_matrix": confusion,
        "per_request_confusion": per_request_confusion,
        "detailed_results": results,
    }


def print_incorrect_examples(
    results: List[Dict[str, Any]],
    data_path: str,
    k: int,
    max_query_len: int = 500
) -> None:
    """
    Print K incorrect examples with shortened queries for analysis.

    Args:
        results: List of result records from evaluation
        data_path: Path to original data file (to reconstruct queries)
        k: Number of incorrect examples to print
        max_query_len: Maximum query length to display
    """
    incorrect = [r for r in results if not r["correct"]]

    if not incorrect:
        print("\nNo incorrect examples found!")
        return

    # Load the original data to reconstruct queries
    items_by_id = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    items_by_id[item.get("item_id")] = item
                except json.JSONDecodeError:
                    continue

    print(f"\n{'='*70}")
    print(f"INCORRECT EXAMPLES ({min(k, len(incorrect))} of {len(incorrect)} total)")
    print('='*70)

    for i, result in enumerate(incorrect[:k]):
        item_id = result["item_id"]
        request_id = result["request_id"]
        req_idx = int(request_id[1])

        item = items_by_id.get(item_id, {})
        query, _ = format_query(item)

        # Shorten query for display
        if len(query) > max_query_len:
            query = query[:max_query_len] + "\n... [truncated]"

        print(f"\n--- Example {i+1} ---")
        print(f"Item ID:     {item_id}")
        print(f"Request:     {request_id}")
        print(f"Gold:        {result['gold']:2d}  |  Pred: {result['pred']:2d}")
        if "error" in result:
            print(f"Error:       {result['error']}")
        print(f"\nUser Request:\n  {USER_REQUESTS[req_idx]}")
        print(f"\nQuery (shortened):\n{query}")
        print()


# =============================================================================
# DUMMY METHOD FOR TESTING
# =============================================================================

def dummy_method(query: str, context: str) -> int:
    """
    Dummy method for testing the evaluation script.

    This is a placeholder that should be replaced with the actual LLM method.
    Returns a deterministic but meaningless prediction based on input hash.

    Args:
        query: The formatted restaurant query string
        context: The user request context

    Returns:
        int: A prediction in {-1, 0, 1}
    """
    # Simple deterministic logic for testing
    # Real implementation would call an LLM
    hash_val = hash(query + context)
    return (hash_val % 3) - 1  # Returns -1, 0, or 1


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM method on Yelp-style restaurant recommendation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_llm.py --data restaurants.jsonl --out results.jsonl
  python evaluate_llm.py --data restaurants.jsonl --out results.jsonl --limit 100
  python evaluate_llm.py --data restaurants.jsonl --out results.jsonl --print_examples 5

Output:
  - Console: Overall accuracy, per-request accuracy, confusion matrix
  - results.jsonl: Detailed per-item results
  - results_summary.json: Aggregated metrics

Note:
  Replace dummy_method() with your actual LLM method implementation.
  The method signature must be: method(query: str, context: str) -> int
        """
    )
    parser.add_argument(
        "--data",
        # required=True,
        default = "data.jsonl",
        help="Path to input JSONL data file"
    )
    parser.add_argument(
        "--out",
        default="results.jsonl",
        help="Path to output results JSONL file (default: results.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of restaurant items to process (optional)"
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=0,
        metavar="K",
        help="Print K incorrect examples with queries (default: 0)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("LLM EVALUATION SCRIPT")
    print("="*70)
    print(f"Data file:    {args.data}")
    print(f"Output file:  {args.out}")
    if args.limit:
        print(f"Item limit:   {args.limit}")
    print(f"Requests:     {len(USER_REQUESTS)} (R0, R1, R2)")
    print()

    # =========================================================================
    # IMPORTANT: Replace dummy_method with your actual LLM method
    #
    # The method signature must be:
    #   def method(query: str, context: str) -> int
    #
    # Where:
    #   - query: Formatted restaurant info (name, location, reviews)
    #   - context: User request describing what they're looking for
    #   - returns: -1 (not recommend), 0 (neutral), 1 (recommend)
    # =========================================================================
    method = dummy_method
    print("Using: dummy_method (replace with your LLM method)")
    print()

    # Run evaluation
    print("Running evaluation...")
    eval_results = evaluate_dataset(
        args.data,
        method,
        limit=args.limit,
        verbose=not args.quiet
    )

    # Print summary results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print(f"\nItems processed: {eval_results['items_processed']}")
    print(f"Total predictions: {eval_results['total_count']}")
    if eval_results['error_count'] > 0:
        print(f"Errors (defaulted to 0): {eval_results['error_count']}")

    print(f"\n{'─'*40}")
    print(f"OVERALL ACCURACY: {eval_results['overall_accuracy']:.4f}")
    print(f"  ({eval_results['total_correct']}/{eval_results['total_count']} correct)")
    print(f"{'─'*40}")

    print("\nPer-Request Accuracy:")
    for i, req_id in enumerate(["R0", "R1", "R2"]):
        acc = eval_results['per_request_accuracy'][req_id]
        correct = eval_results['per_request_correct'].get(req_id, 0)
        count = eval_results['per_request_counts'].get(req_id, 0)
        print(f"  {req_id}: {acc:.4f}  ({correct}/{count})")
        # Print abbreviated request description
        desc = USER_REQUESTS[i][:60] + "..." if len(USER_REQUESTS[i]) > 60 else USER_REQUESTS[i]
        print(f"      \"{desc}\"")

    # Print confusion matrix
    print_confusion_matrix(eval_results['confusion_matrix'])

    # Save detailed results to JSONL
    print(f"\nSaving detailed results to: {args.out}")
    with open(args.out, 'w', encoding='utf-8') as f:
        for record in eval_results['detailed_results']:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save summary to JSON
    summary_path = args.out.replace('.jsonl', '_summary.json')
    if summary_path == args.out:
        summary_path = args.out + '_summary.json'

    summary = {
        "overall_accuracy": eval_results['overall_accuracy'],
        "total_correct": eval_results['total_correct'],
        "total_count": eval_results['total_count'],
        "items_processed": eval_results['items_processed'],
        "error_count": eval_results['error_count'],
        "per_request_accuracy": eval_results['per_request_accuracy'],
        "per_request_correct": eval_results['per_request_correct'],
        "per_request_counts": eval_results['per_request_counts'],
        "confusion_matrix": {
            str(k): {str(k2): v2 for k2, v2 in v.items()}
            for k, v in eval_results['confusion_matrix'].items()
        },
        "user_requests": USER_REQUESTS,
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")

    # Print incorrect examples if requested
    if args.print_examples > 0:
        print_incorrect_examples(
            eval_results['detailed_results'],
            args.data,
            args.print_examples
        )

    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()