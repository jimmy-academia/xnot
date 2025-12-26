#!/usr/bin/env python3
"""LLM evaluation for restaurant recommendation dataset."""

import json
import argparse
from collections import defaultdict
from typing import Any, Callable

USER_REQUESTS = None  # Loaded from requests.json or default

DEFAULT_REQUESTS = [
    {"id": "R0", "context": "I'm in a hurry and need quick service. Is the wait time reasonable?"},
    {"id": "R1", "context": "I've heard mixed things. Is this place consistent in quality?"},
    {"id": "R2", "context": "Planning a special dinner date. Good for romantic occasions?"},
    {"id": "R3", "context": "Is it worth the price? Looking for good value, not necessarily cheap."},
    {"id": "R4", "context": "I care more about food quality than service. How's the food?"},
]


def load_requests(path: str = "requests.json") -> list[dict]:
    """Load user requests from JSON file.

    Supports both simple format (id, context) and complex format (id, text, structure).
    For complex format, 'text' is used as 'context'.
    """
    try:
        with open(path) as f:
            requests = json.load(f)
            # Normalize complex format to have 'context' field
            for req in requests:
                if "text" in req and "context" not in req:
                    req["context"] = req["text"]
            return requests
    except FileNotFoundError:
        return DEFAULT_REQUESTS


def normalize_pred(raw: Any) -> int:
    """Normalize prediction to {-1, 0, 1}."""
    if raw is None:
        raise ValueError("Prediction is None")
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw in {-1, 0, 1}:
            return raw
        raise ValueError(f"Invalid int: {raw}")
    if isinstance(raw, bool):
        return 1 if raw else -1
    if isinstance(raw, float):
        return -1 if raw <= -0.5 else (1 if raw >= 0.5 else 0)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"-1", "0", "1"}:
            return int(s)
        if any(p in s for p in ["not recommend", "don't recommend", "avoid", "reject"]):
            return -1
        if any(p in s for p in ["recommend", "yes", "suitable", "good", "great"]):
            return 1
        if any(p in s for p in ["neutral", "uncertain", "maybe", "mixed"]):
            return 0
        for tok in s.replace(",", " ").replace(":", " ").split():
            tok = tok.strip("()[]{}.,;:")
            if tok in {"-1", "0", "1"}:
                return int(tok)
    raise ValueError(f"Cannot normalize: {repr(raw)}")


def format_query(item: dict, mode: str = "string"):
    """Format restaurant item as query. Excludes ground-truth labels."""
    reviews = item.get("item_data", [])

    if mode == "dict":
        # Return clean dict for structured access
        return {
            "item_name": item.get("item_name", "Unknown"),
            "city": item.get("city", "Unknown"),
            "neighborhood": item.get("neighborhood", "Unknown"),
            "price_range": item.get("price_range", "Unknown"),
            "cuisine": item.get("cuisine", []),
            "item_data": [{"review_id": r.get("review_id", ""), "review": r.get("review", "")}
                          for r in reviews]
        }, len(reviews)

    # String mode (default)
    parts = [
        "Restaurant:",
        f"Name: {item.get('item_name', 'Unknown')}",
        f"City: {item.get('city', 'Unknown')}",
        f"Neighborhood: {item.get('neighborhood', 'Unknown')}",
        f"Price: {item.get('price_range', 'Unknown')}",
        f"Cuisine: {', '.join(item.get('cuisine', [])) or 'Unknown'}",
        "",
        "Reviews:",
    ]
    for r in reviews:
        parts.append(f"[{r.get('review_id', 'unknown')}] {r.get('review', '')}")
    return "\n".join(parts), len(reviews)


def load_data(path: str, limit: int = None) -> list[dict]:
    """Load JSONL data file."""
    items = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if limit and len(items) >= limit:
                    break
    return items


def evaluate(items: list[dict], method: Callable, requests: list[dict], mode: str = "string") -> dict:
    """Run evaluation and collect results."""
    results = []
    req_ids = [r["id"] for r in requests]
    stats = {
        "total": 0, "correct": 0, "errors": 0,
        "per_request": {rid: {"total": 0, "correct": 0} for rid in req_ids},
        "confusion": {g: {p: 0 for p in [-1, 0, 1]} for g in [-1, 0, 1]},
    }

    for item in items:
        item_id = item.get("item_id", "unknown")
        query, num_reviews = format_query(item, mode)
        # Support both old (final_answers) and new (gold_labels) format
        gold_answers = item.get("gold_labels") or item.get("final_answers", {})

        for req in requests:
            req_id = req["id"]
            context = req["context"]
            gold = gold_answers.get(req_id)
            if gold is None:
                continue
            gold = int(gold)

            # Set IDs for knot logging (if enabled)
            try:
                from knot import set_current_ids
                set_current_ids(item_id, req_id)
            except ImportError:
                pass

            try:
                pred = normalize_pred(method(query, context))
            except Exception as e:
                pred = 0
                stats["errors"] += 1

            correct = pred == gold
            stats["total"] += 1
            stats["correct"] += correct
            stats["per_request"][req_id]["total"] += 1
            stats["per_request"][req_id]["correct"] += correct
            stats["confusion"][gold][pred] += 1

            results.append({
                "item_id": item_id,
                "request_id": req_id,
                "pred": pred,
                "gold": gold,
                "correct": correct,
            })

    return {"results": results, "stats": stats, "req_ids": req_ids}


def print_results(stats: dict, req_ids: list[str] = None):
    """Print evaluation summary."""
    total, correct = stats["total"], stats["correct"]
    acc = correct / total if total else 0
    print(f"\nOverall: {acc:.4f} ({correct}/{total})")

    print("\nPer-request:")
    req_ids = req_ids or list(stats["per_request"].keys())
    for req_id in req_ids:
        if req_id in stats["per_request"]:
            r = stats["per_request"][req_id]
            acc = r["correct"] / r["total"] if r["total"] else 0
            print(f"  {req_id}: {acc:.4f} ({r['correct']}/{r['total']})")

    print("\nConfusion (rows=gold, cols=pred):")
    print("       -1    0    1")
    for g in [-1, 0, 1]:
        row = stats["confusion"][g]
        print(f"  {g:2d}  {row[-1]:4d} {row[0]:4d} {row[1]:4d}")


def dummy_method(query: str, context: str) -> int:
    """Placeholder method. Replace with actual LLM method."""
    return (hash(query + context) % 3) - 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")
    parser.add_argument("--data", default="data.jsonl", help="Input JSONL file")
    parser.add_argument("--requests", default="requests.json", help="User requests JSON file")
    parser.add_argument("--out", default="results.jsonl", help="Output results file")
    parser.add_argument("--limit", type=int, help="Limit items to process")
    parser.add_argument("--method", choices=["cot", "not", "knot", "dummy"], default="dummy", help="Method to use")
    parser.add_argument("--mode", choices=["string", "dict"], default="string", help="Input mode for knot")
    parser.add_argument("--knot-approach", choices=["base", "voting", "iterative", "divide"], default="base",
                        help="Approach for knot (base=default, voting=self-consistency, iterative=plan refinement, divide=divide-conquer)")
    args = parser.parse_args()

    # Load data and requests
    items = load_data(args.data, args.limit)
    requests = load_requests(args.requests)
    print(f"Loaded {len(items)} items from {args.data}")
    print(f"Loaded {len(requests)} requests")

    # Select method
    if args.method == "cot":
        from cot import method
    elif args.method == "not":
        from rnot import method
    elif args.method == "knot":
        from knot import create_method
        approach = getattr(args, 'knot_approach', 'base')
        method = create_method(mode=args.mode, approach=approach)
    else:
        method = dummy_method
    print(f"Using method: {args.method}" + (f" (mode={args.mode}, approach={getattr(args, 'knot_approach', 'base')})" if args.method == "knot" else ""))

    eval_out = evaluate(items, method, requests, mode=args.mode if args.method == "knot" else "string")

    # Print results
    print_results(eval_out["stats"], eval_out.get("req_ids"))

    # Save results
    with open(args.out, 'w') as f:
        for r in eval_out["results"]:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
