#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from data.loader import format_query, normalize_pred, load_attacked_data, get_available_attacks


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
                from methods.knot import set_current_ids
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


def evaluate_single(item: dict, req: dict, method: Callable, mode: str = "string") -> dict:
    """Evaluate a single item-request pair (thread-safe)."""
    item_id = item.get("item_id", "unknown")
    req_id = req["id"]
    context = req["context"]
    gold_answers = item.get("gold_labels") or item.get("final_answers", {})
    gold = gold_answers.get(req_id)
    if gold is None:
        return None

    query, _ = format_query(item, mode)
    try:
        pred = normalize_pred(method(query, context))
        error = False
    except Exception:
        pred = 0
        error = True

    return {
        "item_id": item_id,
        "request_id": req_id,
        "pred": pred,
        "gold": int(gold),
        "correct": pred == int(gold),
        "error": error,
    }


def compute_stats(results: list[dict], req_ids: list[str]) -> dict:
    """Aggregate stats from results list."""
    stats = {
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "errors": sum(1 for r in results if r.get("error", False)),
        "per_request": {rid: {"total": 0, "correct": 0} for rid in req_ids},
        "confusion": {g: {p: 0 for p in [-1, 0, 1]} for g in [-1, 0, 1]},
    }
    for r in results:
        rid = r["request_id"]
        if rid in stats["per_request"]:
            stats["per_request"][rid]["total"] += 1
            stats["per_request"][rid]["correct"] += r["correct"]
        stats["confusion"][r["gold"]][r["pred"]] += 1
    return stats


def evaluate_parallel(items: list[dict], method: Callable, requests: list[dict],
                      mode: str = "string", max_workers: int = 40) -> dict:
    """Parallel version of evaluate() - all item-request pairs at once."""
    req_ids = [r["id"] for r in requests]
    pairs = [(item, req) for item in items for req in requests]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single, item, req, method, mode)
                   for item, req in pairs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    stats = compute_stats(results, req_ids)
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


# --- Orchestration Helpers ---

def get_attacks_list(args):
    """Determine which attacks to run from pre-generated datasets."""
    available = get_available_attacks()
    if args.attack == "all":
        return available
    elif args.attack == "none":
        return ["clean"]
    else:
        if args.attack not in available:
            raise ValueError(f"Attack '{args.attack}' not found. Available: {available}")
        return [args.attack]


def run_evaluation_loop(args, items_clean, requests, method, attacks, experiment):
    """Core loop - load pre-generated attacked data and evaluate."""
    # Determine eval_mode for variable substitution
    approach = getattr(args, 'knot_approach', 'base')
    if args.method == "knot" and approach in ("v4", "v5"):
        eval_mode = "dict"
    else:
        eval_mode = args.mode if args.method == "knot" else "string"

    all_stats = {}

    for attack_name in attacks:
        print(f"\n{'='*50}\nRunning: {attack_name}\n{'='*50}")

        # Load pre-generated attacked data (or clean data)
        items = load_attacked_data(attack_name, args.data, args.limit)

        # Evaluate
        if args.parallel:
            eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
        else:
            eval_out = evaluate(items, method, requests, mode=eval_mode)

        # Print and save
        print_results(eval_out["stats"], eval_out.get("req_ids"))
        all_stats[attack_name] = eval_out["stats"]

        filename = "results.jsonl" if len(attacks) == 1 else f"results_{attack_name}.jsonl"
        result_path = experiment.save_results(eval_out["results"], filename)
        print(f"Results saved to {result_path}")

    return all_stats


def save_final_config(args, attacks, stats, experiment):
    """Construct and save the run configuration."""
    config = {
        "method": args.method,
        "mode": args.mode if args.method == "knot" else None,
        "approach": getattr(args, 'knot_approach', None) if args.method == "knot" else None,
        "defense": args.defense,
        "data": args.data,
        "requests": args.requests,
        "limit": args.limit,
        "attack": args.attack,
        "attacks_run": attacks,
        "llm_config": {
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        # Unwrap stats if only one attack was run
        "stats": stats if len(attacks) > 1 else stats.get("clean", stats),
    }

    config_path = experiment.save_config(config)
    print(f"\nConfig saved to {config_path}")
