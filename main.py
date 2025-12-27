#!/usr/bin/env python3
"""LLM evaluation for restaurant recommendation dataset."""

import json
import argparse
import glob
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from data.loader import load_data, load_requests, format_query, normalize_pred

RESULTS_DIR = Path("results")

# Attack configurations: name -> (attack_type, kwargs)
ATTACK_CONFIGS = {
    "typo_10": ("typo", {"rate": 0.1}),
    "typo_20": ("typo", {"rate": 0.2}),
    "inject_override": ("injection", {"injection_type": "override", "target": 1}),
    "inject_fake_sys": ("injection", {"injection_type": "fake_system", "target": 1}),
    "inject_hidden": ("injection", {"injection_type": "hidden", "target": 1}),
    "inject_manipulation": ("injection", {"injection_type": "manipulation", "target": 1}),
    "fake_positive": ("fake_review", {"sentiment": "positive"}),
    "fake_negative": ("fake_review", {"sentiment": "negative"}),
}
ATTACK_CHOICES = ["none"] + list(ATTACK_CONFIGS.keys()) + ["all"]


def get_next_run_number() -> int:
    """Scan results/ and find the next available run number."""
    existing = glob.glob(str(RESULTS_DIR / "[0-9]*_*/"))
    if not existing:
        return 1
    numbers = []
    for p in existing:
        folder_name = Path(p).name
        try:
            num = int(folder_name.split("_")[0])
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def create_run_dir(run_name: str) -> Path:
    """Create a numbered run directory and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_num = get_next_run_number()
    run_dir = RESULTS_DIR / f"{run_num}_{run_name}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


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


def run_attack(attack_name: str, items_clean: list[dict], method: Callable,
               requests: list[dict], mode: str, run_dir: Path,
               attack_configs: dict, apply_attack_fn) -> tuple:
    """Run evaluation for a single attack (can run in parallel with other attacks)."""
    print(f"Starting: {attack_name}")

    if attack_name == "clean":
        items = items_clean
    else:
        attack_type, kwargs = attack_configs[attack_name]
        items = apply_attack_fn(items_clean, attack_type, **kwargs)

    # Use parallel evaluation for item-request pairs
    eval_out = evaluate_parallel(items, method, requests, mode)

    # Save results (sorted for deterministic output)
    result_path = run_dir / f"results_{attack_name}.jsonl"
    with open(result_path, 'w') as f:
        for r in sorted(eval_out["results"], key=lambda x: (x["item_id"], x["request_id"])):
            f.write(json.dumps(r) + '\n')

    stats = eval_out["stats"]
    acc = stats["correct"] / stats["total"] if stats["total"] else 0
    print(f"Completed: {attack_name} - {acc:.4f} ({stats['correct']}/{stats['total']})")
    return attack_name, eval_out


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")
    parser.add_argument("--data", default="data/processed/real_data.jsonl", help="Input JSONL file")
    parser.add_argument("--requests", default="data/requests/complex_requests.json", help="User requests JSON file")
    parser.add_argument("--run-name", help="Name for this run (creates results/{N}_{run-name}/)")
    parser.add_argument("--out", help="Output results file (default: auto in run dir)")
    parser.add_argument("--limit", type=int, help="Limit items to process")
    parser.add_argument("--method", choices=["cot", "not", "knot", "dummy"], default="dummy", help="Method to use")
    parser.add_argument("--mode", choices=["string", "dict"], default="string", help="Input mode for knot")
    parser.add_argument("--knot-approach", choices=["base", "voting", "iterative", "divide", "v4"], default="base",
                        help="Approach for knot (base=default, voting=self-consistency, iterative=plan refinement, divide=divide-conquer, v4=hierarchical planning)")
    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none",
                        help="Attack type to apply (none=clean, all=run all attacks)")
    parser.add_argument("--defense", action="store_true",
                        help="Enable defense prompts (attack-resistant mode)")
    parser.add_argument("--max-concurrent", type=int, default=500,
                        help="Max concurrent API calls (default=500, safe for Tier 5)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution (all attacks + all item-request pairs)")
    args = parser.parse_args()

    # Initialize rate limiter
    from llm import init_rate_limiter
    init_rate_limiter(args.max_concurrent)

    # Create run directory
    run_name = args.run_name or args.method
    run_dir = create_run_dir(run_name)
    out_path = Path(args.out) if args.out else run_dir / "results.jsonl"
    print(f"Run directory: {run_dir}")

    # Load data and requests
    items_clean = load_data(args.data, args.limit)
    requests = load_requests(args.requests)
    print(f"Loaded {len(items_clean)} items from {args.data}")
    print(f"Loaded {len(requests)} requests")

    # Select method
    from methods import get_method
    approach = getattr(args, 'knot_approach', 'base')
    method = get_method(args.method, mode=args.mode, approach=approach,
                        defense=args.defense, run_dir=str(run_dir))
    defense_str = " +defense" if args.defense else ""
    print(f"Using method: {args.method}" + (f" (mode={args.mode}, approach={approach})" if args.method == "knot" else "") + defense_str)

    # Determine which attacks to run
    if args.attack == "all":
        attacks_to_run = ["clean"] + list(ATTACK_CONFIGS.keys())
    elif args.attack == "none":
        attacks_to_run = ["clean"]
    else:
        attacks_to_run = [args.attack]

    # Import attack functions if needed
    if args.attack != "none":
        from attack import apply_attack
    else:
        apply_attack = None

    eval_mode = args.mode if args.method == "knot" else "string"
    all_stats = {}

    if args.parallel and len(attacks_to_run) > 1:
        # PARALLEL: Run all attacks concurrently
        print(f"\n{'='*50}")
        print(f"Running {len(attacks_to_run)} attacks in PARALLEL (max {args.max_concurrent} concurrent API calls)")
        print("=" * 50)

        with ThreadPoolExecutor(max_workers=len(attacks_to_run)) as executor:
            futures = {
                executor.submit(run_attack, name, items_clean, method, requests,
                                eval_mode, run_dir, ATTACK_CONFIGS, apply_attack): name
                for name in attacks_to_run
            }

            for future in as_completed(futures):
                attack_name, eval_out = future.result()
                all_stats[attack_name] = eval_out["stats"]
                print_results(eval_out["stats"], eval_out.get("req_ids"))

    else:
        # SEQUENTIAL: Run attacks one by one (original behavior)
        for attack_name in attacks_to_run:
            print(f"\n{'='*50}")
            print(f"Running: {attack_name}")
            print("=" * 50)

            # Apply attack (or use clean data)
            if attack_name == "clean":
                items = items_clean
            else:
                attack_type, attack_kwargs = ATTACK_CONFIGS[attack_name]
                items = apply_attack(items_clean, attack_type, **attack_kwargs)

            # Evaluate (use parallel for item-requests if --parallel, else sequential)
            if args.parallel:
                eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
            else:
                eval_out = evaluate(items, method, requests, mode=eval_mode)

            # Print results
            print_results(eval_out["stats"], eval_out.get("req_ids"))
            all_stats[attack_name] = eval_out["stats"]

            # Save results
            if len(attacks_to_run) == 1:
                result_filename = "results.jsonl"
            else:
                result_filename = f"results_{attack_name}.jsonl"
            result_path = run_dir / result_filename
            with open(result_path, 'w') as f:
                for r in sorted(eval_out["results"], key=lambda x: (x["item_id"], x["request_id"])):
                    f.write(json.dumps(r) + '\n')
            print(f"Results saved to {result_path}")

    # Save run config
    config = {
        "timestamp": datetime.now().isoformat(),
        "method": args.method,
        "mode": args.mode if args.method == "knot" else None,
        "approach": getattr(args, 'knot_approach', None) if args.method == "knot" else None,
        "defense": args.defense,
        "data": args.data,
        "requests": args.requests,
        "limit": args.limit,
        "attack": args.attack,
        "attacks_run": attacks_to_run,
        "stats": all_stats if len(attacks_to_run) > 1 else all_stats.get("clean", all_stats),
    }
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Consolidate debug logs if v4 was used
    if args.method == "knot" and getattr(args, 'knot_approach', 'base') == "v4":
        try:
            from utils.logger import consolidate_logs
            consolidate_logs(str(run_dir))
            print(f"Debug logs consolidated to {run_dir}/debug/")
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not consolidate debug logs: {e}")


if __name__ == "__main__":
    main()
