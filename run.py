#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from data.loader import format_query, format_ranking_query, load_groundtruth_scores
from utils.io import loadjl
from utils.parsing import normalize_pred, parse_index, parse_indices
from utils.output import print_results, print_ranking_results
from attack import apply_attacks, get_all_attack_names


def evaluate(items: list[dict], method: Callable, requests: list[dict], mode: str = "string") -> dict:
    """Run evaluation and collect results."""
    results = []
    req_ids = [r["id"] for r in requests]
    stats = {
        "total": 0, "correct": 0, "errors": 0,
        "per_request": {rid: {"total": 0, "correct": 0} for rid in req_ids},
        "confusion": {g: {p: 0 for p in [-1, 0, 1]} for g in [-1, 0, 1]},
    }

    total_pairs = len(items) * len(requests)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Evaluating...", total=total_pairs)

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
                    progress.update(task, advance=1)
                    continue
                gold = int(gold)

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
                progress.update(task, advance=1)

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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(pairs))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single, item, req, method, mode)
                       for item, req in pairs]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    stats = compute_stats(results, req_ids)
    return {"results": results, "stats": stats, "req_ids": req_ids}


# --- Ranking Evaluation ---

def compute_multi_k_stats(results: list[dict], k: int) -> dict:
    """Compute Hits@1 through Hits@k from results.

    Args:
        results: List of result dicts with 'gold_index' and 'pred_indices'
        k: Maximum k value to compute

    Returns:
        Dict with total, k, and hits_at dict for each level
    """
    total = len(results)
    hits_at = {}
    for j in range(1, k + 1):
        hits = sum(1 for r in results
                   if r["gold_index"] in r["pred_indices"][:j])
        hits_at[j] = {"hits": hits, "accuracy": hits / total if total else 0}
    return {
        "total": total,
        "k": k,
        "hits_at": hits_at,
    }


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth_scores: dict, mode: str = "string", k: int = 1) -> dict:
    """Evaluate using ranking (Top-K accuracy / Hits@K).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that outputs best index
        requests: List of requests
        groundtruth_scores: {request_id: {item_id: total_score}} from precomputed groundtruth
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 1 for top-1 accuracy)

    Returns:
        Dict with results and accuracy stats
    """
    results = []
    req_ids = [r["id"] for r in requests]

    # Build item index mapping (1-based)
    item_ids = [item.get("item_id") for item in items]
    item_id_to_index = {item_id: i + 1 for i, item_id in enumerate(item_ids)}

    # Format all items as a single query
    query, item_count = format_ranking_query(items, mode)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Ranking evaluation...", total=len(requests))

        for req in requests:
            req_id = req["id"]
            context = req["context"]

            # Get gold answer: item with highest score for this request
            scores = groundtruth_scores.get(req_id, {})
            if not scores:
                progress.update(task, advance=1)
                continue

            # Find gold index (item with highest score)
            gold_item_id = max(scores.keys(), key=lambda x: scores.get(x, 0))
            gold_index = item_id_to_index.get(gold_item_id, 0)
            gold_score = scores.get(gold_item_id, 0)

            # Get prediction from LLM
            try:
                response = method(query, context)
                if k == 1:
                    pred_indices = [parse_index(response, item_count)]
                else:
                    pred_indices = parse_indices(response, item_count, k)
            except Exception:
                pred_indices = []

            results.append({
                "request_id": req_id,
                "pred_indices": pred_indices,
                "gold_index": gold_index,
                "gold_item_id": gold_item_id,
                "gold_score": gold_score,
            })
            progress.update(task, advance=1)

    # Compute multi-K stats
    stats = compute_multi_k_stats(results, k)

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
    }


# --- Orchestration Helpers ---

def check_previous_run(args, experiment) -> dict | None:
    """Check for previous run and prompt user.

    Returns:
        Dict with results/stats if user chooses to reload, None otherwise
    """
    prev = experiment.find_previous_run(args.method, args.data)
    if not prev:
        return None

    prev_dir, prev_config = prev
    console = Console()
    console.print(f"\n[yellow]Found previous run with same method/data:[/yellow]")
    console.print(f"  [cyan]{prev_dir}[/cyan]")
    console.print(f"  timestamp: {prev_config.get('timestamp', 'unknown')}")

    response = input("Reload cached results? [Y/n]: ").strip().lower()
    if response in ("", "y", "yes"):
        results_path = prev_dir / "results.jsonl"
        results = loadjl(results_path)
        k = getattr(args, 'k', 5)
        stats = compute_multi_k_stats(results, k)
        return {"results": results, "stats": stats, "path": str(prev_dir)}

    return None


def run_evaluation_loop(args, items_clean, requests, method, experiment):
    """Evaluate dataset(s) - handles both single and multi-attack.

    Args:
        items_clean: Clean items (attacks will be applied in-memory)
        requests: List of request dicts
        method: Method callable
        experiment: ExperimentManager instance

    Returns:
        Dict with stats and attack_params for each attack variant
    """
    # Check for previous run (dev mode only)
    cached = check_previous_run(args, experiment)
    if cached:
        print_ranking_results({"results": cached["results"], "stats": cached["stats"]})
        return {
            "clean": {
                "stats": cached["stats"],
                "attack_params": {"attack": "none", "attack_type": None, "attack_config": None, "seed": None}
            }
        }

    # Check if ranking mode is enabled (default: True for top-1 accuracy)
    ranking_mode = getattr(args, 'ranking', True)

    # Use dict mode for methods that need schema access
    dict_mode_methods = {"anot", "anot_v2", "anot_v3", "anot_origin", "pal", "pot", "cot_table"}
    eval_mode = "dict" if args.method in dict_mode_methods else "string"
    k = getattr(args, 'k', 1)

    # Determine which attacks to run
    if args.attack == "all":
        attack_names = get_all_attack_names()
    elif args.attack in ("none", "clean", None, ""):
        attack_names = ["clean"]
    else:
        attack_names = [args.attack]

    all_results = {}
    for attack_name in attack_names:
        print(f"\n{'='*50}\nRunning: {attack_name}\n{'='*50}")

        # Apply attack (in-memory, temporary)
        attack_key = None if attack_name == "clean" else attack_name
        items, attack_params = apply_attacks(items_clean, attack_key, seed=args.seed)

        if ranking_mode:
            # Ranking evaluation (top-1 accuracy by default)
            try:
                groundtruth_scores = load_groundtruth_scores(args.selection_name)
                eval_out = evaluate_ranking(items, method, requests, groundtruth_scores, mode=eval_mode, k=k)
                print_ranking_results(eval_out)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("Falling back to per-item evaluation")
                ranking_mode = False

        if not ranking_mode:
            # Per-item evaluation (legacy mode)
            if args.parallel:
                eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
            else:
                eval_out = evaluate(items, method, requests, mode=eval_mode)
            print_results(eval_out["stats"], eval_out.get("req_ids"))

        # Store both stats and attack_params
        all_results[attack_name] = {
            "stats": eval_out["stats"],
            "attack_params": attack_params
        }

        filename = "results.jsonl" if len(attack_names) == 1 else f"results_{attack_name}.jsonl"
        result_path = experiment.save_results(eval_out["results"], filename)
        print(f"Results saved to {result_path}")

    return all_results


def save_final_config(args, all_results, experiment):
    """Construct and save the run configuration.

    Args:
        args: Parsed arguments
        all_results: Dict of {attack_name: {"stats": ..., "attack_params": ...}}
        experiment: ExperimentManager instance
    """
    # Unwrap if only one attack was run
    if len(all_results) == 1:
        result = next(iter(all_results.values()))
        final_stats = result["stats"]
        attack_params = result["attack_params"]
    else:
        # Multiple attacks - combine stats
        final_stats = {name: r["stats"] for name, r in all_results.items()}
        # Use the first attack's params as reference (for "all" mode)
        attack_params = next(iter(all_results.values()))["attack_params"]

    config = {
        "method": args.method,
        "defense": args.defense,
        "data": args.data,
        "selection": args.selection_name,
        "limit": args.limit,
        # Attack information - full params for reproducibility
        "attack": attack_params.get("attack", args.attack),
        "attack_type": attack_params.get("attack_type"),
        "attack_config": attack_params.get("attack_config"),
        "attack_seed": attack_params.get("seed"),
        "llm_config": {
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "stats": final_stats,
    }

    config_path = experiment.save_config(config)
    print(f"\nConfig saved to {config_path}")


def run_single(args, experiment, log):
    """Execute a single evaluation run.

    Args:
        args: Parsed command-line arguments
        experiment: ExperimentManager instance
        log: Logger instance

    Returns:
        Dict of results from evaluation
    """
    from data.loader import load_dataset
    from methods import get_method

    run_dir = experiment.setup()

    modestr = "BENCHMARK" if experiment.benchmark_mode else "development"
    log.info(f"Mode: {modestr}")
    log.info(f"Run directory: {run_dir}")

    # Load clean data (attacks applied later in run_evaluation_loop)
    dataset = load_dataset(args.data, args.selection_name, args.limit, review_limit=args.review_limit)
    log.info(f"\n{dataset}")

    # Select method
    method = get_method(args, run_dir)
    print(method)

    # Run evaluation - attacks applied here (in-memory, temporary)
    all_results = run_evaluation_loop(args, dataset.items, dataset.requests, method, experiment)

    # Finalize
    save_final_config(args, all_results, experiment)
    experiment.consolidate_debug_logs()

    return all_results
