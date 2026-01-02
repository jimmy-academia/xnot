#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any
import json

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from data.loader import load_dataset, format_query, format_ranking_query
from utils.io import loadjl
from utils.parsing import normalize_pred, parse_index, parse_indices
from utils.aggregate import print_results, print_ranking_results


# --- Ranking Evaluation ---

def compute_multi_k_stats(results: list[dict], k: int) -> dict:
    """Compute Hits@1 through Hits@k from results.

    Args:
        results: List of result dicts with 'gold_idx' and 'pred_indices'
        k: Maximum k value to compute

    Returns:
        Dict with total, k, and hits_at dict for each level
    """
    total = len(results)
    hits_at = {}
    for j in range(1, k + 1):
        # pred_indices is 1-indexed, gold_idx is 0-indexed
        # Convert gold_idx to 1-indexed for comparison
        hits = sum(1 for r in results
                   if (r["gold_idx"] + 1) in r["pred_indices"][:j])
        hits_at[j] = {"hits": hits, "accuracy": hits / total if total else 0}
    return {
        "total": total,
        "k": k,
        "hits_at": hits_at,
    }


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth: dict, mode: str = "string", k: int = 5) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that returns ranking via evaluate_ranking()
        requests: List of requests
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 5 for Hits@5)

    Returns:
        Dict with results and accuracy stats
    """
    results = []
    req_ids = [r["id"] for r in requests]

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
            context = req.get("context") or req.get("text", "")

            # Get gold answer from groundtruth
            gt = groundtruth.get(req_id)
            if not gt:
                progress.update(task, advance=1)
                continue

            gold_idx = gt["gold_idx"]  # 0-indexed
            gold_restaurant = gt["gold_restaurant"]

            # Get prediction from LLM method
            try:
                # Method should have evaluate_ranking(query, context, k)
                response = method.evaluate_ranking(query, context, k=k)
                pred_indices = parse_indices(response, item_count, k)
            except Exception as e:
                print(f"Error evaluating {req_id}: {e}")
                pred_indices = []

            results.append({
                "request_id": req_id,
                "pred_indices": pred_indices,
                "gold_idx": gold_idx,
                "gold_restaurant": gold_restaurant,
            })
            progress.update(task, advance=1)

    # Compute multi-K stats
    stats = compute_multi_k_stats(results, k)

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
    }


# --- Per-Item Evaluation (legacy) ---

def evaluate_single(item: dict, req: dict, method: Callable, mode: str = "string") -> dict:
    """Evaluate a single item-request pair (thread-safe)."""
    item_id = item.get("item_id", "unknown")
    req_id = req["id"]
    context = req.get("context") or req.get("text", "")

    query, _ = format_query(item, mode)
    try:
        pred = normalize_pred(method.evaluate(query, context))
        error = False
    except Exception:
        pred = 0
        error = True

    return {
        "item_id": item_id,
        "request_id": req_id,
        "pred": pred,
        "correct": None,  # No gold label in ranking mode
        "error": error,
    }


def compute_stats(results: list[dict], req_ids: list[str]) -> dict:
    """Aggregate stats from results list."""
    stats = {
        "total": len(results),
        "errors": sum(1 for r in results if r.get("error", False)),
        "per_request": {rid: {"total": 0} for rid in req_ids},
    }
    for r in results:
        rid = r["request_id"]
        if rid in stats["per_request"]:
            stats["per_request"][rid]["total"] += 1
    return stats


def evaluate_parallel(items: list[dict], method: Callable, requests: list[dict],
                      mode: str = "string", max_workers: int = 40) -> dict:
    """Parallel version of per-item evaluation."""
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


# --- Orchestration ---

def run_evaluation_loop(args, dataset, method, experiment):
    """Run evaluation on the dataset.

    Args:
        args: Parsed arguments
        dataset: Dataset object with items, requests, groundtruth
        method: Method instance
        experiment: ExperimentManager instance

    Returns:
        Dict with stats
    """
    # Use dict mode for methods that need structured access
    dict_mode_methods = {"anot", "weaver"}
    eval_mode = "dict" if args.method in dict_mode_methods else "string"
    k = getattr(args, 'k', 5)

    # Ranking evaluation (default)
    print(f"\nRunning ranking evaluation (k={k})...")
    eval_out = evaluate_ranking(
        dataset.items,
        method,
        dataset.requests,
        dataset.groundtruth,
        mode=eval_mode,
        k=k
    )
    print_ranking_results(eval_out["stats"])

    # Save results
    result_path = experiment.save_results(eval_out["results"], "results.jsonl")
    print(f"\nResults saved to {result_path}")

    return {"stats": eval_out["stats"]}


def save_final_config(args, all_results, experiment):
    """Construct and save the run configuration."""
    config = {
        "method": args.method,
        "defense": getattr(args, "defense", False),
        "data": args.data,
        "limit": args.limit,
        "k": getattr(args, "k", 5),
        "llm_config": {
            "provider": getattr(args, "provider", "openai"),
            "model": getattr(args, "model", None),
            "temperature": getattr(args, "temperature", 0.0),
            "max_tokens": getattr(args, "max_tokens", 1024),
        },
        "stats": all_results.get("stats", {}),
    }

    config_path = experiment.save_config(config)
    print(f"Config saved to {config_path}")


def run_single(args, experiment, log):
    """Execute a single evaluation run.

    Args:
        args: Parsed command-line arguments
        experiment: ExperimentManager instance
        log: Logger instance

    Returns:
        Dict of results from evaluation
    """
    from methods import get_method

    run_dir = experiment.setup()

    modestr = "BENCHMARK" if experiment.benchmark_mode else "development"
    log.info(f"Mode: {modestr}")
    log.info(f"Run directory: {run_dir}")

    # Load dataset
    dataset = load_dataset(
        args.data,
        limit=args.limit,
        review_limit=getattr(args, 'review_limit', None)
    )
    log.info(f"\n{dataset}")

    # Get method instance
    method = get_method(
        args.method,
        run_dir=str(run_dir),
        defense=getattr(args, 'defense', False)
    )
    print(f"\nMethod: {method}")

    # Run evaluation
    all_results = run_evaluation_loop(args, dataset, method, experiment)

    # Finalize
    save_final_config(args, all_results, experiment)
    experiment.consolidate_debug_logs()

    return all_results
