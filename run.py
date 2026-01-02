#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any
import json

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from data.loader import load_dataset, filter_by_candidates, format_query, format_ranking_query
from utils.io import loadjl
from utils.parsing import normalize_pred, parse_index, parse_indices, parse_limit_spec
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


class ContextLengthExceeded(Exception):
    """Raised when LLM context length is exceeded."""
    pass


def evaluate_ranking_single(method, query, context: str, k: int,
                            req: dict, groundtruth: dict, item_count: int) -> dict | None:
    """Evaluate a single request (thread-safe helper).

    Args:
        method: LLM method instance
        query: Formatted query (all items)
        context: Request context/text
        k: Number of top predictions
        req: Request dict with 'id'
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        item_count: Total number of items for parsing

    Returns:
        Result dict or None if no ground truth

    Raises:
        ContextLengthExceeded: If the context length limit is exceeded
    """
    import os
    debug = os.environ.get("KNOT_DEBUG", "0") == "1"

    req_id = req["id"]
    gt = groundtruth.get(req_id)
    if not gt:
        return None

    response = None
    try:
        response = method.evaluate_ranking(query, context, k=k)
        pred_indices = parse_indices(response, item_count, k)

        # Debug: log when parsing fails
        if debug and not pred_indices and response:
            print(f"[DEBUG] {req_id}: No indices parsed from response (len={len(response)})")
            print(f"  Response preview: {response[:300]}...")

    except Exception as e:
        error_str = str(e)
        # Check for context length exceeded - stop early to save budget
        if "context_length_exceeded" in error_str or "too many tokens" in error_str.lower():
            print(f"\n[STOP] Context length exceeded at {req_id}. Stopping all requests.")
            raise ContextLengthExceeded(error_str)

        pred_indices = []
        # Log other exceptions in debug mode
        if debug:
            print(f"[DEBUG] {req_id}: Exception: {type(e).__name__}: {e}")

    return {
        "request_id": req_id,
        "pred_indices": pred_indices,
        "gold_idx": gt["gold_idx"],
        "gold_restaurant": gt["gold_restaurant"],
    }


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth: dict, mode: str = "string", k: int = 5,
                     parallel: bool = True, max_workers: int = 40) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that returns ranking via evaluate_ranking()
        requests: List of requests
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 5 for Hits@5)
        parallel: Whether to use parallel execution (default True)
        max_workers: Maximum number of worker threads (default 40)

    Returns:
        Dict with results and accuracy stats
    """
    req_ids = [r["id"] for r in requests]

    # Format all items as a single query
    query, item_count = format_ranking_query(items, mode)

    context_exceeded = False

    if parallel:
        # Parallel execution with ThreadPoolExecutor
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(f"Ranking evaluation (parallel, {max_workers} workers)...", total=len(requests))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        evaluate_ranking_single,
                        method, query,
                        req.get("context") or req.get("text", ""),
                        k, req, groundtruth, item_count
                    ): req
                    for req in requests
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except ContextLengthExceeded:
                        context_exceeded = True
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                    progress.update(task, advance=1)
    else:
        # Sequential execution
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Ranking evaluation (sequential)...", total=len(requests))

            for req in requests:
                context = req.get("context") or req.get("text", "")
                try:
                    result = evaluate_ranking_single(
                        method, query, context, k, req, groundtruth, item_count
                    )
                    if result:
                        results.append(result)
                except ContextLengthExceeded:
                    context_exceeded = True
                    break
                progress.update(task, advance=1)

    # Compute multi-K stats
    stats = compute_multi_k_stats(results, k)

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
        "context_exceeded": context_exceeded,
    }


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
    parallel = getattr(args, 'parallel', True)
    max_workers = getattr(args, 'max_concurrent', 40)

    # Ranking evaluation (default)
    mode_str = "parallel" if parallel else "sequential"
    print(f"\nRunning ranking evaluation (k={k}, {mode_str})...")
    eval_out = evaluate_ranking(
        dataset.items,
        method,
        dataset.requests,
        dataset.groundtruth,
        mode=eval_mode,
        k=k,
        parallel=parallel,
        max_workers=max_workers
    )
    print_ranking_results(eval_out["stats"], eval_out["results"])

    # Merge with existing results if partial run
    results_to_save = eval_out["results"]
    if experiment.partial:
        results_to_save = experiment.merge_results(results_to_save)
        # Recompute stats from merged results
        merged_stats = compute_multi_k_stats(results_to_save, k)
        print(f"\nMerged {len(eval_out['results'])} new + existing = {len(results_to_save)} total results")
        eval_out["stats"] = merged_stats

    # Save results
    result_path = experiment.save_results(results_to_save, "results.jsonl")
    print(f"\nResults saved to {result_path}")

    return {"stats": eval_out["stats"]}


def save_final_config(args, all_results, experiment):
    """Construct and save the run configuration."""
    config = {
        "method": args.method,
        "defense": getattr(args, "defense", False),
        "data": args.data,
        "k": getattr(args, "k", 5),
        "parallel": getattr(args, "parallel", True),
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

    # Load dataset (full - filtering happens below)
    dataset = load_dataset(
        args.data,
        review_limit=getattr(args, 'review_limit', None)
    )

    # Filter to top-N candidates if --candidates specified
    n_candidates = getattr(args, 'candidates', None)
    if n_candidates:
        dataset = filter_by_candidates(dataset, n_candidates)
        log.info(f"Filtered to top {n_candidates} candidates ({len(dataset.requests)} requests)")

    # Filter requests if --limit specified
    if args.limit:
        indices = parse_limit_spec(args.limit)
        all_requests = dataset.requests
        dataset.requests = [r for i, r in enumerate(all_requests) if i in indices]
        # Also filter groundtruth to only include filtered requests
        filtered_ids = {r["id"] for r in dataset.requests}
        dataset.groundtruth = {k: v for k, v in dataset.groundtruth.items() if k in filtered_ids}
        log.info(f"Filtered to {len(dataset.requests)} requests (indices: {indices[:5]}{'...' if len(indices) > 5 else ''})")

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


# --- Scaling Experiment ---

SCALE_POINTS = [10, 15, 20, 25, 30, 40, 50]


def run_scaling_experiment(args, log):
    """Run scaling experiment across multiple candidate counts.

    Tests how method performance degrades as number of candidates increases.
    Stops early when context length is exceeded.

    Args:
        args: Parsed arguments
        log: Logger instance
    """
    from utils.experiment import create_experiment
    from methods import get_method

    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT: {args.method}")
    print(f"Scale points: {SCALE_POINTS}")
    print(f"{'='*60}\n")

    results_table = []
    context_exceeded_at = None

    for n_candidates in SCALE_POINTS:
        if context_exceeded_at:
            # Already hit context limit, mark remaining as --
            results_table.append({
                "candidates": n_candidates,
                "requests": "--",
                "hits_at_1": "--",
                "hits_at_5": "--",
                "status": "skipped",
            })
            continue

        print(f"\n{'='*60}")
        print(f"Running with {n_candidates} candidates...")
        print(f"{'='*60}")

        # Set candidates for this run
        args.candidates = n_candidates

        # Create experiment (use dev mode for scaling to avoid benchmark conflicts)
        original_benchmark = args.benchmark
        args.benchmark = False  # Use dev mode
        experiment = create_experiment(args)
        run_dir = experiment.setup()

        # Load and filter dataset
        dataset = load_dataset(
            args.data,
            review_limit=getattr(args, 'review_limit', None)
        )
        dataset = filter_by_candidates(dataset, n_candidates)

        # Apply --limit if specified (for testing with fewer requests)
        if args.limit:
            indices = parse_limit_spec(args.limit)
            all_requests = dataset.requests
            dataset.requests = [r for i, r in enumerate(all_requests) if i in indices]
            filtered_ids = {r["id"] for r in dataset.requests}
            dataset.groundtruth = {k: v for k, v in dataset.groundtruth.items() if k in filtered_ids}

        log.info(f"Candidates: {n_candidates}, Items: {len(dataset.items)}, Requests: {len(dataset.requests)}")

        # Get method
        method = get_method(
            args.method,
            run_dir=str(run_dir),
            defense=getattr(args, 'defense', False)
        )

        # Run evaluation
        k = getattr(args, 'k', 5)
        eval_result = evaluate_ranking(
            dataset.items,
            method,
            dataset.requests,
            dataset.groundtruth,
            mode="string",
            k=k,
            parallel=args.parallel,
            max_workers=getattr(args, 'max_concurrent', 200)
        )

        # Check if context exceeded
        if eval_result.get("context_exceeded"):
            context_exceeded_at = n_candidates
            results_table.append({
                "candidates": n_candidates,
                "requests": len(dataset.requests),
                "hits_at_1": "--",
                "hits_at_5": "--",
                "status": "context_exceeded",
            })
            print(f"\n[STOP] Context limit exceeded at {n_candidates} candidates.")
        else:
            stats = eval_result["stats"]
            hits_at = stats.get("hits_at", {})
            h1 = hits_at.get(1, hits_at.get("1", {})).get("accuracy", 0)
            h5 = hits_at.get(5, hits_at.get("5", {})).get("accuracy", 0)

            results_table.append({
                "candidates": n_candidates,
                "requests": len(dataset.requests),
                "hits_at_1": f"{h1:.2%}",
                "hits_at_5": f"{h5:.2%}",
                "status": "ok",
            })

            # Print per-run results
            print_ranking_results(stats, eval_result["results"])

        # Restore benchmark mode
        args.benchmark = original_benchmark

    # Print final summary table
    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT SUMMARY: {args.method}")
    print(f"{'='*60}")
    print(f"\n{'Candidates':<12} {'Requests':<10} {'Hits@1':<10} {'Hits@5':<10} {'Status'}")
    print("-" * 55)
    for row in results_table:
        print(f"{row['candidates']:<12} {row['requests']:<10} {row['hits_at_1']:<10} {row['hits_at_5']:<10} {row['status']}")

    if context_exceeded_at:
        print(f"\nContext limit exceeded at {context_exceeded_at} candidates.")
        print("Remaining scale points marked as '--' (not run).")
