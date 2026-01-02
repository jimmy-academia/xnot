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


# --- Shuffle Utilities ---

def shuffle_gold_to_middle(items: list, gold_idx: int) -> tuple[list, dict]:
    """Move gold item to middle position.

    Args:
        items: List of items
        gold_idx: Original index of gold item (0-indexed)

    Returns:
        (shuffled_items, mapping) where mapping[shuffled_pos] = original_idx
    """
    n = len(items)
    middle = n // 2

    # Build new order: remove gold, insert at middle
    indices = list(range(n))
    indices.remove(gold_idx)
    indices.insert(middle, gold_idx)

    shuffled = [items[i] for i in indices]
    mapping = {new_pos: orig_idx for new_pos, orig_idx in enumerate(indices)}
    return shuffled, mapping


def apply_shuffle(items: list, gold_idx: int, strategy: str) -> tuple[list, dict, int]:
    """Apply shuffle strategy.

    Args:
        items: List of items
        gold_idx: Original index of gold item (0-indexed)
        strategy: "none", "middle", or "random"

    Returns:
        (shuffled_items, mapping, shuffled_gold_pos) where:
        - mapping[shuffled_pos] = original_idx
        - shuffled_gold_pos = position of gold in shuffled list
    """
    if strategy == "none":
        mapping = {i: i for i in range(len(items))}
        return items, mapping, gold_idx
    elif strategy == "middle":
        shuffled, mapping = shuffle_gold_to_middle(items, gold_idx)
        # Find gold's new position
        shuffled_gold_pos = [k for k, v in mapping.items() if v == gold_idx][0]
        return shuffled, mapping, shuffled_gold_pos
    elif strategy == "random":
        import random
        indices = list(range(len(items)))
        random.shuffle(indices)
        shuffled = [items[i] for i in indices]
        mapping = {new_pos: orig_idx for new_pos, orig_idx in enumerate(indices)}
        shuffled_gold_pos = [k for k, v in mapping.items() if v == gold_idx][0]
        return shuffled, mapping, shuffled_gold_pos
    else:
        # Default to no shuffle
        mapping = {i: i for i in range(len(items))}
        return items, mapping, gold_idx


def unmap_predictions(pred_indices: list[int], mapping: dict) -> list[int]:
    """Convert shuffled predictions back to original indices.

    Args:
        pred_indices: Predictions in shuffled space (1-indexed)
        mapping: mapping[shuffled_pos] = original_idx (0-indexed)

    Returns:
        Predictions in original space (1-indexed)
    """
    result = []
    for p in pred_indices:
        shuffled_pos = p - 1  # Convert to 0-indexed
        if shuffled_pos in mapping:
            result.append(mapping[shuffled_pos] + 1)  # Back to 1-indexed
    return result


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


def evaluate_ranking_single(method, items: list, mode: str, shuffle: str,
                            context: str, k: int, req: dict,
                            groundtruth: dict) -> dict | None:
    """Evaluate a single request (thread-safe helper).

    Args:
        method: LLM method instance
        items: All items (will be shuffled per request)
        mode: "string" or "dict" for formatting
        shuffle: Shuffle strategy ("none", "middle", "random")
        context: Request context/text
        k: Number of top predictions
        req: Request dict with 'id'
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}

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

    gold_idx = gt["gold_idx"]

    # Apply shuffle based on gold position
    shuffled_items, mapping, shuffled_gold_pos = apply_shuffle(items, gold_idx, shuffle)

    # Format query with shuffled items
    query, item_count = format_ranking_query(shuffled_items, mode)

    response = None
    shuffled_preds = []
    try:
        response = method.evaluate_ranking(query, context, k=k)
        shuffled_preds = parse_indices(response, item_count, k)

        # Debug: log when parsing fails
        if debug and not shuffled_preds and response:
            print(f"[DEBUG] {req_id}: No indices parsed from response (len={len(response)})")
            print(f"  Response preview: {response[:300]}...")

    except Exception as e:
        error_str = str(e)
        # Check for context length exceeded - stop early to save budget
        if "context_length_exceeded" in error_str or "too many tokens" in error_str.lower():
            print(f"\n[STOP] Context length exceeded at {req_id}. Stopping all requests.")
            raise ContextLengthExceeded(error_str)

        shuffled_preds = []
        # Log other exceptions in debug mode
        if debug:
            print(f"[DEBUG] {req_id}: Exception: {type(e).__name__}: {e}")

    # Map predictions back to original indices
    pred_indices = unmap_predictions(shuffled_preds, mapping)

    return {
        "request_id": req_id,
        "pred_indices": pred_indices,  # In original space
        "shuffled_preds": shuffled_preds,  # In shuffled space (for debugging)
        "gold_idx": gold_idx,
        "shuffled_gold_pos": shuffled_gold_pos + 1,  # 1-indexed for display
        "gold_restaurant": gt["gold_restaurant"],
    }


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth: dict, mode: str = "string", k: int = 5,
                     shuffle: str = "middle", parallel: bool = True,
                     max_workers: int = 40) -> dict:
    """Evaluate using ranking (Hits@K accuracy).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that returns ranking via evaluate_ranking()
        requests: List of requests
        groundtruth: {request_id: {"gold_restaurant": str, "gold_idx": int}}
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 5 for Hits@5)
        shuffle: Shuffle strategy ("none", "middle", "random") - default "middle"
        parallel: Whether to use parallel execution (default True)
        max_workers: Maximum number of worker threads (default 40)

    Returns:
        Dict with results and accuracy stats
    """
    req_ids = [r["id"] for r in requests]

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
                        method, items, mode, shuffle,
                        req.get("context") or req.get("text", ""),
                        k, req, groundtruth
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
                        method, items, mode, shuffle, context, k, req, groundtruth
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
    shuffle = getattr(args, 'shuffle', 'middle')
    parallel = getattr(args, 'parallel', True)
    max_workers = getattr(args, 'max_concurrent', 40)

    # Ranking evaluation (default)
    mode_str = "parallel" if parallel else "sequential"
    shuffle_str = f", shuffle={shuffle}" if shuffle != "none" else ""
    print(f"\nRunning ranking evaluation (k={k}, {mode_str}{shuffle_str})...")
    eval_out = evaluate_ranking(
        dataset.items,
        method,
        dataset.requests,
        dataset.groundtruth,
        mode=eval_mode,
        k=k,
        shuffle=shuffle,
        parallel=parallel,
        max_workers=max_workers
    )
    # Get current usage for display
    from utils.usage import get_usage_tracker
    usage_for_display = get_usage_tracker().get_summary()
    print_ranking_results(eval_out["stats"], eval_out["results"], usage_for_display)

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
    from utils.usage import get_usage_tracker

    # Get usage summary
    tracker = get_usage_tracker()
    usage_summary = tracker.get_summary()

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
        "usage": usage_summary,
    }

    config_path = experiment.save_config(config)
    print(f"Config saved to {config_path}")

    # Save detailed usage log
    usage_path = experiment.run_dir / "usage.jsonl"
    tracker.save_to_file(usage_path)
    print(f"Usage log saved to {usage_path}")


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
    from utils.usage import get_usage_tracker

    # Reset usage tracker at start of run
    tracker = get_usage_tracker()
    tracker.reset()

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


def load_existing_results(run_dir: Path, n_candidates: int) -> dict:
    """Load existing results for a candidate count.

    Returns:
        Dict of {request_id: result_dict}
    """
    results_path = run_dir / f"results_{n_candidates}.jsonl"
    if not results_path.exists():
        return {}

    existing = {}
    for line in results_path.read_text().strip().split("\n"):
        if line:
            r = json.loads(line)
            existing[r["request_id"]] = r
    return existing


def save_scaling_summary(run_dir: Path, results_table: list, k: int):
    """Save scaling experiment summary to JSON.

    Args:
        run_dir: Run directory path
        results_table: List of {candidates, requests, hits_at_1, hits_at_5, status}
        k: K value used for evaluation
    """
    summary = {
        "scale_points": SCALE_POINTS,
        "k": k,
        "results": results_table,
    }
    summary_path = run_dir / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScaling summary saved to {summary_path}")


def load_failed_scales(run_dir: Path) -> set:
    """Load scales that already hit context limit.

    Returns:
        Set of candidate counts that failed (context_exceeded or skipped)
    """
    summary_path = run_dir / "scaling_summary.json"
    if not summary_path.exists():
        return set()

    summary = json.loads(summary_path.read_text())
    failed = set()
    for r in summary.get("results", []):
        if r.get("status") in ("context_exceeded", "skipped"):
            failed.add(r["candidates"])
    return failed


def run_scaling_experiment(args, log):
    """Run scaling experiment across multiple candidate counts.

    Tests how method performance degrades as number of candidates increases.
    Stops early when context length is exceeded.

    Output structure:
        run_dir/
          results_10.jsonl
          results_15.jsonl
          ...
          usage_10.jsonl
          usage_15.jsonl
          ...
          scaling_summary.json
          config.json

    Args:
        args: Parsed arguments
        log: Logger instance
    """
    from utils.experiment import create_experiment
    from methods import get_method
    from utils.usage import get_usage_tracker

    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT: {args.method}")
    print(f"Scale points: {SCALE_POINTS}")
    print(f"{'='*60}\n")

    # Create experiment for target run (default: run_1)
    experiment = create_experiment(args)
    run_dir = experiment.setup()
    log.info(f"Run directory: {run_dir}")

    # Get tracker reference
    tracker = get_usage_tracker()

    # Load previously failed scales (context_exceeded or skipped)
    failed_scales = load_failed_scales(run_dir) if not args.force else set()
    if failed_scales:
        log.info(f"Skipping previously failed scales: {sorted(failed_scales)}")

    results_table = []
    context_exceeded_at = None
    k = getattr(args, 'k', 5)

    for n_candidates in SCALE_POINTS:
        # Skip previously failed scales (unless --force)
        if n_candidates in failed_scales:
            results_table.append({
                "candidates": n_candidates,
                "requests": "--",
                "hits_at_1": "--",
                "hits_at_5": "--",
                "status": "skipped",
            })
            continue

        if context_exceeded_at:
            # Already hit context limit in this run, mark remaining as --
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

        # Reset usage tracker for this scale point
        tracker.reset()

        # Load and filter dataset by candidates
        dataset = load_dataset(
            args.data,
            review_limit=getattr(args, 'review_limit', None)
        )
        dataset = filter_by_candidates(dataset, n_candidates)

        # Apply --limit if specified (filter to specific request indices)
        if args.limit:
            indices = parse_limit_spec(args.limit)
            all_requests = dataset.requests
            dataset.requests = [r for i, r in enumerate(all_requests) if i in indices]
            filtered_ids = {r["id"] for r in dataset.requests}
            dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in filtered_ids}

        if len(dataset.requests) == 0:
            log.info(f"Candidates: {n_candidates}, No valid requests (gold not in candidate set)")
            results_table.append({
                "candidates": n_candidates,
                "requests": 0,
                "hits_at_1": "--",
                "hits_at_5": "--",
                "status": "no_requests",
            })
            continue

        # Check if results already exist for this scale
        existing = load_existing_results(run_dir, n_candidates)
        if existing and not args.force:
            expected_ids = {r["id"] for r in dataset.requests}
            completed_ids = set(existing.keys())

            if expected_ids == completed_ids:
                # All requests complete - skip evaluation, just report
                log.info(f"Scale {n_candidates}: Already complete ({len(existing)} requests)")
                stats = compute_multi_k_stats(list(existing.values()), k)
                hits_at = stats.get("hits_at", {})
                h1 = hits_at.get(1, {}).get("accuracy", 0)
                h5 = hits_at.get(5, {}).get("accuracy", 0)
                results_table.append({
                    "candidates": n_candidates,
                    "requests": len(existing),
                    "hits_at_1": f"{h1:.2%}",
                    "hits_at_5": f"{h5:.2%}",
                    "status": "ok",
                })
                print_ranking_results(stats, list(existing.values()))
                continue

            # Partial results exist - only run missing requests
            missing_ids = expected_ids - completed_ids
            dataset.requests = [r for r in dataset.requests if r["id"] in missing_ids]
            dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in missing_ids}
            log.info(f"Scale {n_candidates}: Running {len(missing_ids)} missing requests")

        log.info(f"Candidates: {n_candidates}, Items: {len(dataset.items)}, Requests: {len(dataset.requests)}")

        # Get method
        method = get_method(
            args.method,
            run_dir=str(run_dir),
            defense=getattr(args, 'defense', False)
        )

        # Run evaluation
        shuffle = getattr(args, 'shuffle', 'middle')
        eval_result = evaluate_ranking(
            dataset.items,
            method,
            dataset.requests,
            dataset.groundtruth,
            mode="string",
            k=k,
            shuffle=shuffle,
            parallel=args.parallel,
            max_workers=getattr(args, 'max_concurrent', 200)
        )

        # Merge with existing results (or replace if --force)
        if args.force:
            existing = {}
        else:
            existing = load_existing_results(run_dir, n_candidates)
        for r in eval_result["results"]:
            existing[r["request_id"]] = r
        merged_results = list(existing.values())

        # Save merged results to candidate-specific file
        results_path = run_dir / f"results_{n_candidates}.jsonl"
        with open(results_path, "w") as f:
            for r in sorted(merged_results, key=lambda x: x.get("request_id", "")):
                f.write(json.dumps(r) + "\n")

        # Save usage for this scale point
        usage_path = run_dir / f"usage_{n_candidates}.jsonl"
        tracker.save_to_file(usage_path)

        # Recompute stats from merged results
        stats = compute_multi_k_stats(merged_results, k)

        # Check if context exceeded
        if eval_result.get("context_exceeded"):
            context_exceeded_at = n_candidates
            results_table.append({
                "candidates": n_candidates,
                "requests": len(merged_results),
                "hits_at_1": "--",
                "hits_at_5": "--",
                "status": "context_exceeded",
            })
            print(f"\n[STOP] Context limit exceeded at {n_candidates} candidates.")
        else:
            hits_at = stats.get("hits_at", {})
            h1 = hits_at.get(1, hits_at.get("1", {})).get("accuracy", 0)
            h5 = hits_at.get(5, hits_at.get("5", {})).get("accuracy", 0)

            results_table.append({
                "candidates": n_candidates,
                "requests": len(merged_results),
                "hits_at_1": f"{h1:.2%}",
                "hits_at_5": f"{h5:.2%}",
                "status": "ok",
            })

            # Print per-run results with usage
            usage_for_display = tracker.get_summary()
            print_ranking_results(stats, merged_results, usage_for_display)

    # Save scaling summary
    save_scaling_summary(run_dir, results_table, k)

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
