#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any
import json

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.table import Table

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
    req_id = req["id"]
    gt = groundtruth.get(req_id)
    if not gt:
        return None

    gold_idx = gt["gold_idx"]

    # Apply shuffle based on gold position
    shuffled_items, mapping, shuffled_gold_pos = apply_shuffle(items, gold_idx, shuffle)

    # Format query with shuffled items
    query, item_count = format_ranking_query(shuffled_items, mode)

    # Track per-request token usage
    from utils.usage import get_usage_tracker
    tracker = get_usage_tracker()
    start_idx = len(tracker.get_records())

    response = None
    shuffled_preds = []
    try:
        # Only pass request_id if method accepts it (e.g., ANoT)
        import inspect
        sig = inspect.signature(method.evaluate_ranking)
        if 'request_id' in sig.parameters:
            response = method.evaluate_ranking(query, context, k=k, request_id=req_id)
        else:
            response = method.evaluate_ranking(query, context, k=k)
        shuffled_preds = parse_indices(response, item_count, k)

    except Exception as e:
        error_str = str(e)
        # Check for context length exceeded - stop early to save budget
        if "context_length_exceeded" in error_str or "too many tokens" in error_str.lower():
            raise ContextLengthExceeded(error_str)
        shuffled_preds = []

    # Compute usage for this request
    request_records = tracker.get_records()[start_idx:]
    request_prompt_tokens = sum(r['prompt_tokens'] for r in request_records)
    request_completion_tokens = sum(r['completion_tokens'] for r in request_records)
    request_tokens = request_prompt_tokens + request_completion_tokens
    request_cost = sum(r['cost_usd'] for r in request_records)
    request_latency = sum(r['latency_ms'] for r in request_records)

    # Map predictions back to original indices
    pred_indices = unmap_predictions(shuffled_preds, mapping)

    return {
        "request_id": req_id,
        "pred_indices": pred_indices,  # In original space
        "shuffled_preds": shuffled_preds,  # In shuffled space (for debugging)
        "gold_idx": gold_idx,
        "shuffled_gold_pos": shuffled_gold_pos + 1,  # 1-indexed for display
        "gold_restaurant": gt["gold_restaurant"],
        # Per-request usage
        "prompt_tokens": request_prompt_tokens,
        "completion_tokens": request_completion_tokens,
        "tokens": request_tokens,
        "cost_usd": request_cost,
        "latency_ms": request_latency,
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
    # Start rich Live display if method supports it (e.g., ANoT)
    has_rich_display = hasattr(method, 'start_display') and hasattr(method, 'stop_display')
    if has_rich_display:
        title = f"ANoT: {len(items)} candidates, k={k}"
        method.start_display(title=title, total=len(requests), requests=requests)

    try:
        return _evaluate_ranking_inner(
            items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display
        )
    finally:
        if has_rich_display:
            method.stop_display()


def _run_with_progress(generator, has_rich_display: bool, description: str, total: int):
    """Run a generator with optional progress display.

    Args:
        generator: Generator that yields progress increments
        has_rich_display: If True, skip Progress bar (method has own display)
        description: Progress bar description
        total: Total number of items for progress bar
    """
    if has_rich_display:
        for _ in generator:
            pass
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(description, total=total)
            for _ in generator:
                progress.update(task, advance=1)


def _evaluate_ranking_inner(items, method, requests, groundtruth, mode, k, shuffle, parallel, max_workers, has_rich_display):
    """Inner implementation of evaluate_ranking (wrapped by display context)."""
    req_ids = [r["id"] for r in requests]
    context_exceeded = False
    results = []

    if parallel:
        def run_eval():
            nonlocal context_exceeded
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
                        for f in futures:
                            f.cancel()
                        break
                    yield 1

        description = f"Ranking evaluation (parallel, {max_workers} workers)..."
    else:
        def run_eval():
            nonlocal context_exceeded
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
                    return
                yield 1

        description = "Ranking evaluation (sequential)..."

    _run_with_progress(run_eval(), has_rich_display, description, len(requests))

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
    print(f"\nRunning ranking evaluation (k={k}, {mode_str}{shuffle_str})...", flush=True)
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
    # Only print per-run results if not in benchmark mode (aggregated summary shown at end)
    if not getattr(args, 'benchmark', False):
        show_details = getattr(args, 'full', False)
        print_ranking_results(eval_out["stats"], eval_out["results"], usage_for_display, show_details=show_details)

    # Merge with existing results if resuming
    n_candidates = getattr(args, 'candidates', None) or len(dataset.items)
    results_filename = f"results_{n_candidates}.jsonl"  # Always use results_{n}.jsonl
    results_to_save = eval_out["results"]

    # Always merge with existing (unless --force)
    if not args.force:
        existing = load_existing_results(experiment.run_dir, n_candidates)
        if existing:
            for r in results_to_save:
                existing[r["request_id"]] = r
            results_to_save = list(existing.values())
            # Recompute stats from merged results
            merged_stats = compute_multi_k_stats(results_to_save, k)
            print(f"\nMerged {len(eval_out['results'])} new + existing = {len(results_to_save)} total")
            eval_out["stats"] = merged_stats
    result_path = experiment.save_results(results_to_save, results_filename)
    print(f"\nResults saved to {result_path}")

    # Merge usage into usage.jsonl
    existing_usage = load_usage(experiment.run_dir)
    new_usage = extract_usage_from_results(results_to_save, n_candidates)
    existing_usage.update(new_usage)  # Overwrite on re-run
    save_usage(experiment.run_dir, existing_usage)

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

    # Check for existing results and skip/resume
    n_candidates = getattr(args, 'candidates', None) or len(dataset.items)
    if not args.force:
        existing = load_existing_results(run_dir, n_candidates)
        if existing:
            expected_ids = {r["id"] for r in dataset.requests}
            completed_ids = set(existing.keys())

            if expected_ids == completed_ids:
                # All requests complete - skip evaluation
                log.info(f"Already complete ({len(existing)} requests). Use --force to re-run.")
                cached_results = list(existing.values())
                k = getattr(args, 'k', 5)
                stats = compute_multi_k_stats(cached_results, k)
                from utils.usage import get_usage_tracker
                usage_for_display = get_usage_tracker().get_summary()
                # Only print per-run results if not in benchmark mode (aggregated summary shown at end)
                if not getattr(args, 'benchmark', False):
                    show_details = getattr(args, 'full', False)
                    print_ranking_results(stats, cached_results, usage_for_display, show_details=show_details)
                # Save config even when returning early (for aggregation)
                all_results = {"stats": stats, "results": cached_results}
                save_final_config(args, all_results, experiment)
                return {"stats": stats}

            # Partial results exist - only run missing requests
            missing_ids = expected_ids - completed_ids
            dataset.requests = [r for r in dataset.requests if r["id"] in missing_ids]
            dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in missing_ids}
            log.info(f"Resuming: {len(completed_ids)} complete, running {len(missing_ids)} missing")

    log.info(f"\n{dataset}")

    # Get method instance
    method = get_method(
        args.method,
        run_dir=str(run_dir),
        defense=getattr(args, 'defense', False),
        verbose=getattr(args, 'verbose', True)
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


def load_usage(run_dir: Path) -> dict:
    """Load existing usage records from usage.jsonl.

    Returns:
        Dict keyed by (request_id, n_candidates) -> usage record
    """
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        return {}

    existing = {}
    for line in usage_path.read_text().strip().split("\n"):
        if line:
            r = json.loads(line)
            key = (r["request_id"], r["n_candidates"])
            existing[key] = r
    return existing


def save_usage(run_dir: Path, usage_dict: dict):
    """Save usage records to usage.jsonl.

    Args:
        run_dir: Run directory path
        usage_dict: Dict keyed by (request_id, n_candidates) -> usage record
    """
    usage_path = run_dir / "usage.jsonl"
    # Sort by n_candidates then request_id for consistent output
    sorted_records = sorted(usage_dict.values(), key=lambda x: (x["n_candidates"], x["request_id"]))
    with open(usage_path, "w") as f:
        for r in sorted_records:
            f.write(json.dumps(r) + "\n")


def extract_usage_from_results(results: list, n_candidates: int) -> dict:
    """Extract per-request usage records from results.

    Args:
        results: List of result dicts from evaluation
        n_candidates: Number of candidates for this run

    Returns:
        Dict keyed by (request_id, n_candidates) -> usage record
    """
    from datetime import datetime
    timestamp = datetime.now().isoformat()

    usage_dict = {}
    for r in results:
        key = (r["request_id"], n_candidates)
        usage_dict[key] = {
            "request_id": r["request_id"],
            "n_candidates": n_candidates,
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "tokens": r.get("tokens", 0),
            "cost_usd": r.get("cost_usd", 0),
            "latency_ms": r.get("latency_ms", 0),
            "timestamp": timestamp,
        }
    return usage_dict


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
          usage.jsonl          (merged per-request usage across all scale points)
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
                "hits_at_2": "--",
                "hits_at_3": "--",
                "hits_at_4": "--",
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
                "hits_at_2": "--",
                "hits_at_3": "--",
                "hits_at_4": "--",
                "hits_at_5": "--",
                "status": "skipped",
            })
            continue

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
            log.info(f"Scale {n_candidates}: No valid requests (gold not in candidate set)")
            results_table.append({
                "candidates": n_candidates,
                "requests": 0,
                "hits_at_1": "--",
                "hits_at_2": "--",
                "hits_at_3": "--",
                "hits_at_4": "--",
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
                cached_results = list(existing.values())
                stats = compute_multi_k_stats(cached_results, k)
                hits_at = stats.get("hits_at", {})
                h1 = hits_at.get(1, hits_at.get("1", {})).get("accuracy", 0)
                h2 = hits_at.get(2, hits_at.get("2", {})).get("accuracy", 0)
                h3 = hits_at.get(3, hits_at.get("3", {})).get("accuracy", 0)
                h4 = hits_at.get(4, hits_at.get("4", {})).get("accuracy", 0)
                h5 = hits_at.get(5, hits_at.get("5", {})).get("accuracy", 0)
                results_table.append({
                    "candidates": n_candidates,
                    "requests": len(existing),
                    "hits_at_1": f"{h1:.2%}",
                    "hits_at_2": f"{h2:.2%}",
                    "hits_at_3": f"{h3:.2%}",
                    "hits_at_4": f"{h4:.2%}",
                    "hits_at_5": f"{h5:.2%}",
                    "status": "ok",
                })
                # Compute usage from cached per-request data
                cached_usage = {
                    "total_calls": len(cached_results),
                    "total_prompt_tokens": sum(r.get('prompt_tokens', 0) for r in cached_results),
                    "total_completion_tokens": sum(r.get('completion_tokens', 0) for r in cached_results),
                    "total_tokens": sum(r.get('tokens', 0) for r in cached_results),
                    "total_cost_usd": sum(r.get('cost_usd', 0) for r in cached_results),
                    "total_latency_ms": sum(r.get('latency_ms', 0) for r in cached_results),
                }
                # Only print per-scale results if --full
                if getattr(args, 'full', False):
                    print_ranking_results(stats, cached_results, cached_usage, show_details=True)
                continue

            # Partial results exist - only run missing requests
            missing_ids = expected_ids - completed_ids
            dataset.requests = [r for r in dataset.requests if r["id"] in missing_ids]
            dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in missing_ids}
            log.info(f"Scale {n_candidates}: Running {len(missing_ids)} missing requests")

        # Print header only when there's work to do
        print(f"\n{'='*60}")
        print(f"Running with {n_candidates} candidates...")
        print(f"{'='*60}")

        # Reset usage tracker for this scale point
        tracker.reset()

        log.info(f"Candidates: {n_candidates}, Items: {len(dataset.items)}, Requests: {len(dataset.requests)}")

        # Get method
        method = get_method(
            args.method,
            run_dir=str(run_dir),
            defense=getattr(args, 'defense', False),
            verbose=getattr(args, 'verbose', True)
        )

        # Run evaluation
        # Use dict mode for methods that need structured access
        dict_mode_methods = {"anot", "weaver"}
        eval_mode = "dict" if args.method in dict_mode_methods else "string"
        shuffle = getattr(args, 'shuffle', 'middle')
        eval_result = evaluate_ranking(
            dataset.items,
            method,
            dataset.requests,
            dataset.groundtruth,
            mode=eval_mode,
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

        # Merge usage into unified usage.jsonl
        existing_usage = load_usage(run_dir)
        new_usage = extract_usage_from_results(merged_results, n_candidates)
        existing_usage.update(new_usage)
        save_usage(run_dir, existing_usage)

        # Recompute stats from merged results
        stats = compute_multi_k_stats(merged_results, k)

        # Check if context exceeded
        if eval_result.get("context_exceeded"):
            context_exceeded_at = n_candidates
            results_table.append({
                "candidates": n_candidates,
                "requests": len(merged_results),
                "hits_at_1": "--",
                "hits_at_2": "--",
                "hits_at_3": "--",
                "hits_at_4": "--",
                "hits_at_5": "--",
                "status": "context_exceeded",
            })
            print(f"\n[STOP] Context limit exceeded at {n_candidates} candidates.")
        else:
            hits_at = stats.get("hits_at", {})
            h1 = hits_at.get(1, hits_at.get("1", {})).get("accuracy", 0)
            h2 = hits_at.get(2, hits_at.get("2", {})).get("accuracy", 0)
            h3 = hits_at.get(3, hits_at.get("3", {})).get("accuracy", 0)
            h4 = hits_at.get(4, hits_at.get("4", {})).get("accuracy", 0)
            h5 = hits_at.get(5, hits_at.get("5", {})).get("accuracy", 0)

            results_table.append({
                "candidates": n_candidates,
                "requests": len(merged_results),
                "hits_at_1": f"{h1:.2%}",
                "hits_at_2": f"{h2:.2%}",
                "hits_at_3": f"{h3:.2%}",
                "hits_at_4": f"{h4:.2%}",
                "hits_at_5": f"{h5:.2%}",
                "status": "ok",
            })

            # Only print per-scale results if --full
            if getattr(args, 'full', False):
                usage_for_display = tracker.get_summary()
                print_ranking_results(stats, merged_results, usage_for_display, show_details=True)

    # Save scaling summary
    save_scaling_summary(run_dir, results_table, k)

    # Print final summary table with Rich
    console = Console()
    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT SUMMARY: {args.method}")
    print(f"{'='*60}\n")

    table = Table(title="Scaling Results")
    table.add_column("Candidates", style="cyan")
    table.add_column("Requests", style="cyan")
    table.add_column("Hits@1", style="yellow")
    table.add_column("Hits@2", style="yellow")
    table.add_column("Hits@3", style="yellow")
    table.add_column("Hits@4", style="yellow")
    table.add_column("Hits@5", style="yellow")
    table.add_column("Status", style="green")

    for row in results_table:
        table.add_row(
            str(row['candidates']),
            str(row['requests']),
            row.get('hits_at_1', '--'),
            row.get('hits_at_2', '--'),
            row.get('hits_at_3', '--'),
            row.get('hits_at_4', '--'),
            row.get('hits_at_5', '--'),
            row['status']
        )
    console.print(table)

    if context_exceeded_at:
        print(f"\nContext limit exceeded at {context_exceeded_at} candidates.")
        print("Remaining scale points marked as '--' (not run).")
