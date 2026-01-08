#!/usr/bin/env python3
"""Scaling experiment for testing method performance across candidate counts."""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from data.loader import load_dataset, filter_by_candidates, DICT_MODE_METHODS
from utils.parsing import parse_limit_spec
from utils.aggregate import print_ranking_results
from utils.usage import get_usage_tracker

from .evaluate import evaluate_ranking, compute_multi_k_stats, extract_hits_at
from .io import (
    load_existing_results, load_usage, save_usage,
    extract_usage_from_results, load_failed_scales, save_scaling_summary
)


# Scale points for scaling experiment
SCALE_POINTS = [10, 20, 30, 40, 50]


def _format_compact(tokens: int, cost: float, latency_ms: float) -> str:
    """Format tokens, cost, latency into compact string like '608k, $0.20, 45s'."""
    if not tokens and not cost:
        return "--"
    # Format tokens: 608357 -> "608k"
    if tokens >= 1_000_000:
        tok_str = f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        tok_str = f"{tokens // 1_000}k"
    else:
        tok_str = str(tokens)
    # Format latency: ms -> seconds
    lat_str = f"{latency_ms / 1000:.0f}s" if latency_ms else "0s"
    return f"{tok_str}, ${cost:.2f}, {lat_str}"


def _make_row_skipped(n_candidates: int) -> dict:
    """Create a skipped result row."""
    return {
        "candidates": n_candidates,
        "requests": "--",
        "at_1": "--",
        "at_2": "--",
        "at_3": "--",
        "at_4": "--",
        "at_5": "--",
        "usage": "--",
        "status": "skipped",
    }


def _make_row_from_stats(n_candidates: int, n_requests: int, stats: dict,
                          tokens: int = 0, cost: float = 0.0,
                          latency_ms: float = 0.0, status: str = "ok") -> dict:
    """Create a result row from stats."""
    hits = extract_hits_at(stats, k=5)
    return {
        "candidates": n_candidates,
        "requests": n_requests,
        "at_1": f"{hits['hits_at_1']:.0%}",
        "at_2": f"{hits['hits_at_2']:.0%}",
        "at_3": f"{hits['hits_at_3']:.0%}",
        "at_4": f"{hits['hits_at_4']:.0%}",
        "at_5": f"{hits['hits_at_5']:.0%}",
        "usage": _format_compact(tokens, cost, latency_ms),
        "status": status,
    }


def _run_scale_point(args, run_dir: Path, n_candidates: int, k: int, tracker, log) -> tuple[list, dict | None, bool]:
    """Run evaluation for a single scale point.

    Returns:
        (merged_results, stats, context_exceeded)
    """
    from methods import get_method

    # Load and filter dataset
    dataset = load_dataset(
        args.data,
        review_limit=getattr(args, 'review_limit', None)
    )
    dataset = filter_by_candidates(dataset, n_candidates)

    # Build attack config for per-request attacks
    attack = getattr(args, 'attack', 'none')
    attack_config = None
    if attack not in ("none", "clean", None, ""):
        from attack import get_attack_config
        attack_config = get_attack_config(
            attack,
            n_restaurants=getattr(args, 'attack_restaurants', None),
            n_reviews=getattr(args, 'attack_reviews', 1),
            seed=getattr(args, 'seed', None),
            target_len=getattr(args, 'attack_target_len', None)
        )
        log.info(f"Attack configured: {attack} (per-request, protecting gold)")

    # Apply --limit if specified
    if args.limit:
        indices = parse_limit_spec(args.limit)
        all_requests = dataset.requests
        dataset.requests = [r for i, r in enumerate(all_requests) if i in indices]
        filtered_ids = {r["id"] for r in dataset.requests}
        dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in filtered_ids}

    # Apply request text override for testing (--request N --rtext "...")
    if getattr(args, 'rtext', None) and dataset.requests:
        req = dataset.requests[0]  # --request selects exactly one
        original_text = req.get('text', '')
        req['text'] = args.rtext
        req['context'] = args.rtext  # Also update context field
        log.info(f"[TEST MODE] Overriding {req['id']} text:")
        log.info(f"  Original: {original_text[:80]}{'...' if len(original_text) > 80 else ''}")
        log.info(f"  Override: {args.rtext[:80]}{'...' if len(args.rtext) > 80 else ''}")

    if len(dataset.requests) == 0:
        log.info(f"Scale {n_candidates}: No valid requests (gold not in candidate set)")
        return [], None, False

    # Check for existing results
    existing = load_existing_results(run_dir, n_candidates)
    if existing and not args.force:
        expected_ids = {r["id"] for r in dataset.requests}
        completed_ids = set(existing.keys())

        if expected_ids == completed_ids:
            # All complete - return cached
            log.info(f"Scale {n_candidates}: Already complete ({len(existing)} requests)")
            return list(existing.values()), None, False

        # Partial - only run missing
        missing_ids = expected_ids - completed_ids
        dataset.requests = [r for r in dataset.requests if r["id"] in missing_ids]
        dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in missing_ids}
        log.info(f"Scale {n_candidates}: Running {len(missing_ids)} missing requests")

    # Print header
    print(f"\n{'='*60}")
    print(f"Running with {n_candidates} candidates...")
    print(f"{'='*60}")

    # Reset tracker
    tracker.reset()

    log.info(f"Candidates: {n_candidates}, Items: {len(dataset.items)}, Requests: {len(dataset.requests)}")

    # Get method
    method = get_method(
        args.method,
        run_dir=str(run_dir),
        defense=getattr(args, 'defense', False),
        verbose=getattr(args, 'verbose', True),
        hierarchical=getattr(args, 'hierarchical', False)
    )

    # Run evaluation
    eval_mode = "dict" if args.method in DICT_MODE_METHODS else "string"
    shuffle = getattr(args, 'shuffle', 'random')

    # String-mode methods use adaptive truncation on context exceeded
    eval_result = evaluate_ranking(
        dataset.items,
        method,
        dataset.requests,
        dataset.groundtruth,
        mode=eval_mode,
        k=k,
        shuffle=shuffle,
        parallel=args.parallel,
        max_workers=getattr(args, 'max_concurrent', 200),
        attack_config=attack_config,
    )

    # Merge with existing
    if args.force:
        existing = {}
    else:
        existing = load_existing_results(run_dir, n_candidates)
    for r in eval_result["results"]:
        existing[r["request_id"]] = r
    merged_results = list(existing.values())

    # Save results
    results_path = run_dir / f"results_{n_candidates}.jsonl"
    with open(results_path, "w") as f:
        for r in sorted(merged_results, key=lambda x: x.get("request_id", "")):
            f.write(json.dumps(r) + "\n")

    # Merge usage
    existing_usage = load_usage(run_dir)
    new_usage = extract_usage_from_results(merged_results, n_candidates)
    existing_usage.update(new_usage)
    save_usage(run_dir, existing_usage)

    return merged_results, eval_result, eval_result.get("context_exceeded", False)


def run_scaling_experiment(args, log):
    """Run scaling experiment across multiple candidate counts.

    Tests how method performance degrades as number of candidates increases.
    Stops early when context length is exceeded.
    """
    from utils.experiment import create_experiment

    print(f"\n{'='*60}")
    print(f"SCALING EXPERIMENT: {args.method}")
    print(f"Scale points: {SCALE_POINTS}")
    print(f"{'='*60}\n")

    # Create experiment
    experiment = create_experiment(args)
    run_dir = experiment.setup()
    log.info(f"Run directory: {run_dir}")

    tracker = get_usage_tracker()

    # Load previously failed scales
    failed_scales = load_failed_scales(run_dir) if not args.force else set()
    if failed_scales:
        log.info(f"Skipping previously failed scales: {sorted(failed_scales)}")

    results_table = []
    context_exceeded_at = None
    k = getattr(args, 'k', 5)
    any_new_work = False  # Track if any new evaluations were run

    for n_candidates in SCALE_POINTS:
        # Skip previously failed scales
        if n_candidates in failed_scales:
            results_table.append(_make_row_skipped(n_candidates))
            continue

        # Skip if context already exceeded
        if context_exceeded_at:
            results_table.append(_make_row_skipped(n_candidates))
            continue

        # Run this scale point
        merged_results, eval_result, context_exceeded = _run_scale_point(
            args, run_dir, n_candidates, k, tracker, log
        )

        # Handle empty results (no valid requests)
        if not merged_results and eval_result is None:
            results_table.append({
                "candidates": n_candidates,
                "requests": 0,
                "at_1": "--", "at_2": "--", "at_3": "--", "at_4": "--", "at_5": "--",
                "usage": "--",
                "status": "no_requests",
            })
            continue

        # Handle cached results (eval_result is None)
        if eval_result is None:
            stats = compute_multi_k_stats(merged_results, k)
            cached_tokens = sum(r.get('tokens', 0) for r in merged_results)
            cached_cost = sum(r.get('cost_usd', 0) for r in merged_results)
            cached_latency = sum(r.get('latency_ms', 0) for r in merged_results)
            results_table.append(_make_row_from_stats(
                n_candidates, len(merged_results), stats,
                tokens=cached_tokens, cost=cached_cost, latency_ms=cached_latency
            ))
            # Print accuracy summary for cached results
            hits = extract_hits_at(stats, k=5)
            print(f"  → Scale {n_candidates} (cached): {hits['hits_at_1']:.0%} @1, {hits['hits_at_2']:.0%} @2, {hits['hits_at_3']:.0%} @3, {hits['hits_at_4']:.0%} @4, {hits['hits_at_5']:.0%} @5")
            # Print if --full
            if getattr(args, 'full', False):
                cached_usage = {
                    "total_calls": len(merged_results),
                    "total_prompt_tokens": sum(r.get('prompt_tokens', 0) for r in merged_results),
                    "total_completion_tokens": sum(r.get('completion_tokens', 0) for r in merged_results),
                    "total_tokens": sum(r.get('tokens', 0) for r in merged_results),
                    "total_cost_usd": sum(r.get('cost_usd', 0) for r in merged_results),
                    "total_latency_ms": sum(r.get('latency_ms', 0) for r in merged_results),
                }
                print_ranking_results(stats, merged_results, cached_usage, show_details=True)
            continue

        # Handle context exceeded
        if context_exceeded:
            context_exceeded_at = n_candidates
            any_new_work = True
            results_table.append({
                "candidates": n_candidates,
                "requests": len(merged_results),
                "at_1": "--", "at_2": "--", "at_3": "--", "at_4": "--", "at_5": "--",
                "usage": "--",
                "status": "context_exceeded",
            })
            print(f"\n[STOP] Context limit exceeded at {n_candidates} candidates.")
            continue

        # Normal result - new work was done
        any_new_work = True
        stats = compute_multi_k_stats(merged_results, k)
        usage = tracker.get_summary()
        results_table.append(_make_row_from_stats(
            n_candidates, len(merged_results), stats,
            tokens=usage.get('total_tokens', 0),
            cost=usage.get('total_cost_usd', 0),
            latency_ms=usage.get('total_latency_ms', 0)
        ))

        # Print accuracy summary for this scale point
        hits = extract_hits_at(stats, k=5)
        print(f"  → Scale {n_candidates}: {hits['hits_at_1']:.0%} @1, {hits['hits_at_2']:.0%} @2, {hits['hits_at_3']:.0%} @3, {hits['hits_at_4']:.0%} @4, {hits['hits_at_5']:.0%} @5")

        # Print if --full
        if getattr(args, 'full', False):
            usage_for_display = tracker.get_summary()
            print_ranking_results(stats, merged_results, usage_for_display, show_details=True)

    # Save summary only if new work was done
    if any_new_work:
        save_scaling_summary(run_dir, results_table, k, SCALE_POINTS)

    # Print final summary
    _print_scaling_summary(args.method, results_table, context_exceeded_at)


def _print_scaling_summary(method_name: str, results_table: list, context_exceeded_at: int | None):
    """Print final scaling summary table."""
    console = Console()
    print(f"\n{'='*70}")
    print(f"SCALING EXPERIMENT SUMMARY: {method_name}")
    print(f"{'='*70}\n")

    table = Table(title="Scaling Results")
    table.add_column("N", style="cyan")
    table.add_column("Req", style="cyan")
    table.add_column("@1", style="yellow")
    table.add_column("@2", style="yellow")
    table.add_column("@3", style="yellow")
    table.add_column("@4", style="yellow")
    table.add_column("@5", style="yellow")
    table.add_column("Usage (tok, $, time)", style="magenta")
    table.add_column("Status", style="dim")

    for row in results_table:
        table.add_row(
            str(row['candidates']),
            str(row['requests']),
            row.get('at_1', '--'),
            row.get('at_2', '--'),
            row.get('at_3', '--'),
            row.get('at_4', '--'),
            row.get('at_5', '--'),
            row.get('usage', '--'),
            row['status']
        )
    console.print(table)

    if context_exceeded_at:
        print(f"\nContext limit exceeded at {context_exceeded_at} candidates.")
        print("Remaining scale points marked as '--' (not run).")
