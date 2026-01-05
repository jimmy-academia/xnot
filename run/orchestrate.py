#!/usr/bin/env python3
"""Run orchestration for single evaluations."""

import json
from pathlib import Path

from data.loader import load_dataset, filter_by_candidates
from utils.parsing import parse_limit_spec
from utils.aggregate import print_ranking_results
from utils.usage import get_usage_tracker

from .evaluate import evaluate_ranking, compute_multi_k_stats
from .io import load_existing_results, load_usage, save_usage, extract_usage_from_results


def run_evaluation_loop(args, dataset, method, experiment, attack_config=None):
    """Run evaluation on the dataset.

    Args:
        args: Parsed arguments
        dataset: Dataset object with items, requests, groundtruth
        method: Method instance
        experiment: ExperimentManager instance
        attack_config: Optional attack configuration for per-request attacks

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
    attack_str = f", attack={attack_config['attack']}" if attack_config else ""
    print(f"\nRunning ranking evaluation (k={k}, {mode_str}{shuffle_str}{attack_str})...", flush=True)
    eval_out = evaluate_ranking(
        dataset.items,
        method,
        dataset.requests,
        dataset.groundtruth,
        mode=eval_mode,
        k=k,
        shuffle=shuffle,
        parallel=parallel,
        max_workers=max_workers,
        attack_config=attack_config
    )
    # Get current usage for display
    usage_for_display = get_usage_tracker().get_summary()
    # Show results: always in dev mode, or in benchmark mode if --full
    show_details = getattr(args, 'full', False)
    if not getattr(args, 'benchmark', False) or show_details:
        print_ranking_results(eval_out["stats"], eval_out["results"], usage_for_display, show_details=show_details)

    # Merge with existing results if resuming
    n_candidates = getattr(args, 'candidates', None) or len(dataset.items)
    results_filename = f"results_{n_candidates}.jsonl"
    results_to_save = eval_out["results"]

    # Always merge with existing (unless --force)
    if not args.force:
        existing = load_existing_results(experiment.run_dir, n_candidates)
        if existing:
            for r in results_to_save:
                existing[r["request_id"]] = r
            results_to_save = list(existing.values())
            merged_stats = compute_multi_k_stats(results_to_save, k)
            print(f"\nMerged {len(eval_out['results'])} new + existing = {len(results_to_save)} total")
            eval_out["stats"] = merged_stats
    result_path = experiment.save_results(results_to_save, results_filename)
    print(f"\nResults saved to {result_path}")

    # Merge usage into usage.jsonl
    existing_usage = load_usage(experiment.run_dir)
    new_usage = extract_usage_from_results(results_to_save, n_candidates)
    existing_usage.update(new_usage)
    save_usage(experiment.run_dir, existing_usage)

    return {"stats": eval_out["stats"]}


def save_final_config(args, all_results, experiment, attack_config=None):
    """Construct and save the run configuration."""
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
        "attack_config": attack_config,
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

    # Filter requests if --limit specified
    if args.limit:
        indices = parse_limit_spec(args.limit)
        all_requests = dataset.requests
        dataset.requests = [r for i, r in enumerate(all_requests) if i in indices]
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
                log.info(f"Already complete ({len(existing)} requests). Use --force to re-run.")
                cached_results = list(existing.values())
                k = getattr(args, 'k', 5)
                stats = compute_multi_k_stats(cached_results, k)
                usage_for_display = get_usage_tracker().get_summary()
                if not getattr(args, 'benchmark', False):
                    show_details = getattr(args, 'full', False)
                    print_ranking_results(stats, cached_results, usage_for_display, show_details=show_details)
                all_results = {"stats": stats, "results": cached_results}
                save_final_config(args, all_results, experiment)
                return {"stats": stats}

            missing_ids = expected_ids - completed_ids
            dataset.requests = [r for r in dataset.requests if r["id"] in missing_ids]
            dataset.groundtruth = {rid: gt for rid, gt in dataset.groundtruth.items() if rid in missing_ids}
            log.info(f"Resuming: {len(completed_ids)} complete, running {len(missing_ids)} missing")

    log.info(f"\n{dataset}")

    method = get_method(
        args.method,
        run_dir=str(run_dir),
        defense=getattr(args, 'defense', False),
        verbose=getattr(args, 'verbose', True)
    )
    print(f"\nMethod: {method}")

    all_results = run_evaluation_loop(args, dataset, method, experiment, attack_config)

    save_final_config(args, all_results, experiment, attack_config)
    experiment.consolidate_debug_logs()

    return all_results
