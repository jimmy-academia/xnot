#!/usr/bin/env python3
"""
main.py

LLM evaluation for restaurant recommendation dataset.
"""

from utils.arguments import parse_args
from utils.logger import setup_logger_level, logger
from utils.llm import config_llm
from utils.experiment import create_experiment
from utils.aggregate import aggregate_benchmark_runs, print_summary

from data.loader import load_dataset
from methods import get_method

from run import run_evaluation_loop, save_final_config


def run_single(args, experiment, log):
    """Execute a single evaluation run."""
    run_dir = experiment.setup()

    modestr = "BENCHMARK" if experiment.benchmark_mode else "development"
    log.info(f"Mode: {modestr}")
    log.info(f"Run directory: {run_dir}")

    # Load data
    dataset = load_dataset(args.data, args.selection_name, args.limit, args.attack, args.review_limit)
    log.info(f"\n{dataset}")

    # Select method
    method = get_method(args, run_dir)
    print(method)

    # Run evaluation (handles both parallel and sequential)
    stats = run_evaluation_loop(args, dataset.items, dataset.requests, method, experiment)

    # Finalize
    save_final_config(args, stats, experiment)
    experiment.consolidate_debug_logs()

    return stats


def main():
    args = parse_args()
    log = setup_logger_level(args.verbose)
    config_llm(args)

    if args.benchmark and args.auto:
        # Auto-run mode: run multiple times for averaging
        experiment = create_experiment(args)
        completed = experiment.get_completed_runs()
        needed = args.auto - completed

        if needed <= 0:
            print(f"Already have {completed} runs (requested {args.auto}). Skipping new runs.")
        else:
            print(f"Running {needed} more runs (have {completed}, need {args.auto})")
            for i in range(needed):
                print(f"\n{'='*60}")
                print(f"Run {completed + i + 1} of {args.auto}")
                print(f"{'='*60}")
                experiment = create_experiment(args)
                run_single(args, experiment, log)

        # Always aggregate at end (benchmark mode)
        summary = aggregate_benchmark_runs(args.method, args.data, args.selection_name)
        print_summary(summary)

    elif args.benchmark:
        # Single benchmark run
        experiment = create_experiment(args)
        run_single(args, experiment, log)

        # Aggregate existing runs
        summary = aggregate_benchmark_runs(args.method, args.data, args.selection_name)
        print_summary(summary)

    else:
        # Dev mode: single run, no aggregation
        experiment = create_experiment(args)
        run_single(args, experiment, log)


if __name__ == "__main__":
    main()
