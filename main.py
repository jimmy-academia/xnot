#!/usr/bin/env python3
"""
main.py

LLM evaluation for restaurant recommendation dataset.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.arguments import parse_args
from utils.logger import setup_logger_level, logger
from utils.llm import config_llm
from utils.experiment import create_experiment
from utils.aggregate import aggregate_benchmark_runs, aggregate_all_attacks, print_summary
from attack import get_all_attack_names

from run import run_single


def run_single_attack(args, attack_name, verbose=False):
    """Run evaluation for a single attack (can run in parallel).

    This is a top-level function for multiprocessing compatibility.
    """
    # Re-setup logging and LLM config for subprocess
    log = setup_logger_level(verbose)
    config_llm(args)

    # Create experiment with specific attack
    experiment = create_experiment(args, attack=attack_name)

    # Override attack in args for this run
    import copy
    run_args = copy.copy(args)
    run_args.attack = attack_name

    print(f"\n{'='*60}")
    print(f"Running attack: {attack_name}")
    print(f"{'='*60}")

    return run_single(run_args, experiment, log)


def main():
    args = parse_args()
    log = setup_logger_level(args.verbose)
    config_llm(args)

    # Handle --attack all mode
    if args.attack == "all":
        attack_names = get_all_attack_names()
        print(f"Running all attacks: {attack_names}")

        if args.benchmark:
            # Benchmark mode: run each attack separately with own directory
            # Use sequential execution (LLM calls are already parallelized)
            for attack_name in attack_names:
                run_single_attack(args, attack_name, args.verbose)

            # Aggregate all attacks
            summary = aggregate_all_attacks(args.method, args.data, args.selection_name)
            print(f"\n{'='*60}")
            print("All Attacks Summary")
            print(f"{'='*60}")
            for attack, attack_summary in summary.items():
                if attack != "error":
                    print(f"\n{attack}:")
                    print_summary(attack_summary)
        else:
            # Dev mode: single run handles all attacks
            experiment = create_experiment(args)
            run_single(args, experiment, log)

        return

    # Standard single-attack or clean mode
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
        attack = args.attack if args.attack not in ("none", "clean", None) else "clean"
        summary = aggregate_benchmark_runs(args.method, args.data, args.selection_name, attack)
        print_summary(summary)

    elif args.benchmark:
        # Single benchmark run
        experiment = create_experiment(args)
        run_single(args, experiment, log)

        # Aggregate existing runs
        attack = args.attack if args.attack not in ("none", "clean", None) else "clean"
        summary = aggregate_benchmark_runs(args.method, args.data, args.selection_name, attack)
        print_summary(summary)

    else:
        # Dev mode: single run, no aggregation
        experiment = create_experiment(args)
        run_single(args, experiment, log)


if __name__ == "__main__":
    main()
