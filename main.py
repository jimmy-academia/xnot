#!/usr/bin/env python3
"""
main.py - LLM evaluation for restaurant recommendation.

Entry point for running evaluations with different methods.
"""

import logging

from utils.arguments import parse_args
from utils.llm import config_llm
from utils.experiment import create_experiment
from utils.aggregate import aggregate_benchmark_runs, print_summary
from data.loader import load_dataset

from run import run_single, run_scaling_experiment


def get_all_attacks(include_heterogeneity: bool = False) -> list:
    """Get list of all attack types for --attack all."""
    from attack import ATTACK_CONFIGS
    attacks = list(ATTACK_CONFIGS.keys())
    if not include_heterogeneity:
        attacks = [a for a in attacks if a != "heterogeneity"]
    return attacks


def run_all_attacks(args, log):
    """Run evaluation for all attack types."""
    # Determine which attacks to run
    include_hetero = getattr(args, 'attack_target_len', None) is not None
    attacks = get_all_attacks(include_heterogeneity=include_hetero)

    print(f"\n{'='*60}")
    print(f"RUNNING ALL ATTACKS ({len(attacks)} total)")
    print(f"Attacks: {', '.join(attacks)}")
    print(f"{'='*60}\n")

    for i, attack in enumerate(attacks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(attacks)}] Attack: {attack}")
        print(f"{'='*60}")

        # Update args.attack for this iteration
        args.attack = attack

        # Create new experiment for each attack (different directory)
        experiment = create_experiment(args)
        run_single(args, experiment, log)

    # Print summary of all attacks
    print(f"\n{'='*60}")
    print(f"ALL ATTACKS COMPLETE")
    print(f"{'='*60}")
    for attack in attacks:
        args.attack = attack
        summary = aggregate_benchmark_runs(args.method, args.data, attack=attack)
        if summary:
            print(f"\n{attack}:")
            print_summary(summary, show_details=False)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )
    # Suppress verbose logs (must be after basicConfig)
    for logger_name in ["httpx", "httpcore", "openai", "httpx._client", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    log = setup_logging(args.verbose)
    config_llm(args)

    # Handle --attack all
    if args.attack == "all":
        if args.candidates is None:
            print("ERROR: --attack all requires --candidates N (e.g., --candidates 10)")
            return
        run_all_attacks(args, log)
        return

    # Scaling experiment (default) vs single run (--candidates N)
    if args.candidates is None:
        run_scaling_experiment(args, log)
        return

    # Check if this is a partial run (--limit was specified)
    is_partial = args.limit is not None

    if args.benchmark:
        if is_partial:
            # Partial run: always run and merge into latest/target run
            experiment = create_experiment(args)
            print(f"\n{'='*60}")
            print(f"Partial run (--limit {args.limit})")
            print(f"{'='*60}")
            run_single(args, experiment, log)
        else:
            # Full benchmark mode: run_single handles resume via results_{n}.jsonl
            experiment = create_experiment(args)
            run_single(args, experiment, log)

            # Check for additional runs needed (args.auto)
            experiment = create_experiment(args)
            completed = experiment.get_completed_runs()
            needed = args.auto - completed

            if needed > 0:
                print(f"Running {needed} more run(s) (have {completed}, need {args.auto})")
                for i in range(needed):
                    print(f"\n{'='*60}")
                    print(f"Run {completed + i + 1} of {args.auto}")
                    print(f"{'='*60}")
                    experiment = create_experiment(args)
                    run_single(args, experiment, log)

        # Always aggregate at end (benchmark mode)
        summary = aggregate_benchmark_runs(args.method, args.data)
        print_summary(summary, show_details=True)

    else:
        # Dev mode: single run, no aggregation
        experiment = create_experiment(args)
        run_single(args, experiment, log)


if __name__ == "__main__":
    main()
