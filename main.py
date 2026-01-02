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

from run import run_single


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    log = setup_logging(args.verbose)
    config_llm(args)

    if args.benchmark:
        # Benchmark mode: check existing runs first
        experiment = create_experiment(args)
        completed = experiment.get_completed_runs()
        needed = args.auto - completed

        if needed <= 0:
            print(f"Already have {completed} run(s) (requested {args.auto}). Printing summary.")
        else:
            print(f"Running {needed} more run(s) (have {completed}, need {args.auto})")
            for i in range(needed):
                print(f"\n{'='*60}")
                print(f"Run {completed + i + 1} of {args.auto}")
                print(f"{'='*60}")
                experiment = create_experiment(args)
                run_single(args, experiment, log)

        # Always aggregate at end (benchmark mode)
        summary = aggregate_benchmark_runs(args.method, args.data)
        print_summary(summary)

    else:
        # Dev mode: single run, no aggregation
        experiment = create_experiment(args)
        run_single(args, experiment, log)


if __name__ == "__main__":
    main()
