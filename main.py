#!/usr/bin/env python3
"""
main.py

LLM evaluation for restaurant recommendation dataset.
"""

from utils.arguments import parse_args
from utils.logger import setup_logger_level, logger
from utils.llm import config_llm
from utils.experiment import create_experiment

from data.loader import load_data, load_requests
from methods import get_method

from run import run_evaluation_loop, save_final_config

def main():
    args = parse_args()
    logger = setup_logger_level(args.verbose)
    config_llm(args)

    # Create experiment (handles dev vs benchmark mode)
    experiment = create_experiment(args)
    run_dir = experiment.setup()

    modestr = "BENCHMARK" if experiment.benchmark_mode else "development"
    logger.info(f"Mode: {modestr}")
    logger.info(f"Run directory: {run_dir}")

    # Load data and requests
    data = load_data(args.data, args.limit, args.attack)
    print(data)
    input()
    requests = load_requests(args.requests)

    if isinstance(data, dict):
        logger.info(f"Loaded {len(data)} attack variants")
    else:
        logger.info(f"Loaded {len(data)} items from {args.data}")
    logger.info(f"Loaded {len(requests)} requests")

    # Select method
    method = get_method(args, run_dir)

    # Run evaluation (handles both parallel and sequential)
    stats = run_evaluation_loop(args, data, requests, method, experiment)

    # Finalize
    save_final_config(args, stats, experiment)
    experiment.consolidate_debug_logs()

if __name__ == "__main__":
    main()
