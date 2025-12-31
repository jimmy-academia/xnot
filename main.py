#!/usr/bin/env python3
"""
main.py

LLM evaluation for restaurant recommendation dataset.
"""

from utils.arguments import parse_args
from utils.logger import setup_logger_level, logger
from utils.llm import config_llm
from utils.experiment import create_experiment

from data.loader import load_dataset
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

    # Load data
    dataset = load_dataset(args.data, args.selection_name, args.limit, args.attack)
    logger.info(f"\n{dataset}")
    
    # Select method
    method = get_method(args, run_dir)
    input()
    
    # Run evaluation (handles both parallel and sequential)
    stats = run_evaluation_loop(args, dataset.items, dataset.requests, method, experiment)

    # Finalize
    save_final_config(args, stats, experiment)
    experiment.consolidate_debug_logs()

if __name__ == "__main__":
    main()
