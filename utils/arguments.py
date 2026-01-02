"""CLI argument parsing for the evaluation framework."""

import argparse
from pathlib import Path

# Mode flags - can be overridden via CLI args
PARALLEL_MODE = True
BENCHMARK_MODE = True

# Default data path
DEFAULT_DATA = "philly_cafes"
DATA_DIR = Path(__file__).parent.parent / "data"

# Method choices - champions only
METHOD_CHOICES = ["cot", "ps", "listwise", "weaver", "anot", "dummy"]

# Attack choices
ATTACK_CHOICES = ["none", "clean", "all", "typo", "verbose", "duplicate",
                  "inject_override", "inject_system", "fake_positive",
                  "fake_negative", "sarcastic"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")

    # Data arguments
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Dataset name (e.g., 'philly_cafes') or path to data directory")
    parser.add_argument("--run-name", help="Name for this run (creates results/{N}_{run-name}/)")
    parser.add_argument("--limit", type=int, help="Limit number of requests to process")
    parser.add_argument("--review-limit", type=int, help="Limit reviews per restaurant (for testing)")

    # Method arguments
    parser.add_argument("--method", choices=METHOD_CHOICES, default="cot",
                        help="Method to use")

    # Attack arguments
    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none",
                        help="Attack type to apply (none=clean, all=run all attacks)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible attacks")
    parser.add_argument("--defense", action="store_true",
                        help="Enable defense prompts (attack-resistant mode)")

    # LLM configuration
    parser.add_argument("--provider", choices=["openai", "anthropic", "local"], default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Override model (default: role-based selection)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for response (default: 1024)")
    parser.add_argument("--max-tokens-reasoning", type=int, default=4096,
                        help="Max tokens for reasoning models (default: 4096)")
    parser.add_argument("--base-url", default="",
                        help="Base URL for local provider")

    # Ranking/evaluation arguments
    parser.add_argument("--k", type=int, default=5,
                        help="Number of predictions for Hits@K evaluation (default: 5)")

    # Execution arguments
    parser.add_argument("--max-concurrent", type=int, default=200,
                        help="Max concurrent API calls (default=200)")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable parallel execution (run sequentially instead)")
    parser.add_argument("--auto", type=int, default=1,
                        help="Target number of runs for benchmark (default: 1)")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (results/dev/) instead of benchmark mode")

    # Verbose
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose/debug output")

    args = parser.parse_args()

    # Derived arguments
    args.parallel = PARALLEL_MODE and not args.sequential
    args.benchmark = BENCHMARK_MODE and not args.dev

    # Resolve data path
    if not Path(args.data).is_absolute():
        data_path = DATA_DIR / args.data
        if data_path.exists():
            args.data_dir = data_path
        else:
            args.data_dir = Path(args.data)
    else:
        args.data_dir = Path(args.data)

    # For backward compatibility
    args.selection_name = "all"

    return args


def get_data_paths(args) -> dict:
    """Get paths to data files from parsed args.

    Returns:
        Dictionary with paths to restaurants, reviews, requests, groundtruth
    """
    data_dir = args.data_dir
    return {
        "restaurants": data_dir / "restaurants.jsonl",
        "reviews": data_dir / "reviews.jsonl",
        "requests": data_dir / "requests.jsonl",
        "groundtruth": data_dir / "groundtruth.jsonl",
    }
