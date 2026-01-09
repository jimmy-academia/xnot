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
METHOD_CHOICES = ["cot", "ps", "plan_act", "listwise", "weaver", "anot", "react", "dummy"]

# Attack choices - import from attack.py for consistency
# Note: heterogeneity requires --attack-target-len
ATTACK_CHOICES = [
    "none",  # Default: no attack (clean baseline)
    "typo_10", "typo_20",
    "inject_override", "inject_fake_sys", "inject_hidden", "inject_manipulation",
    "fake_positive", "fake_negative",
    "sarcastic_wifi", "sarcastic_noise", "sarcastic_outdoor", "sarcastic_all",
    "heterogeneity",
    "all",   # Run all attacks
    "both",  # Run clean baseline + all attacks
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")

    # Data arguments
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Dataset name (e.g., 'philly_cafes') or path to data directory")
    parser.add_argument("--run-name", help="Name for this run (creates results/{N}_{run-name}/)")
    parser.add_argument("--limit", type=str, default=None,
                        help="Filter requests: N (first N), N-M (range), or N,M,O (specific indices)")
    parser.add_argument("--group", type=str, metavar="N[,N,...]",
                        help="Filter by group(s): single (3), comma-separated (1,2,3), or range (1-3)")
    parser.add_argument("--run", type=int, default=1,
                        help="Target run number (default: 1)")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing results")
    parser.add_argument("--review-limit", type=int, help="Limit reviews per restaurant (for testing)")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Run single evaluation with N candidates (default: scaling experiment)")

    # Method arguments
    parser.add_argument("--method", choices=METHOD_CHOICES, default="anot",
                        help="Method to use")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical parallel phase2 for ANoT (experimental)")

    # Attack arguments
    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none",
                        help="Attack type (none=clean baseline, all=all attacks, both=clean+all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible attacks")
    parser.add_argument("--defense", action="store_true",
                        help="Enable defense prompts (attack-resistant mode)")
    parser.add_argument("--attack-restaurants", type=int, default=None,
                        help="Number of non-gold restaurants to attack (default: all)")
    parser.add_argument("--attack-reviews", type=int, default=1,
                        help="Number of reviews per restaurant to modify (default: 1)")
    parser.add_argument("--attack-target-len", type=int, default=None,
                        help="Target character length for heterogeneity attack")

    # LLM configuration
    parser.add_argument("--provider", choices=["openai", "anthropic", "local", "slm", "vllm"], default="openai",
                        help="LLM provider: openai, anthropic, local, slm, or vllm (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Override model (default: role-based selection)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=32000,
                        help="Max tokens for response (default: 32000)")
    parser.add_argument("--max-tokens-reasoning", type=int, default=32000,
                        help="Max tokens for reasoning models (default: 32000)")
    parser.add_argument("--base-url", default="",
                        help="Base URL for local provider")

    # Ranking/evaluation arguments
    parser.add_argument("--k", type=int, default=5,
                        help="Number of predictions for Hits@K evaluation (default: 5)")
    parser.add_argument("--shuffle", choices=["none", "middle", "random"], default="random",
                        help="Shuffle strategy for candidates (default: random)")

    # Execution arguments
    parser.add_argument("--max-concurrent", type=int, default=200,
                        help="Max concurrent API calls (default=200)")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable parallel execution (run sequentially instead)")
    parser.add_argument("--auto", type=int, default=1,
                        help="Target number of runs for benchmark (default: 1)")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (results/dev/) instead of benchmark mode")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run 1 request per group (10 total, fast validation)")

    # Verbose (default: True for now, use --no-verbose to disable)
    parser.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable verbose/debug output (default: True)")

    # Full output (show per-request details)
    parser.add_argument("--full", action="store_true", default=False,
                        help="Show full per-request results (default: summary only)")

    # Request text override for testing (without modifying files)
    parser.add_argument("--request", type=int, default=None,
                        help="Request number to test (e.g., 1 for R01). Implies --limit to that request.")
    parser.add_argument("--rtext", type=str, default=None,
                        help="Override request text for --request (for testing new phrasings)")

    args = parser.parse_args()

    # Validate --limit: single integer must be positive; ranges/lists validated later
    if args.limit:
        try:
            if int(args.limit) <= 0:
                parser.error(f"--limit must be positive (got {args.limit})")
        except ValueError:
            pass

    # Validate --request and --rtext
    if args.rtext and not args.request:
        parser.error("--rtext requires --request")
    if args.request:
        # Auto-set limit to just that request
        # Use range format 'N-N' to select exactly request N (e.g., '5-5' = R05 only)
        args.limit = f"{args.request}-{args.request}"
        # Force dev mode for testing
        args.dev = True
        # Force sequential for easier debugging
        args.sequential = True

    # Derived arguments
    args.parallel = PARALLEL_MODE and not args.sequential
    args.benchmark = BENCHMARK_MODE and not args.dev

    # Forbid dummy method in benchmark mode
    if args.method == "dummy" and args.benchmark:
        parser.error("dummy method cannot be used in benchmark mode (use --dev)")

    # Resolve data path
    if not Path(args.data).is_absolute():
        data_path = DATA_DIR / args.data
        if data_path.exists():
            args.data_dir = data_path
        else:
            args.data_dir = Path(args.data)
    else:
        args.data_dir = Path(args.data)

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
