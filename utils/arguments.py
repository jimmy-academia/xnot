import argparse
from attack import ATTACK_CHOICES

PARALLEL_MODE = True
BENCHMARK_MODE = True

# Default selection number for each dataset
DEFAULT_SELECTIONS = {
    "yelp": 1,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")

    # Data arguments
    parser.add_argument("--data", default="yelp",
                        help="Dataset name (e.g., 'yelp', 'real') or explicit path to JSONL file")
    parser.add_argument("--selection", type=int, default=None,
                        help="Selection number to use (default: from DEFAULT_SELECTIONS)")
    parser.add_argument("--run-name", help="Name for this run (creates results/{N}_{run-name}/)")
    parser.add_argument("--out", help="Output results file (default: auto in run dir)")
    parser.add_argument("--limit", type=int, help="Limit items to process")
    parser.add_argument("--review-limit", type=int, help="Limit reviews per restaurant (for testing)")

    # Method arguments
    parser.add_argument("--method", choices=["cot", "cotsc", "l2m", "ps", "selfask", "parade", "rankgpt", "prp", "setwise", "listwise", "not", "anot", "anot_v2", "anot_origin", "dummy", "react", "decomp", "finegrained", "pal", "pot", "cot_table", "weaver"], default="cot", help="Method to use")

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
    parser.add_argument("--auto", type=int, default=None,
                        help="Auto-run N times for benchmark averaging (benchmark mode only)")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (results/dev/) instead of benchmark mode")
    # Verbose
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose/debug output")

    args = parser.parse_args()
    args.parallel = PARALLEL_MODE and not args.sequential
    args.benchmark = BENCHMARK_MODE and not args.dev

    # Resolve selection_name from --selection or DEFAULT_SELECTIONS
    if args.selection is not None:
        args.selection_name = f"selection_{args.selection}"
    elif args.data in DEFAULT_SELECTIONS:
        args.selection_name = f"selection_{DEFAULT_SELECTIONS[args.data]}"
    else:
        args.selection_name = None

    return args