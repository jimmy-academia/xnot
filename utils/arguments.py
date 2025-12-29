import argparse
from attack import ATTACK_CHOICES

PARALLEL_MODE = True
BENCHMARK_MODE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM on restaurant recommendations")

    # Data arguments
    parser.add_argument("--data", default="data/processed/real_data.jsonl", help="Input JSONL file")
    parser.add_argument("--requests", default="data/requests/complex_requests.json", help="User requests JSON file")
    parser.add_argument("--run-name", help="Name for this run (creates results/{N}_{run-name}/)")
    parser.add_argument("--out", help="Output results file (default: auto in run dir)")
    parser.add_argument("--limit", type=int, help="Limit items to process")

    # Method arguments
    parser.add_argument("--method", choices=["cot", "not", "knot", "dummy"], default="dummy", help="Method to use")
    parser.add_argument("--mode", choices=["string", "dict"], default="string", help="Input mode for knot")
    parser.add_argument("--knot-approach", choices=["base", "voting", "iterative", "divide", "v4", "v5"], default="base",
                        help="Approach for knot (base=default, voting=self-consistency, iterative=plan refinement, divide=divide-conquer, v4=hierarchical planning, v5=robust adversarial)")

    # Attack arguments
    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none",
                        help="Attack type to apply (none=clean, all=run all attacks)")
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

    # Execution arguments
    parser.add_argument("--max-concurrent", type=int, default=500,
                        help="Max concurrent API calls (default=500, safe for Tier 5)")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable parallel execution (run sequentially instead)")
    # Verbose
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose/debug output")

    args = parser.parse_args()
    args.parallel = PARALLEL_MODE and not args.sequential
    args.benchmark = BENCHMARK_MODE
    return args