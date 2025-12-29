#!/usr/bin/env python3
"""LLM evaluation for restaurant recommendation dataset."""

import json
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from data.loader import load_data, load_requests
from eval import evaluate, evaluate_parallel, print_results
from attack import ATTACK_CONFIGS, ATTACK_CHOICES, run_attack, apply_attack

RESULTS_DIR = Path("results")


def get_next_run_number() -> int:
    """Scan results/ and find the next available run number."""
    existing = glob.glob(str(RESULTS_DIR / "[0-9]*_*/"))
    if not existing:
        return 1
    numbers = []
    for p in existing:
        folder_name = Path(p).name
        try:
            num = int(folder_name.split("_")[0])
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def create_run_dir(run_name: str) -> Path:
    """Create a numbered run directory and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_num = get_next_run_number()
    run_dir = RESULTS_DIR / f"{run_num}_{run_name}"
    run_dir.mkdir(exist_ok=True)
    return run_dir


def main():
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
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", default=True,
                        help="Disable parallel execution (run sequentially instead)")

    args = parser.parse_args()

    # Initialize LLM configuration
    from llm import init_rate_limiter, configure
    init_rate_limiter(args.max_concurrent)
    configure(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_tokens_reasoning=args.max_tokens_reasoning,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    # Create run directory
    run_name = args.run_name or args.method
    run_dir = create_run_dir(run_name)
    out_path = Path(args.out) if args.out else run_dir / "results.jsonl"
    print(f"Run directory: {run_dir}")

    # Load data and requests
    items_clean = load_data(args.data, args.limit)
    requests = load_requests(args.requests)
    print(f"Loaded {len(items_clean)} items from {args.data}")
    print(f"Loaded {len(requests)} requests")

    # Select method
    from methods import get_method
    approach = getattr(args, 'knot_approach', 'base')
    method = get_method(args.method, mode=args.mode, approach=approach,
                        defense=args.defense, run_dir=str(run_dir))
    defense_str = " +defense" if args.defense else ""
    print(f"Using method: {args.method}" + (f" (mode={args.mode}, approach={approach})" if args.method == "knot" else "") + defense_str)

    # Determine which attacks to run
    if args.attack == "all":
        attacks_to_run = ["clean"] + list(ATTACK_CONFIGS.keys())
    elif args.attack == "none":
        attacks_to_run = ["clean"]
    else:
        attacks_to_run = [args.attack]

    # v4 and v5 approaches require dict mode for variable substitution
    if args.method == "knot" and approach in ("v4", "v5"):
        eval_mode = "dict"
    else:
        eval_mode = args.mode if args.method == "knot" else "string"
    all_stats = {}

    if args.parallel and len(attacks_to_run) > 1:
        # PARALLEL: Run all attacks concurrently
        print(f"\n{'='*50}")
        print(f"Running {len(attacks_to_run)} attacks in PARALLEL (max {args.max_concurrent} concurrent API calls)")
        print("=" * 50)

        with ThreadPoolExecutor(max_workers=len(attacks_to_run)) as executor:
            futures = {
                executor.submit(run_attack, name, items_clean, method, requests,
                                eval_mode, run_dir, evaluate_parallel): name
                for name in attacks_to_run
            }

            for future in as_completed(futures):
                attack_name, eval_out = future.result()
                all_stats[attack_name] = eval_out["stats"]
                print_results(eval_out["stats"], eval_out.get("req_ids"))

    else:
        # SEQUENTIAL: Run attacks one by one (original behavior)
        for attack_name in attacks_to_run:
            print(f"\n{'='*50}")
            print(f"Running: {attack_name}")
            print("=" * 50)

            # Apply attack (or use clean data)
            if attack_name == "clean":
                items = items_clean
            else:
                attack_type, attack_kwargs = ATTACK_CONFIGS[attack_name]
                items = apply_attack(items_clean, attack_type, **attack_kwargs)

            # Evaluate (use parallel for item-requests if --parallel, else sequential)
            if args.parallel:
                eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
            else:
                eval_out = evaluate(items, method, requests, mode=eval_mode)

            # Print results
            print_results(eval_out["stats"], eval_out.get("req_ids"))
            all_stats[attack_name] = eval_out["stats"]

            # Save results
            if len(attacks_to_run) == 1:
                result_filename = "results.jsonl"
            else:
                result_filename = f"results_{attack_name}.jsonl"
            result_path = run_dir / result_filename
            with open(result_path, 'w') as f:
                for r in sorted(eval_out["results"], key=lambda x: (x["item_id"], x["request_id"])):
                    f.write(json.dumps(r) + '\n')
            print(f"Results saved to {result_path}")

    # Save run config
    config = {
        "timestamp": datetime.now().isoformat(),
        "method": args.method,
        "mode": args.mode if args.method == "knot" else None,
        "approach": getattr(args, 'knot_approach', None) if args.method == "knot" else None,
        "defense": args.defense,
        "data": args.data,
        "requests": args.requests,
        "limit": args.limit,
        "attack": args.attack,
        "attacks_run": attacks_to_run,
        "llm_config": {
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "stats": all_stats if len(attacks_to_run) > 1 else all_stats.get("clean", all_stats),
    }
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Consolidate debug logs if v4 was used
    if args.method == "knot" and getattr(args, 'knot_approach', 'base') == "v4":
        try:
            from utils.logger import consolidate_logs
            consolidate_logs(str(run_dir))
            print(f"Debug logs consolidated to {run_dir}/debug/")
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not consolidate debug logs: {e}")


if __name__ == "__main__":
    main()
