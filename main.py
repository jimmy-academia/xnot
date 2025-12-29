#!/usr/bin/env python3
"""LLM evaluation for restaurant recommendation dataset."""

from utils.arguments import parse_args
from utils.logger import setup_logger_level, logger
from utils.llm import config_llm
from utils.experiment import create_experiment, ExperimentError

from data.loader import load_data, load_requests
from methods import get_method

from concurrent.futures import ThreadPoolExecutor, as_completed

from eval import evaluate, evaluate_parallel, print_results
from attack import ATTACK_CONFIGS, ATTACK_CHOICES, run_attack, apply_attack

def main():
    args = parse_args()
    log = setup_logger_level(args.verbose)
    config_llm(args)

    # Create experiment (handles dev vs benchmark mode)
    try:
        experiment = create_experiment(args)
        run_dir = experiment.setup()
    except ExperimentError as e:
        log.error(str(e))
        return 1

    mode_str = "BENCHMARK" if experiment.benchmark_mode else "development"
    log.info(f"Mode: {mode_str}")
    log.info(f"Run directory: {run_dir}")

    # Load data and requests
    items_clean = load_data(args.data, args.limit)
    requests = load_requests(args.requests)

    log.info(f"Loaded {len(items_clean)} items from {args.data}")
    log.info(f"Loaded {len(requests)} requests")

    # # Select method
    # approach = getattr(args, 'knot_approach', 'base')
    # method = get_method(args.method, mode=args.mode, approach=approach,
    #                     defense=args.defense, run_dir=str(run_dir))
    # defense_str = " +defense" if args.defense else ""
    # print(f"Using method: {args.method}" + (f" (mode={args.mode}, approach={approach})" if args.method == "knot" else "") + defense_str)

    # # Determine which attacks to run
    # if args.attack == "all":
    #     attacks_to_run = ["clean"] + list(ATTACK_CONFIGS.keys())
    # elif args.attack == "none":
    #     attacks_to_run = ["clean"]
    # else:
    #     attacks_to_run = [args.attack]

    # # v4 and v5 approaches require dict mode for variable substitution
    # if args.method == "knot" and approach in ("v4", "v5"):
    #     eval_mode = "dict"
    # else:
    #     eval_mode = args.mode if args.method == "knot" else "string"
    # all_stats = {}

    # if args.parallel and len(attacks_to_run) > 1:
    #     # PARALLEL: Run all attacks concurrently
    #     print(f"\n{'='*50}")
    #     print(f"Running {len(attacks_to_run)} attacks in PARALLEL (max {args.max_concurrent} concurrent API calls)")
    #     print("=" * 50)

    #     with ThreadPoolExecutor(max_workers=len(attacks_to_run)) as executor:
    #         futures = {
    #             executor.submit(run_attack, name, items_clean, method, requests,
    #                             eval_mode, run_dir, evaluate_parallel): name
    #             for name in attacks_to_run
    #         }

    #         for future in as_completed(futures):
    #             attack_name, eval_out = future.result()
    #             all_stats[attack_name] = eval_out["stats"]
    #             print_results(eval_out["stats"], eval_out.get("req_ids"))

    # else:
    #     # SEQUENTIAL: Run attacks one by one (original behavior)
    #     for attack_name in attacks_to_run:
    #         print(f"\n{'='*50}")
    #         print(f"Running: {attack_name}")
    #         print("=" * 50)

    #         # Apply attack (or use clean data)
    #         if attack_name == "clean":
    #             items = items_clean
    #         else:
    #             attack_type, attack_kwargs = ATTACK_CONFIGS[attack_name]
    #             items = apply_attack(items_clean, attack_type, **attack_kwargs)

    #         # Evaluate (use parallel for item-requests if --parallel, else sequential)
    #         if args.parallel:
    #             eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
    #         else:
    #             eval_out = evaluate(items, method, requests, mode=eval_mode)

    #         # Print results
    #         print_results(eval_out["stats"], eval_out.get("req_ids"))
    #         all_stats[attack_name] = eval_out["stats"]

    #         # Save results
    #         if len(attacks_to_run) == 1:
    #             result_filename = "results.jsonl"
    #         else:
    #             result_filename = f"results_{attack_name}.jsonl"
    #         result_path = run_dir / result_filename
    #         with open(result_path, 'w') as f:
    #             for r in sorted(eval_out["results"], key=lambda x: (x["item_id"], x["request_id"])):
    #                 f.write(json.dumps(r) + '\n')
    #         print(f"Results saved to {result_path}")

    # # Save run config
    # config = {
    #     "timestamp": datetime.now().isoformat(),
    #     "method": args.method,
    #     "mode": args.mode if args.method == "knot" else None,
    #     "approach": getattr(args, 'knot_approach', None) if args.method == "knot" else None,
    #     "defense": args.defense,
    #     "data": args.data,
    #     "requests": args.requests,
    #     "limit": args.limit,
    #     "attack": args.attack,
    #     "attacks_run": attacks_to_run,
    #     "llm_config": {
    #         "provider": args.provider,
    #         "model": args.model,
    #         "temperature": args.temperature,
    #         "max_tokens": args.max_tokens,
    #     },
    #     "stats": all_stats if len(attacks_to_run) > 1 else all_stats.get("clean", all_stats),
    # }
    # config_path = run_dir / "config.json"
    # with open(config_path, 'w') as f:
    #     json.dump(config, f, indent=2)
    # print(f"\nConfig saved to {config_path}")

    # # Consolidate debug logs if v4 was used
    # if args.method == "knot" and getattr(args, 'knot_approach', 'base') == "v4":
    #     try:
    #         from utils.logger import consolidate_logs
    #         consolidate_logs(str(run_dir))
    #         print(f"Debug logs consolidated to {run_dir}/debug/")
    #     except ImportError:
    #         pass
    #     except Exception as e:
    #         print(f"Warning: Could not consolidate debug logs: {e}")


if __name__ == "__main__":
    main()
