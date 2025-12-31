#!/usr/bin/env python3
"""Evaluation and orchestration functions for LLM assessment."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.table import Table

import re
from data.loader import format_query, format_ranking_query, normalize_pred, load_groundtruth_scores
from utils.io import loadjl


def evaluate(items: list[dict], method: Callable, requests: list[dict], mode: str = "string") -> dict:
    """Run evaluation and collect results."""
    results = []
    req_ids = [r["id"] for r in requests]
    stats = {
        "total": 0, "correct": 0, "errors": 0,
        "per_request": {rid: {"total": 0, "correct": 0} for rid in req_ids},
        "confusion": {g: {p: 0 for p in [-1, 0, 1]} for g in [-1, 0, 1]},
    }

    total_pairs = len(items) * len(requests)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Evaluating...", total=total_pairs)

        for item in items:
            item_id = item.get("item_id", "unknown")
            query, num_reviews = format_query(item, mode)
            # Support both old (final_answers) and new (gold_labels) format
            gold_answers = item.get("gold_labels") or item.get("final_answers", {})

            for req in requests:
                req_id = req["id"]
                context = req["context"]
                gold = gold_answers.get(req_id)
                if gold is None:
                    progress.update(task, advance=1)
                    continue
                gold = int(gold)

                # Set IDs for knot logging (if enabled)
                try:
                    from methods.knot import set_current_ids
                    set_current_ids(item_id, req_id)
                except ImportError:
                    pass

                try:
                    pred = normalize_pred(method(query, context))
                except Exception as e:
                    pred = 0
                    stats["errors"] += 1

                correct = pred == gold
                stats["total"] += 1
                stats["correct"] += correct
                stats["per_request"][req_id]["total"] += 1
                stats["per_request"][req_id]["correct"] += correct
                stats["confusion"][gold][pred] += 1

                results.append({
                    "item_id": item_id,
                    "request_id": req_id,
                    "pred": pred,
                    "gold": gold,
                    "correct": correct,
                })
                progress.update(task, advance=1)

    return {"results": results, "stats": stats, "req_ids": req_ids}


def evaluate_single(item: dict, req: dict, method: Callable, mode: str = "string") -> dict:
    """Evaluate a single item-request pair (thread-safe)."""
    item_id = item.get("item_id", "unknown")
    req_id = req["id"]
    context = req["context"]
    gold_answers = item.get("gold_labels") or item.get("final_answers", {})
    gold = gold_answers.get(req_id)
    if gold is None:
        return None

    query, _ = format_query(item, mode)
    try:
        pred = normalize_pred(method(query, context))
        error = False
    except Exception:
        pred = 0
        error = True

    return {
        "item_id": item_id,
        "request_id": req_id,
        "pred": pred,
        "gold": int(gold),
        "correct": pred == int(gold),
        "error": error,
    }


def compute_stats(results: list[dict], req_ids: list[str]) -> dict:
    """Aggregate stats from results list."""
    stats = {
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "errors": sum(1 for r in results if r.get("error", False)),
        "per_request": {rid: {"total": 0, "correct": 0} for rid in req_ids},
        "confusion": {g: {p: 0 for p in [-1, 0, 1]} for g in [-1, 0, 1]},
    }
    for r in results:
        rid = r["request_id"]
        if rid in stats["per_request"]:
            stats["per_request"][rid]["total"] += 1
            stats["per_request"][rid]["correct"] += r["correct"]
        stats["confusion"][r["gold"]][r["pred"]] += 1
    return stats


def evaluate_parallel(items: list[dict], method: Callable, requests: list[dict],
                      mode: str = "string", max_workers: int = 40) -> dict:
    """Parallel version of evaluate() - all item-request pairs at once."""
    req_ids = [r["id"] for r in requests]
    pairs = [(item, req) for item in items for req in requests]
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(pairs))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single, item, req, method, mode)
                       for item, req in pairs]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    stats = compute_stats(results, req_ids)
    return {"results": results, "stats": stats, "req_ids": req_ids}


def print_results(stats: dict, req_ids: list[str] = None):
    """Print evaluation summary."""
    total, correct = stats["total"], stats["correct"]
    acc = correct / total if total else 0
    print(f"\nOverall: {acc:.4f} ({correct}/{total})")

    print("\nPer-request:")
    req_ids = req_ids or list(stats["per_request"].keys())
    for req_id in req_ids:
        if req_id in stats["per_request"]:
            r = stats["per_request"][req_id]
            acc = r["correct"] / r["total"] if r["total"] else 0
            print(f"  {req_id}: {acc:.4f} ({r['correct']}/{r['total']})")

    print("\nConfusion (rows=gold, cols=pred):")
    print("       -1    0    1")
    for g in [-1, 0, 1]:
        row = stats["confusion"][g]
        print(f"  {g:2d}  {row[-1]:4d} {row[0]:4d} {row[1]:4d}")


# --- Ranking Evaluation ---

def parse_index(response: str, max_index: int = 20) -> int:
    """Parse LLM response to extract item index (1 to max_index).

    Args:
        response: LLM response text
        max_index: Maximum valid index

    Returns:
        Index (1-based) or 0 if parsing fails
    """
    if response is None:
        return 0

    # Try to find a number 1-20 in the response
    # First try: exact match at start
    match = re.match(r'^\s*(\d+)', str(response))
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    # Second try: find any number in brackets like [5] or (5)
    match = re.search(r'[\[\(](\d+)[\]\)]', str(response))
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    # Third try: find standalone number
    for match in re.finditer(r'\b(\d+)\b', str(response)):
        idx = int(match.group(1))
        if 1 <= idx <= max_index:
            return idx

    return 0  # Failed to parse


def parse_indices(response: str, max_index: int = 20, k: int = 5) -> list[int]:
    """Parse LLM response to extract up to k item indices.

    Args:
        response: LLM response text
        max_index: Maximum valid index
        k: Maximum number of indices to extract

    Returns:
        List of indices (1-based), up to k unique items
    """
    if response is None:
        return []

    indices = []
    for match in re.finditer(r'\b(\d+)\b', str(response)):
        idx = int(match.group(1))
        if 1 <= idx <= max_index and idx not in indices:
            indices.append(idx)
            if len(indices) >= k:
                break
    return indices


def compute_multi_k_stats(results: list[dict], k: int) -> dict:
    """Compute Hits@1 through Hits@k from results.

    Args:
        results: List of result dicts with 'gold_index' and 'pred_indices'
        k: Maximum k value to compute

    Returns:
        Dict with total, k, and hits_at dict for each level
    """
    total = len(results)
    hits_at = {}
    for j in range(1, k + 1):
        hits = sum(1 for r in results
                   if r["gold_index"] in r["pred_indices"][:j])
        hits_at[j] = {"hits": hits, "accuracy": hits / total if total else 0}
    return {
        "total": total,
        "k": k,
        "hits_at": hits_at,
    }


def evaluate_ranking(items: list[dict], method: Callable, requests: list[dict],
                     groundtruth_scores: dict, mode: str = "string", k: int = 1) -> dict:
    """Evaluate using ranking (Top-K accuracy / Hits@K).

    Args:
        items: All items (restaurants) to rank
        method: LLM method that outputs best index
        requests: List of requests
        groundtruth_scores: {request_id: {item_id: total_score}} from precomputed groundtruth
        mode: "string" or "dict" for formatting
        k: Number of top predictions to check (default 1 for top-1 accuracy)

    Returns:
        Dict with results and accuracy stats
    """
    results = []
    req_ids = [r["id"] for r in requests]

    # Build item index mapping (1-based)
    item_ids = [item.get("item_id") for item in items]
    item_id_to_index = {item_id: i + 1 for i, item_id in enumerate(item_ids)}

    # Format all items as a single query
    query, item_count = format_ranking_query(items, mode)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Ranking evaluation...", total=len(requests))

        for req in requests:
            req_id = req["id"]
            context = req["context"]

            # Get gold answer: item with highest score for this request
            scores = groundtruth_scores.get(req_id, {})
            if not scores:
                progress.update(task, advance=1)
                continue

            # Find gold index (item with highest score)
            gold_item_id = max(scores.keys(), key=lambda x: scores.get(x, 0))
            gold_index = item_id_to_index.get(gold_item_id, 0)
            gold_score = scores.get(gold_item_id, 0)

            # Get prediction from LLM
            try:
                response = method(query, context)
                if k == 1:
                    pred_indices = [parse_index(response, item_count)]
                else:
                    pred_indices = parse_indices(response, item_count, k)
            except Exception:
                pred_indices = []

            results.append({
                "request_id": req_id,
                "pred_indices": pred_indices,
                "gold_index": gold_index,
                "gold_item_id": gold_item_id,
                "gold_score": gold_score,
            })
            progress.update(task, advance=1)

    # Compute multi-K stats
    stats = compute_multi_k_stats(results, k)

    return {
        "results": results,
        "req_ids": req_ids,
        "stats": stats,
    }


def print_ranking_results(eval_out: dict):
    """Print ranking evaluation summary with multi-K stats using rich."""
    stats = eval_out["stats"]
    results = eval_out["results"]
    console = Console()

    # Hits@K table
    table = Table(title=f"Hits@K Results (total={stats['total']})")
    table.add_column("Metric", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Hits", style="yellow")

    for j in range(1, stats["k"] + 1):
        h = stats["hits_at"][j]
        table.add_row(f"Hits@{j}", f"{h['accuracy']:.4f}", f"{h['hits']}/{stats['total']}")

    console.print(table)

    # Per-request details
    console.print("\n[bold]Per-request:[/bold]")
    for r in results:
        hit = r["gold_index"] in r["pred_indices"]
        symbol = "[green]✓[/green]" if hit else "[red]✗[/red]"
        pred_str = ",".join(str(i) for i in r['pred_indices']) if r['pred_indices'] else "none"
        console.print(f"  {r['request_id']}: {symbol} pred=[{pred_str}] gold={r['gold_index']} (score={r.get('gold_score', 'N/A')})")


# --- Orchestration Helpers ---

def check_previous_run(args, experiment) -> dict | None:
    """Check for previous run and prompt user.

    Returns:
        Dict with results/stats if user chooses to reload, None otherwise
    """
    prev = experiment.find_previous_run(args.method, args.data)
    if not prev:
        return None

    prev_dir, prev_config = prev
    console = Console()
    console.print(f"\n[yellow]Found previous run with same method/data:[/yellow]")
    console.print(f"  [cyan]{prev_dir}[/cyan]")
    console.print(f"  timestamp: {prev_config.get('timestamp', 'unknown')}")

    response = input("Reload cached results? [Y/n]: ").strip().lower()
    if response in ("", "y", "yes"):
        results_path = prev_dir / "results.jsonl"
        results = loadjl(results_path)
        k = getattr(args, 'k', 5)
        stats = compute_multi_k_stats(results, k)
        return {"results": results, "stats": stats, "path": str(prev_dir)}

    return None


def run_evaluation_loop(args, data, requests, method, experiment):
    """Evaluate dataset(s) - handles both single and multi-attack.

    Args:
        data: Either list[dict] (single attack) or dict[str, list[dict]] (all attacks)
    """
    # Check for previous run (dev mode only)
    cached = check_previous_run(args, experiment)
    if cached:
        print_ranking_results({"results": cached["results"], "stats": cached["stats"]})
        return {"clean": cached["stats"]}  # Return stats dict for consistency

    # Determine eval_mode for variable substitution
    approach = getattr(args, 'knot_approach', 'base')
    if args.method == "knot" and approach in ("v4", "v5"):
        eval_mode = "dict"
    else:
        eval_mode = args.mode if args.method == "knot" else "string"

    # Check if ranking mode is enabled (default: True for top-1 accuracy)
    ranking_mode = getattr(args, 'ranking', True)
    k = getattr(args, 'k', 1)

    # Single attack case - wrap in dict for uniform handling
    if isinstance(data, list):
        attack_name = args.attack if args.attack not in ("none", "clean", None) else "clean"
        data = {attack_name: data}

    all_stats = {}
    for attack_name, items in data.items():
        print(f"\n{'='*50}\nRunning: {attack_name}\n{'='*50}")

        if ranking_mode:
            # Ranking evaluation (top-1 accuracy by default)
            try:
                groundtruth_scores = load_groundtruth_scores(args.selection_name)
                eval_out = evaluate_ranking(items, method, requests, groundtruth_scores, mode=eval_mode, k=k)
                print_ranking_results(eval_out)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("Falling back to per-item evaluation")
                ranking_mode = False

        if not ranking_mode:
            # Per-item evaluation (legacy mode)
            if args.parallel:
                eval_out = evaluate_parallel(items, method, requests, mode=eval_mode)
            else:
                eval_out = evaluate(items, method, requests, mode=eval_mode)
            print_results(eval_out["stats"], eval_out.get("req_ids"))

        all_stats[attack_name] = eval_out["stats"]

        filename = "results.jsonl" if len(data) == 1 else f"results_{attack_name}.jsonl"
        result_path = experiment.save_results(eval_out["results"], filename)
        print(f"Results saved to {result_path}")

    return all_stats


def save_final_config(args, stats, experiment):
    """Construct and save the run configuration."""
    # Unwrap stats if only one attack was run
    final_stats = stats if len(stats) > 1 else next(iter(stats.values()))

    config = {
        "method": args.method,
        "mode": args.mode if args.method == "knot" else None,
        "approach": getattr(args, 'knot_approach', None) if args.method == "knot" else None,
        "defense": args.defense,
        "data": args.data,
        "selection": args.selection_name,
        "limit": args.limit,
        "attack": args.attack,
        "llm_config": {
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "stats": final_stats,
    }

    config_path = experiment.save_config(config)
    print(f"\nConfig saved to {config_path}")
