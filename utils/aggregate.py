"""
Aggregation utilities for benchmark runs.
Computes mean/std across multiple runs.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, Any, List

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Base results directory
BENCHMARK_DIR = Path("results") / "benchmarks"


def aggregate_benchmark_runs(method: str, data: str, attack: str = "clean") -> Dict[str, Any]:
    """
    Aggregate stats across all runs for a specific attack variant.

    Args:
        method: Method name (e.g., 'cot')
        data: Data name (e.g., 'philly_cafes')
        attack: Attack name (e.g., 'clean', 'typo')

    Returns:
        Summary dict with mean/std and per-run stats
    """
    # Structure: {method}_{data}/{attack}/run_{N}/
    attack_dir = BENCHMARK_DIR / f"{method}_{data}" / attack
    if not attack_dir.exists():
        return {"runs": 0, "error": f"No benchmark directory found for attack '{attack}'"}

    run_dirs = sorted(attack_dir.glob("run_*/"))
    if not run_dirs:
        return {"runs": 0, "error": f"No runs found for attack '{attack}'"}

    return _aggregate_from_dirs(run_dirs, method, data, attack, attack_dir)


def _aggregate_from_dirs(run_dirs: List[Path], method: str, data: str,
                         attack: str, summary_dir: Path) -> Dict[str, Any]:
    """Helper to aggregate stats from a list of run directories."""
    # Load stats and usage from each run's config.json
    all_stats = []
    all_usage = []
    skipped = 0
    skipped_reasons = {"no_config": 0, "no_stats": 0, "parse_error": 0}

    for d in run_dirs:
        config_path = d / "config.json"
        if not config_path.exists():
            skipped += 1
            skipped_reasons["no_config"] += 1
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            skipped += 1
            skipped_reasons["parse_error"] += 1
            continue

        if "stats" in config:
            all_stats.append(config["stats"])
            # Also collect usage if present
            if "usage" in config:
                all_usage.append(config["usage"])
        else:
            skipped += 1
            skipped_reasons["no_stats"] += 1

    # Print warnings for skipped runs
    if skipped > 0:
        print(f"WARNING: Skipped {skipped} of {len(run_dirs)} runs:")
        if skipped_reasons["no_config"] > 0:
            print(f"  - {skipped_reasons['no_config']} without config.json")
        if skipped_reasons["no_stats"] > 0:
            print(f"  - {skipped_reasons['no_stats']} without stats in config")
        if skipped_reasons["parse_error"] > 0:
            print(f"  - {skipped_reasons['parse_error']} with parse errors")

    if not all_stats:
        return {"runs": 0, "error": "No stats found in runs"}

    # Determine stats type and compute aggregates
    summary = _aggregate_stats(all_stats)
    summary["runs"] = len(all_stats)
    summary["method"] = method
    summary["data"] = data
    summary["attack"] = attack

    # Add usage aggregation if available
    if all_usage:
        summary["usage"] = _aggregate_usage(all_usage)

    # Save summary.json in attack dir
    summary_path = summary_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def aggregate_all_attacks(method: str, data: str) -> Dict[str, Any]:
    """
    Aggregate across all attack variants for comparison.

    Args:
        method: Method name (e.g., 'cot')
        data: Data name (e.g., 'philly_cafes')

    Returns:
        Dict mapping attack names to their aggregated summaries
    """
    parent = BENCHMARK_DIR / f"{method}_{data}"
    if not parent.exists():
        return {"error": "No benchmark directory found"}

    results = {}
    for attack_dir in sorted(parent.iterdir()):
        if not attack_dir.is_dir():
            continue
        # Skip if it's a legacy run directory (has run_ in name)
        if "run_" in attack_dir.name:
            continue

        attack_name = attack_dir.name
        summary = aggregate_benchmark_runs(method, data, attack_name)
        if summary.get("runs", 0) > 0:
            results[attack_name] = summary

    # Save combined summary
    if results:
        combined_path = parent / "all_attacks_summary.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def _validate_stat_formats(all_stats: List[Dict]) -> str:
    """Check all stats have consistent format.

    Returns:
        Format type: "ranking" or "accuracy"

    Raises:
        ValueError: If stats have mixed formats
    """
    formats = set()
    for stat in all_stats:
        if "hits_at" in stat:
            formats.add("ranking")
        elif "correct" in stat:
            formats.add("accuracy")
        else:
            formats.add("unknown")

    if len(formats) > 1:
        raise ValueError(f"Mixed stat formats in runs: {formats}. Cannot aggregate.")
    return formats.pop()


def _aggregate_stats(all_stats: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate stats list, handling both ranking and per-item modes.

    Validates format consistency before aggregating.
    """
    stat_format = _validate_stat_formats(all_stats)

    if stat_format == "ranking":
        return _aggregate_ranking_stats(all_stats)
    elif stat_format == "accuracy":
        return _aggregate_accuracy_stats(all_stats)
    else:
        return {"error": f"Unknown stat format: {stat_format}"}


def _aggregate_ranking_stats(all_stats: List[Dict]) -> Dict[str, Any]:
    """Aggregate ranking mode stats (Hits@K)."""
    k = all_stats[0].get("k", 5)

    # Collect accuracies for each K
    hits_at_summary = {}
    for j in range(1, k + 1):
        accuracies = [s["hits_at"][str(j)]["accuracy"] for s in all_stats if str(j) in s.get("hits_at", {})]
        if accuracies:
            hits_at_summary[j] = {
                "mean": statistics.mean(accuracies),
                "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "values": accuracies,
            }

    return {
        "type": "ranking",
        "k": k,
        "hits_at": hits_at_summary,
        "per_run": all_stats,
    }


def _aggregate_accuracy_stats(all_stats: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-item accuracy stats."""
    accuracies = []
    for s in all_stats:
        total = s.get("total", 0)
        correct = s.get("correct", 0)
        if total > 0:
            accuracies.append(correct / total)

    return {
        "type": "accuracy",
        "mean": statistics.mean(accuracies) if accuracies else 0.0,
        "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "values": accuracies,
        "per_run": all_stats,
    }


def _aggregate_usage(all_usage: List[Dict]) -> Dict[str, Any]:
    """Aggregate usage stats across runs."""
    total_calls = sum(u.get("total_calls", 0) for u in all_usage)
    total_tokens = sum(u.get("total_tokens", 0) for u in all_usage)
    total_prompt = sum(u.get("total_prompt_tokens", 0) for u in all_usage)
    total_completion = sum(u.get("total_completion_tokens", 0) for u in all_usage)
    total_cost = sum(u.get("total_cost_usd", 0) for u in all_usage)

    return {
        "total_calls": total_calls,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "mean_cost_per_run": round(total_cost / len(all_usage), 6) if all_usage else 0,
        "per_run": all_usage,
    }


def get_latest_run_dir(attack_dir: Path) -> Path:
    """Get the latest run directory."""
    import re
    if not attack_dir.exists():
        return None
    existing = list(attack_dir.glob("run_*/"))
    if not existing:
        return None
    nums = []
    for p in existing:
        match = re.search(r'run_(\d+)$', p.name)
        if match:
            nums.append((int(match.group(1)), p))
    if not nums:
        return None
    return max(nums, key=lambda x: x[0])[1]


def load_results_from_file(path: Path) -> List[Dict]:
    """Load results from a JSONL file."""
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def print_summary(summary: Dict[str, Any], show_details: bool = False):
    """Print aggregated summary.

    Args:
        summary: Summary dict from aggregate_benchmark_runs
        show_details: If True, also show per-request details from latest run
    """
    if summary.get("error"):
        print(f"Error: {summary['error']}")
        return

    runs = summary.get("runs", 0)
    print(f"\nBenchmark Summary ({runs} run{'s' if runs != 1 else ''})")
    print(f"  Method: {summary.get('method', 'N/A')}")
    print(f"  Data: {summary.get('data', 'N/A')}")
    print(f"  Attack: {summary.get('attack', 'N/A')}")

    if summary.get("type") == "ranking":
        if HAS_RICH:
            _print_ranking_rich(summary)
        else:
            _print_ranking_plain(summary)

    elif summary.get("type") == "accuracy":
        values_str = ", ".join(f"{v:.4f}" for v in summary.get("values", []))
        print(f"\nOverall Accuracy")
        print(f"  Mean: {summary.get('mean', 0):.4f}")
        print(f"  Std:  {summary.get('std', 0):.4f}")
        print(f"  Values: [{values_str}]")

    # Print usage summary if available
    usage = summary.get("usage")
    if usage:
        print(f"\nToken Usage (aggregated across {runs} run{'s' if runs != 1 else ''}):")
        print(f"  Total API Calls: {usage.get('total_calls', 0):,}")
        print(f"  Total Tokens: {usage.get('total_tokens', 0):,}")
        print(f"  Total Cost: ${usage.get('total_cost_usd', 0):.4f}")
        if usage.get("mean_cost_per_run"):
            print(f"  Mean Cost/Run: ${usage.get('mean_cost_per_run', 0):.4f}")

    # Show per-request details from latest run if requested
    if show_details and summary.get("type") == "ranking":
        method = summary.get("method")
        data = summary.get("data")
        attack = summary.get("attack", "clean")
        if method and data:
            attack_dir = BENCHMARK_DIR / f"{method}_{data}" / attack
            latest_run = get_latest_run_dir(attack_dir)
            if latest_run:
                results = load_results_from_file(latest_run / "results.jsonl")
                if results:
                    _print_per_request_details(results)


def _print_per_request_details(results: List[Dict]):
    """Print per-request details in double-column format."""
    if not results:
        return

    sorted_results = sorted(results, key=lambda x: x.get("request_id", ""))
    half = (len(sorted_results) + 1) // 2

    if HAS_RICH:
        console = Console()
        console.print("\n[bold]Per-request:[/bold]")
        for i in range(half):
            left = _format_result_entry(sorted_results[i])
            left = left.replace("✓", "[green]✓[/green]").replace("✗", "[red]✗[/red]")
            if i + half < len(sorted_results):
                right = _format_result_entry(sorted_results[i + half])
                right = right.replace("✓", "[green]✓[/green]").replace("✗", "[red]✗[/red]")
                console.print(f"  {left:<40} | {right}")
            else:
                console.print(f"  {left}")
    else:
        print("\nPer-request:")
        for i in range(half):
            left = _format_result_entry(sorted_results[i])
            if i + half < len(sorted_results):
                right = _format_result_entry(sorted_results[i + half])
                print(f"  {left:<40} | {right}")
            else:
                print(f"  {left}")


def _print_ranking_rich(summary: Dict[str, Any]):
    """Print ranking results with rich tables."""
    console = Console()
    table = Table(title="Hits@K Aggregated Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Values", style="dim")

    for k, stats in summary.get("hits_at", {}).items():
        values_str = ", ".join(f"{v:.4f}" for v in stats["values"])
        table.add_row(
            f"Hits@{k}",
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            values_str
        )
    console.print(table)


def _print_ranking_plain(summary: Dict[str, Any]):
    """Print ranking results without rich."""
    print("\nHits@K Aggregated Results:")
    print(f"{'Metric':<10} {'Mean':<10} {'Std':<10} {'Values'}")
    print("-" * 50)
    for k, stats in summary.get("hits_at", {}).items():
        values_str = ", ".join(f"{v:.4f}" for v in stats["values"])
        print(f"Hits@{k:<5} {stats['mean']:<10.4f} {stats['std']:<10.4f} [{values_str}]")


def print_results(stats: Dict[str, Any]):
    """Print single-run results (per-item accuracy mode)."""
    total = stats.get("total", 0)
    correct = stats.get("correct", 0)
    accuracy = correct / total if total > 0 else 0

    print(f"\nResults Summary")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Print confusion matrix if available
    confusion = stats.get("confusion", {})
    if confusion:
        print("\nConfusion Matrix:")
        header = "Gold \\ Pred"
        print(f"{header:<12} {'-1':<8} {'0':<8} {'1':<8}")
        for gold in [-1, 0, 1]:
            row = confusion.get(str(gold), {})
            print(f"{gold:<12} {row.get('-1', 0):<8} {row.get('0', 0):<8} {row.get('1', 0):<8}")


def _format_result_entry(r: Dict) -> str:
    """Format a single result entry for display.

    Shows both shuffled and original positions when shuffle was applied.
    Format: R00: ✓ [shuf]→[orig] gold=N(→M)
    Where:
      - [shuf] = predictions in shuffled space
      - [orig] = predictions mapped to original space
      - gold=N(→M) = original gold index (shuffled position if different)
    """
    gold_idx = r.get("gold_idx", -1)
    pred_indices = r.get("pred_indices", [])
    shuffled_preds = r.get("shuffled_preds", None)
    shuffled_gold_pos = r.get("shuffled_gold_pos", None)

    hit = (gold_idx + 1) in pred_indices
    symbol = "✓" if hit else "✗"

    # Format predictions
    orig_str = ",".join(str(i) for i in pred_indices) if pred_indices else ""

    # Check if shuffle was applied (shuffled_preds present and different from pred_indices)
    if shuffled_preds is not None and shuffled_preds != pred_indices:
        shuf_str = ",".join(str(i) for i in shuffled_preds) if shuffled_preds else ""
        pred_part = f"[{shuf_str}]→[{orig_str}]"
    else:
        pred_part = f"[{orig_str}]"

    # Format gold (show shuffled position if different)
    gold_1idx = gold_idx + 1
    if shuffled_gold_pos is not None and shuffled_gold_pos != gold_1idx:
        gold_part = f"gold={gold_1idx}(→{shuffled_gold_pos})"
    else:
        gold_part = f"gold={gold_1idx}"

    return f"{r['request_id']}: {symbol} {pred_part} {gold_part}"


def print_ranking_results(stats: Dict[str, Any], results: List[Dict] = None,
                          usage: Dict[str, Any] = None):
    """Print single-run ranking results (Hits@K mode).

    Args:
        stats: Stats dict with total, k, hits_at
        results: Optional list of per-request results for detailed output
        usage: Optional usage dict with token counts and costs
    """
    total = stats.get("total", 0)
    k = stats.get("k", 5)

    print(f"\nRanking Results (n={total})")

    if HAS_RICH:
        console = Console()
        table = Table(title=f"Hits@K Results")
        table.add_column("K", style="cyan")
        table.add_column("Hits", style="green")
        table.add_column("Accuracy", style="yellow")

        for j in range(1, k + 1):
            hit_data = stats.get("hits_at", {}).get(j, stats.get("hits_at", {}).get(str(j), {}))
            hits = hit_data.get("hits", 0)
            acc = hit_data.get("accuracy", 0)
            table.add_row(str(j), str(hits), f"{acc:.4f}")
        console.print(table)

        # Per-request details (double column)
        if results:
            console.print("\n[bold]Per-request:[/bold]")
            sorted_results = sorted(results, key=lambda x: x.get("request_id", ""))
            half = (len(sorted_results) + 1) // 2

            # Pre-calculate left entries and max width
            left_entries = [_format_result_entry(sorted_results[i]) for i in range(half)]
            max_width = max(len(e) for e in left_entries) if left_entries else 35

            for i in range(half):
                # Pad BEFORE adding colors (Rich markup adds non-display chars)
                left_padded = left_entries[i].ljust(max_width)
                left_colored = left_padded.replace("✓", "[green]✓[/green]").replace("✗", "[red]✗[/red]")

                if i + half < len(sorted_results):
                    right = _format_result_entry(sorted_results[i + half])
                    right_colored = right.replace("✓", "[green]✓[/green]").replace("✗", "[red]✗[/red]")
                    console.print(f"  {left_colored} | {right_colored}")
                else:
                    console.print(f"  {left_colored}")
    else:
        print(f"{'K':<5} {'Hits':<8} {'Accuracy'}")
        print("-" * 25)
        for j in range(1, k + 1):
            hit_data = stats.get("hits_at", {}).get(j, stats.get("hits_at", {}).get(str(j), {}))
            hits = hit_data.get("hits", 0)
            acc = hit_data.get("accuracy", 0)
            print(f"{j:<5} {hits:<8} {acc:.4f}")

        # Per-request details (double column, plain text)
        if results:
            print("\nPer-request:")
            sorted_results = sorted(results, key=lambda x: x.get("request_id", ""))
            half = (len(sorted_results) + 1) // 2

            # Pre-calculate left entries and max width
            left_entries = [_format_result_entry(sorted_results[i]) for i in range(half)]
            max_width = max(len(e) for e in left_entries) if left_entries else 35

            for i in range(half):
                left_padded = left_entries[i].ljust(max_width)
                if i + half < len(sorted_results):
                    right = _format_result_entry(sorted_results[i + half])
                    print(f"  {left_padded} | {right}")
                else:
                    print(f"  {left_padded}")

    # Print usage summary if provided
    if usage and usage.get("total_calls", 0) > 0:
        print(f"\nToken Usage:")
        print(f"  Total Calls: {usage.get('total_calls', 0):,}")
        print(f"  Total Tokens: {usage.get('total_tokens', 0):,}")
        print(f"  Total Cost: ${usage.get('total_cost_usd', 0):.4f}")
