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
    # Load stats from each run's config.json
    all_stats = []
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


def print_summary(summary: Dict[str, Any]):
    """Print aggregated summary."""
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


def print_ranking_results(stats: Dict[str, Any]):
    """Print single-run ranking results (Hits@K mode)."""
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
    else:
        print(f"{'K':<5} {'Hits':<8} {'Accuracy'}")
        print("-" * 25)
        for j in range(1, k + 1):
            hit_data = stats.get("hits_at", {}).get(j, stats.get("hits_at", {}).get(str(j), {}))
            hits = hit_data.get("hits", 0)
            acc = hit_data.get("accuracy", 0)
            print(f"{j:<5} {hits:<8} {acc:.4f}")
