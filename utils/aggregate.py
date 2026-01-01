"""
Aggregation utilities for benchmark runs.
Computes mean/std across multiple runs.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, Any, List

from rich.console import Console
from rich.table import Table

# Base results directory
BENCHMARK_DIR = Path("results") / "benchmarks"


def aggregate_benchmark_runs(method: str, data: str, selection_name: str,
                             attack: str = "clean") -> Dict[str, Any]:
    """
    Aggregate stats across all runs for a specific attack variant.

    Args:
        method: Method name (e.g., 'cot')
        data: Data name (e.g., 'yelp')
        selection_name: Selection name (e.g., 'selection_1')
        attack: Attack name (e.g., 'clean', 'typo_10')

    Returns:
        Summary dict with mean/std and per-run stats
    """
    # New structure: {method}_{data}/{attack}/{selection}_run_{N}/
    attack_dir = BENCHMARK_DIR / f"{method}_{data}" / attack
    if not attack_dir.exists():
        # Fallback: try legacy structure without attack subdir
        legacy_parent = BENCHMARK_DIR / f"{method}_{data}"
        if legacy_parent.exists():
            run_dirs = sorted(legacy_parent.glob(f"{selection_name}_run_*/"))
            if run_dirs:
                # Use legacy structure
                return _aggregate_from_dirs(run_dirs, method, data, selection_name, attack, legacy_parent)
        return {"runs": 0, "error": f"No benchmark directory found for attack '{attack}'"}

    run_dirs = sorted(attack_dir.glob(f"{selection_name}_run_*/"))
    if not run_dirs:
        return {"runs": 0, "error": f"No runs found for attack '{attack}'"}

    return _aggregate_from_dirs(run_dirs, method, data, selection_name, attack, attack_dir)


def _aggregate_from_dirs(run_dirs: List[Path], method: str, data: str,
                         selection_name: str, attack: str, summary_dir: Path) -> Dict[str, Any]:
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
    summary["selection"] = selection_name
    summary["attack"] = attack

    # Save summary.json in attack dir
    summary_path = summary_dir / f"{selection_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def aggregate_all_attacks(method: str, data: str, selection_name: str) -> Dict[str, Any]:
    """
    Aggregate across all attack variants for comparison.

    Args:
        method: Method name (e.g., 'cot')
        data: Data name (e.g., 'yelp')
        selection_name: Selection name (e.g., 'selection_1')

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
        if "_run_" in attack_dir.name:
            continue

        attack_name = attack_dir.name
        summary = aggregate_benchmark_runs(method, data, selection_name, attack_name)
        if summary.get("runs", 0) > 0:
            results[attack_name] = summary

    # Save combined summary
    if results:
        combined_path = parent / f"{selection_name}_all_attacks_summary.json"
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
    """Print aggregated summary using rich tables."""
    console = Console()

    if summary.get("error"):
        console.print(f"[red]{summary['error']}[/red]")
        return

    runs = summary.get("runs", 0)
    console.print(f"\n[bold]Benchmark Summary[/bold] ({runs} run{'s' if runs != 1 else ''})")
    console.print(f"  Method: {summary.get('method', 'N/A')}")
    console.print(f"  Data: {summary.get('data', 'N/A')}")
    console.print(f"  Selection: {summary.get('selection', 'N/A')}")

    if summary.get("type") == "ranking":
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

    elif summary.get("type") == "accuracy":
        values_str = ", ".join(f"{v:.4f}" for v in summary.get("values", []))
        console.print(f"\n[bold]Overall Accuracy[/bold]")
        console.print(f"  Mean: {summary.get('mean', 0):.4f}")
        console.print(f"  Std:  {summary.get('std', 0):.4f}")
        console.print(f"  Values: [{values_str}]")
