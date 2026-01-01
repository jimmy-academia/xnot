#!/usr/bin/env python3
"""Output formatting and printing utilities for evaluation results."""

from rich.console import Console
from rich.table import Table


def print_results(stats: dict, req_ids: list[str] = None):
    """Print evaluation summary.

    Args:
        stats: Statistics dict with total, correct, per_request, confusion
        req_ids: Optional list of request IDs to display
    """
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


def print_ranking_results(eval_out: dict):
    """Print ranking evaluation summary with multi-K stats using rich.

    Args:
        eval_out: Evaluation output dict with stats, results
    """
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
