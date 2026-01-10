#!/usr/bin/env python3
"""
G1a-ANoT Evaluation Driver

Evaluates the G1a-ANoT method on the peanut allergy safety task
and computes Adjusted AUPRC metrics.

Usage:
    python g1_anot_eval.py --limit 5           # Quick test on 5 restaurants
    python g1_anot_eval.py                     # Full eval on all restaurants
    python g1_anot_eval.py --compare           # Compare with Direct LLM baseline
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import configure

from g1_anot import G1aANoT
from g1_gt_compute import compute_gt_for_k
from g1_allergy import get_task, TASK_REGISTRY
from score_auprc import (
    calculate_ordinal_auprc, compute_avg_primitive_accuracy,
    print_report, CLASS_ORDER
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(k: int) -> List[Dict]:
    """Load dataset for given K value."""
    dataset_file = DATA_DIR / f"dataset_K{k}.jsonl"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    restaurants = []
    with open(dataset_file) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


async def eval_restaurant_anot(
    anot: G1aANoT,
    restaurant: Dict,
    k: int,
    version: str = "v1"
) -> Dict:
    """
    Evaluate a single restaurant using G1a-ANoT.

    Args:
        anot: G1aANoT instance
        restaurant: Restaurant data dict
        k: Number of reviews
        version: Formula version (v1 or v2)

    Returns:
        Result dict with predictions and ground truth
    """
    name = restaurant['business']['name']

    # Get ground truth
    try:
        gt = compute_gt_for_k(name, k=k, version=version)
    except ValueError:
        return {
            'restaurant': name,
            'k': k,
            'error': 'No GT available',
            'gt_verdict': 'Low Risk',
            'gt_score': 2.75,
            'pred_verdict': None,
            'pred_score': None,
        }

    # Run ANoT evaluation
    try:
        result = await anot.evaluate(restaurant)
    except Exception as e:
        print(f"  ERROR evaluating {name}: {e}")
        return {
            'restaurant': name,
            'k': k,
            'error': str(e),
            'gt_verdict': gt.verdict,
            'gt_score': gt.final_risk_score,
            'pred_verdict': None,
            'pred_score': None,
        }

    return {
        'restaurant': name,
        'k': k,
        'n_reviews': len(restaurant['reviews']),
        'gt_verdict': gt.verdict,
        'gt_score': gt.final_risk_score,
        'gt_incidents': gt.n_total_incidents,
        'pred_verdict': result.get('VERDICT'),
        'pred_score': result.get('FINAL_RISK_SCORE'),
        'pred_incidents': result.get('N_TOTAL_INCIDENTS'),
        'parsed': result,
        'ground_truth': asdict(gt),
    }


async def run_anot_eval(
    task_id: str = "G1a",
    k: int = 200,
    limit: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Run G1a-ANoT evaluation and compute AUPRC metrics.

    Args:
        task_id: Task identifier (e.g., 'G1a')
        k: Number of reviews (K value)
        limit: Limit number of restaurants (None for all)
        verbose: Print per-restaurant results

    Returns:
        Dictionary with AUPRC metrics and individual results
    """
    # Load dataset
    restaurants = load_dataset(k)
    if limit:
        restaurants = restaurants[:limit]

    # Get task config
    task = get_task(task_id)
    version = task.get('version', 'v1')

    print(f"\n{'='*70}")
    print(f"G1a-ANoT Evaluation: {task_id} with K={k} on {len(restaurants)} restaurants")
    print(f"{'='*70}")

    # Initialize ANoT
    anot = G1aANoT(verbose=False)

    # Run evaluations
    results = []
    for i, restaurant in enumerate(restaurants):
        name = restaurant['business']['name']
        if verbose:
            print(f"[{i+1}/{len(restaurants)}] {name[:40]:<40}", end=" ")

        result = await eval_restaurant_anot(anot, restaurant, k, version)
        results.append(result)

        if verbose and not result.get('error'):
            verdict_match = "OK" if result['pred_verdict'] == result['gt_verdict'] else "X"
            print(f"| GT: {result['gt_verdict']:<13} | Pred: {str(result['pred_verdict']):<13} {verdict_match}")
        elif verbose:
            print(f"| ERROR: {result.get('error', 'unknown')}")

    # Process results for AUPRC
    outputs = []
    y_true_ordinal = []
    y_scores = []

    for result in results:
        if result.get('error'):
            continue

        outputs.append(result)

        gt_verdict = result['gt_verdict']
        pred_score = result['pred_score']

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

    # Calculate AUPRC metrics
    if len(y_true_ordinal) > 0:
        y_true_ordinal = np.array(y_true_ordinal)
        y_scores = np.array(y_scores)
        metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

        # Calculate primitive accuracy
        runs_for_prim = [
            {
                'ground_truth': r.get('ground_truth', {}),
                'parsed': r.get('parsed', {})
            }
            for r in outputs
        ]
        avg_prim_acc, per_restaurant_acc = compute_avg_primitive_accuracy(runs_for_prim)

        # Add adjusted metrics
        metrics['avg_primitive_accuracy'] = avg_prim_acc
        metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * avg_prim_acc
        metrics['adjusted_nap'] = metrics['ordinal_nap'] * avg_prim_acc
        metrics['primitive_accuracy_std'] = float(np.std(per_restaurant_acc)) if per_restaurant_acc else 0.0
        metrics['primitive_accuracy_min'] = float(min(per_restaurant_acc)) if per_restaurant_acc else 0.0
        metrics['primitive_accuracy_max'] = float(max(per_restaurant_acc)) if per_restaurant_acc else 0.0
    else:
        metrics = {'error': 'No valid predictions'}

    # Print AUPRC report
    print()
    print_report(metrics)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"{task_id}_anot_k{k}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed = {
        'timestamp': datetime.now().isoformat(),
        'method': 'g1a_anot',
        'task': task_id,
        'k': k,
        'n_restaurants': len(outputs),
        'metrics': metrics,
        'runs': [
            {
                'restaurant': r['restaurant'],
                'results': {
                    'verdict': {'expected': r['gt_verdict'], 'predicted': r['pred_verdict']},
                    'final_risk_score': {'expected': r['gt_score'], 'predicted': r['pred_score']},
                },
                'ground_truth': r.get('ground_truth', {}),
                'parsed': r.get('parsed', {}),
            }
            for r in outputs
        ]
    }

    with open(run_dir / "detailed_results.json", 'w') as f:
        json.dump(detailed, f, indent=2, default=str)

    # Update latest symlink
    latest_link = RESULTS_DIR / "latest_anot"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(run_dir.name, target_is_directory=True)
    except Exception:
        pass

    print(f"\nResults saved to: {run_dir}")

    return {
        'k': k,
        'n': len(outputs),
        'metrics': metrics,
        'outputs': outputs
    }


async def run_comparison(task_id: str = "G1a", limit: int = 20):
    """Compare G1a-ANoT with Direct LLM baseline."""
    from eval import run_eval

    print("\n" + "="*70)
    print("G1a-ANoT vs Direct LLM Comparison")
    print("="*70)

    # Run ANoT
    print("\n>>> G1a-ANoT Evaluation")
    anot_result = await run_anot_eval(task_id, k=200, limit=limit, verbose=False)

    # Run Direct (synchronous wrapper)
    print("\n>>> Direct LLM Baseline")
    direct_result = run_eval(task_id, k=200, limit=limit, verbose=False)

    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Ordinal AUPRC':<15} {'Prim Acc':<12} {'Adjusted AUPRC':<15}")
    print("-"*70)

    anot_m = anot_result['metrics']
    direct_m = direct_result['metrics']

    print(f"{'G1a-ANoT':<20} {anot_m.get('ordinal_auprc', 0):<15.3f} {anot_m.get('avg_primitive_accuracy', 0):<12.3f} {anot_m.get('adjusted_auprc', 0):<15.3f}")
    print(f"{'Direct LLM':<20} {direct_m.get('ordinal_auprc', 0):<15.3f} {direct_m.get('avg_primitive_accuracy', 0):<12.3f} {direct_m.get('adjusted_auprc', 0):<15.3f}")

    # Delta
    delta = anot_m.get('adjusted_auprc', 0) - direct_m.get('adjusted_auprc', 0)
    pct = (delta / max(direct_m.get('adjusted_auprc', 0.001), 0.001)) * 100
    print("-"*70)
    print(f"{'Delta (ANoT - Direct)':<20} {'':<15} {'':<12} {delta:+.3f} ({pct:+.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1a-ANoT Evaluation")
    parser.add_argument("--task", default="G1a", help="Task ID (e.g., G1a)")
    parser.add_argument("--k", type=int, default=200, help="Number of reviews (K value)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of restaurants")
    parser.add_argument("--compare", action="store_true", help="Compare with Direct LLM baseline")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    configure(temperature=0.0)

    if args.compare:
        asyncio.run(run_comparison(args.task, limit=args.limit or 20))
    else:
        asyncio.run(run_anot_eval(args.task, args.k, limit=args.limit, verbose=not args.quiet))
