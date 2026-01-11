#!/usr/bin/env python3
"""
General ANoT Evaluation on G1a-v2 Benchmark

Runs the General ANoT framework on any task prompt. The framework:
1. Phase 1: Dynamically generates FormulaSeed from task prompt (LLM figures out computations)
2. Phase 2: Executes FormulaSeed on each restaurant

This is truly task-agnostic - no pre-generated seeds, the LLM discovers the required
computations from the task description alone.

Usage:
    python general_anot_eval.py                    # Full eval: N=100, K=200
    python general_anot_eval.py --limit 10         # Quick test
    python general_anot_eval.py --workers 200      # Adjust parallelism
    python general_anot_eval.py --task G1a         # Use V1 formula
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

# Phase 1: Dynamic FormulaSeed generation
from general_anot.phase1_step1 import extract_conditions
from general_anot.phase1_step2 import build_computation_graph
from general_anot.phase1_step3 import generate_formula_seed, FormulaSeed

# Phase 2: FormulaSeed execution
from general_anot.phase2 import Phase2Executor

from g1_gt_compute import compute_gt_for_k
from g1_allergy import get_task
from score_auprc import calculate_ordinal_auprc, CLASS_ORDER, DEFAULT_TOLERANCES_V1, DEFAULT_TOLERANCES_V2

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Primitive levels for reporting
PRIMITIVE_LEVELS = {
    # Level 2
    'n_total_incidents': 2,
    'n_allergy_reviews': 2,
    'trajectory_multiplier': 2,
    'recency_decay': 2,
    'credibility_factor': 2,
    'cuisine_impact': 2,
    'incident_score': 2,
    # Level 3
    'trust_score': 3,
    'trust_impact': 3,
    'positive_credit': 3,
    'adjusted_incident_score': 3,
    # Level 4
    'incident_impact': 4,
}


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


async def generate_formula_seed_dynamic(task_prompt: str, task_name: str, verbose: bool = True) -> FormulaSeed:
    """
    Dynamically generate FormulaSeed from task prompt using Phase 1.

    This is the core of General ANoT's task-agnostic design:
    - Given ANY task prompt, the LLM figures out what to extract and compute
    - No pre-generated seeds, no task-specific code

    Args:
        task_prompt: The task specification (e.g., TASK_G1_PROMPT_V2)
        task_name: Name identifier for the task
        verbose: Print progress

    Returns:
        FormulaSeed ready for Phase 2 execution
    """
    if verbose:
        print(f"\n[Phase 1] Generating FormulaSeed from task prompt...")
        print(f"  Task: {task_name}")

    # Step 1.1: Extract conditions (what to filter/extract/aggregate)
    if verbose:
        print(f"  [Step 1.1] Extracting conditions...")
    conditions = await extract_conditions(task_prompt, verbose=False)
    if verbose:
        print(f"    - {len(conditions.extraction_fields)} extraction fields")
        print(f"    - {len(conditions.filter_condition.keywords if conditions.filter_condition else [])} filter keywords")
        print(f"    - {len(conditions.aggregation_conditions)} aggregation conditions")

    # Step 1.2: Build computation graph (formulas in dependency order)
    if verbose:
        print(f"  [Step 1.2] Building computation graph...")
    graph = await build_computation_graph(task_prompt, conditions, verbose=False)
    if verbose:
        print(f"    - {len(graph.lookups)} lookup tables")
        print(f"    - {len(graph.computations)} computation steps")
        print(f"    - {len(graph.output_fields)} output fields")

    # Step 1.3: Generate FormulaSeed (combine into executable spec)
    if verbose:
        print(f"  [Step 1.3] Generating FormulaSeed...")
    seed = generate_formula_seed(task_name, conditions, graph, verbose=False)
    if verbose:
        print(f"    - {len(seed.extraction_fields)} extraction fields")
        print(f"    - {len(seed.computations)} computations")
        print(f"[Phase 1] FormulaSeed generated successfully")

    return seed


def compute_per_primitive_accuracy(
    parsed: Dict,
    ground_truth: Dict,
    tolerances: Dict[str, float]
) -> Dict[str, Dict]:
    """Compute accuracy for each primitive with details."""
    results = {}

    for field, tol in tolerances.items():
        if field == 'final_risk_score':
            continue

        gt_val = ground_truth.get(field)
        # Try both cases for predicted value
        pred_val = parsed.get(field.upper()) or parsed.get(field)

        level = PRIMITIVE_LEVELS.get(field, 2)

        if gt_val is None:
            continue

        if pred_val is None:
            results[field] = {
                'expected': gt_val,
                'predicted': None,
                'correct': False,
                'tolerance': tol,
                'level': level,
            }
            continue

        # Check if within tolerance
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            correct = abs(float(gt_val) - float(pred_val)) <= tol
        else:
            correct = gt_val == pred_val

        results[field] = {
            'expected': gt_val,
            'predicted': pred_val,
            'correct': correct,
            'tolerance': tol,
            'level': level,
        }

    return results


async def eval_restaurant_async(
    executor: Phase2Executor,
    restaurant: Dict,
    k: int,
    version: str,
    tolerances: Dict[str, float],
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single restaurant asynchronously."""
    async with semaphore:
        name = restaurant['business']['name']

        # Get ground truth
        try:
            gt = compute_gt_for_k(name, k=k, version=version)
        except ValueError:
            return {
                'restaurant': name,
                'error': 'No GT available',
            }

        # Run General ANoT
        try:
            result = await executor.execute(restaurant)
        except Exception as e:
            return {
                'restaurant': name,
                'error': str(e),
                'gt_verdict': gt.verdict,
                'gt_score': gt.final_risk_score,
            }

        # Compute per-primitive accuracy
        gt_dict = asdict(gt)
        primitive_results = compute_per_primitive_accuracy(result, gt_dict, tolerances)

        return {
            'restaurant': name,
            'k': k,
            'gt_verdict': gt.verdict,
            'gt_score': gt.final_risk_score,
            'pred_verdict': result.get('VERDICT'),
            'pred_score': result.get('FINAL_RISK_SCORE'),
            'parsed': result,
            'ground_truth': gt_dict,
            'primitive_results': primitive_results,
        }


def aggregate_primitive_stats(outputs: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-primitive statistics."""
    stats = {}

    for output in outputs:
        prim_results = output.get('primitive_results', {})
        for field, result in prim_results.items():
            if field not in stats:
                stats[field] = {
                    'correct': 0,
                    'total': 0,
                    'level': result.get('level', 2),
                }

            if result.get('correct') is not None:
                stats[field]['total'] += 1
                if result['correct']:
                    stats[field]['correct'] += 1

    for field, s in stats.items():
        s['accuracy'] = s['correct'] / s['total'] if s['total'] > 0 else 0.0

    return stats


def print_primitive_breakdown(stats: Dict[str, Dict]) -> float:
    """Print per-primitive accuracy breakdown."""
    print("\n" + "=" * 80)
    print("PER-PRIMITIVE ACCURACY BREAKDOWN")
    print("=" * 80)

    by_level = {}
    for field, s in stats.items():
        level = s['level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append((field, s))

    total_correct = 0
    total_count = 0

    for level in sorted(by_level.keys()):
        print(f"\n--- Level {level} Primitives ---")
        print(f"{'Primitive':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print("-" * 60)

        for field, s in sorted(by_level[level]):
            acc = s['accuracy']
            correct = s['correct']
            total = s['total']
            total_correct += correct
            total_count += total

            marker = "  " if acc >= 0.8 else "* " if acc >= 0.5 else "**"
            print(f"{marker}{field:<28} {acc:>10.1%} {correct:>10} {total:>10}")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print("\n" + "=" * 80)
    print(f"{'OVERALL PRIMITIVE ACCURACY':<30} {overall_acc:>10.1%} {total_correct:>10} {total_count:>10}")
    print("=" * 80)

    return overall_acc


async def run_general_anot_eval(
    task_id: str = "G1a-v2",
    k: int = 200,
    limit: Optional[int] = None,
    workers: int = 200,
    verbose: bool = True
) -> Dict:
    """
    Run General ANoT evaluation.

    Args:
        task_id: Task identifier (G1a or G1a-v2)
        k: Number of reviews (K value)
        limit: Limit number of restaurants
        workers: Number of parallel workers
        verbose: Print per-restaurant results
    """
    # Load dataset
    restaurants = load_dataset(k)
    if limit:
        restaurants = restaurants[:limit]

    # Get task config
    task = get_task(task_id)
    version = task.get('version', 'v1')
    tolerances = task.get('tolerances', DEFAULT_TOLERANCES_V1 if version == 'v1' else DEFAULT_TOLERANCES_V2)
    task_prompt = task.get('prompt')

    print(f"\n{'='*80}")
    print(f"General ANoT Evaluation: {task_id} with K={k} on {len(restaurants)} restaurants")
    print(f"Using {workers} parallel workers")
    print(f"{'='*80}")

    # Phase 1: Dynamically generate FormulaSeed from task prompt
    # This is what makes General ANoT task-agnostic - the LLM figures out
    # what computations are needed from the task description alone
    seed = await generate_formula_seed_dynamic(task_prompt, task_id, verbose=verbose)

    print(f"\n[Phase 2] Executing FormulaSeed on {len(restaurants)} restaurants...")

    # Initialize executor
    executor = Phase2Executor(seed, verbose=False)

    # Create semaphore
    semaphore = asyncio.Semaphore(workers)

    # Run evaluations
    tasks = [
        eval_restaurant_async(executor, r, k, version, tolerances, semaphore)
        for r in restaurants
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    outputs = []
    y_true_ordinal = []
    y_scores = []

    for result in results:
        if isinstance(result, Exception):
            print(f"ERROR: {result}")
            continue
        if result.get('error'):
            if verbose:
                print(f"SKIP: {result['restaurant']} - {result.get('error')}")
            continue

        outputs.append(result)

        gt_verdict = result['gt_verdict']
        pred_score = result['pred_score']

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

        if verbose:
            match = "Y" if result['pred_verdict'] == result['gt_verdict'] else "X"
            print(f"{result['restaurant'][:40]:<40} | GT: {result['gt_verdict']:<13} | Pred: {str(result['pred_verdict']):<13} {match}")

    # Aggregate primitive stats
    prim_stats = aggregate_primitive_stats(outputs)
    overall_prim_acc = print_primitive_breakdown(prim_stats)

    # Calculate AUPRC
    if len(y_true_ordinal) > 0:
        y_true_ordinal = np.array(y_true_ordinal)
        y_scores = np.array(y_scores)
        metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

        metrics['avg_primitive_accuracy'] = overall_prim_acc
        metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * overall_prim_acc
        metrics['adjusted_nap'] = metrics['ordinal_nap'] * overall_prim_acc
        metrics['per_primitive'] = {
            field: {'accuracy': s['accuracy'], 'level': s['level']}
            for field, s in prim_stats.items()
        }
    else:
        metrics = {'error': 'No valid predictions'}

    # Print final report
    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"Method:               General ANoT")
    print(f"Task:                 {task_id}")
    print(f"Samples:              {metrics.get('n_samples', len(outputs))}")
    print(f"Distribution:         {metrics.get('distribution', {})}")
    print()
    print(f"Ordinal AUPRC:        {metrics.get('ordinal_auprc', 0):.3f}")
    print(f"Primitive Accuracy:   {metrics.get('avg_primitive_accuracy', 0):.3f}")
    print(f"Adjusted AUPRC:       {metrics.get('adjusted_auprc', 0):.3f}  (= AUPRC x PrimAcc)")
    print("=" * 80)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"GeneralANoT_{task_id}_k{k}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed = {
        'timestamp': datetime.now().isoformat(),
        'method': 'general_anot',
        'task': task_id,
        'k': k,
        'workers': workers,
        'seed_task': seed.task_name,
        'seed_generated_dynamically': True,  # Key: seed was NOT loaded from file
        'n_restaurants': len(outputs),
        'metrics': metrics,
        'primitive_stats': {
            field: {'accuracy': s['accuracy'], 'level': s['level'], 'correct': s['correct'], 'total': s['total']}
            for field, s in prim_stats.items()
        },
        'runs': [
            {
                'restaurant': r['restaurant'],
                'results': {
                    'verdict': {'expected': r['gt_verdict'], 'predicted': r['pred_verdict']},
                    'final_risk_score': {'expected': r['gt_score'], 'predicted': r['pred_score']},
                },
                'ground_truth': r.get('ground_truth', {}),
                'parsed': r.get('parsed', {}),
                'primitive_results': r.get('primitive_results', {}),
            }
            for r in outputs
        ]
    }

    with open(run_dir / "detailed_results.json", 'w') as f:
        json.dump(detailed, f, indent=2, default=str)

    # Save the dynamically generated FormulaSeed for inspection
    with open(run_dir / "generated_seed.json", 'w') as f:
        json.dump(seed.to_dict(), f, indent=2, default=str)

    # Update symlink
    latest_link = RESULTS_DIR / "latest_general_anot"
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
        'primitive_stats': prim_stats,
    }


def run_eval(task_id: str, k: int, limit: Optional[int] = None, workers: int = 200, verbose: bool = True) -> Dict:
    """Synchronous wrapper."""
    return asyncio.run(run_general_anot_eval(task_id, k, limit, workers, verbose))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="General ANoT Evaluation")
    parser.add_argument("--task", default="G1a-v2", help="Task ID (G1a or G1a-v2)")
    parser.add_argument("--k", type=int, default=200, help="Number of reviews")
    parser.add_argument("--limit", type=int, default=None, help="Limit restaurants")
    parser.add_argument("--workers", type=int, default=200, help="Parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    configure(temperature=0.0)

    run_eval(args.task, args.k, limit=args.limit, workers=args.workers, verbose=not args.quiet)
