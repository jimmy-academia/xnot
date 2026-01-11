#!/usr/bin/env python3
"""
G1a-v2 Evaluation with Per-Primitive Accuracy Breakdown

Uses the harder V2 formula with Level 3/4 primitives:
- Level 2: Direct aggregations (n_total_incidents, n_allergy_reviews, trajectory_multiplier, recency_decay, credibility_factor, cuisine_impact)
- Level 3: Derived from Level 2 (trust_score, trust_impact, positive_credit, adjusted_incident_score)
- Level 4: Compound (incident_impact = adjusted_incident_score * trajectory * recency * credibility)

Usage:
    python eval_v2.py                    # Full eval: N=100, K=200
    python eval_v2.py --limit 10         # Quick test
    python eval_v2.py --workers 200      # Adjust parallelism
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm_async, configure
from g1_allergy import TASK_REGISTRY, get_task
from g1_gt_compute import compute_gt_for_k
from score_auprc import calculate_ordinal_auprc, CLASS_ORDER, DEFAULT_TOLERANCES_V2

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Primitive hierarchy for V2
PRIMITIVE_LEVELS = {
    # Level 2: Direct aggregations
    'n_total_incidents': 2,
    'n_allergy_reviews': 2,
    'trajectory_multiplier': 2,
    'recency_decay': 2,
    'credibility_factor': 2,
    'cuisine_impact': 2,
    # Level 3: Derived from Level 2
    'trust_score': 3,       # Depends on n_positive, n_negative, n_betrayal
    'trust_impact': 3,      # = (1 - trust_score) * 3.0
    'positive_credit': 3,   # = n_positive * trust_score * 0.5
    'adjusted_incident_score': 3,  # Depends on trust_score + counts
    # Level 4: Compound
    'incident_impact': 4,   # = adjusted_incident_score * trajectory * recency * credibility
}

SYSTEM_PROMPT_TEMPLATE = '''You are an expert data analyst. Your task is to analyze restaurant review data and compute specific metrics.

## INSTRUCTIONS

1. Read all the data carefully before computing answers
2. Show your intermediate calculations for each metric
3. Be precise with decimal places as specified in the task
4. Complete this analysis in a single response
5. Do NOT ask questions - make reasonable assumptions for edge cases

## RESTAURANT METADATA

{restaurant_data}

## REVIEWS ({n_reviews} total)

{reviews_data}

## TASK

**You MUST calculate ALL values listed below**. Follow the formulas exactly as specified.
Do not skip any primitive, even if it seems redundant or trivial.

{task_prompt}

## OUTPUT FORMAT

After your analysis, output your final answers in EXACTLY this format:

===FINAL ANSWERS===
{output_fields}
===END===

Replace each field with your computed value. Do not include units or extra text in the values.
'''


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


def build_output_fields(ground_truth_class) -> str:
    """Build output field template from ground truth dataclass."""
    from dataclasses import fields as dc_fields
    type_map = {int: '[integer]', float: '[decimal]', str: '[string]'}
    return '\n'.join(f"{f.name.upper()}: {type_map.get(f.type, '[value]')}" for f in dc_fields(ground_truth_class))


def build_prompt(task_id: str, restaurant: Dict, k: int) -> str:
    """Build evaluation prompt for a restaurant."""
    task = get_task(task_id)
    business = restaurant['business']
    reviews = restaurant['reviews']

    restaurant_data = f"""Name: {business['name']}
Categories: {business.get('categories', 'N/A')}
Stars: {business.get('stars', 'N/A')}
Review Count: {len(reviews)}"""

    reviews_data = '\n\n'.join(
        f"[R{i+1}] Date: {r.get('date', 'N/A')} | Stars: {r.get('stars', 'N/A')} | Useful: {r.get('useful', 0)}\n{r.get('text', '')}"
        for i, r in enumerate(reviews)
    )

    output_fields = build_output_fields(task['ground_truth_class'])

    return SYSTEM_PROMPT_TEMPLATE.format(
        restaurant_data=restaurant_data,
        n_reviews=len(reviews),
        reviews_data=reviews_data,
        task_prompt=task['prompt'],
        output_fields=output_fields
    )


def parse_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into field values."""
    parsed = {}

    start_match = re.search(r'===\s*FINAL\s*ANSWERS\s*===', response, re.IGNORECASE)
    if start_match:
        remaining = response[start_match.end():]
        end_match = re.search(r'===\s*END\s*===', remaining, re.IGNORECASE)
        final_block = remaining[:end_match.start()] if end_match else remaining
    else:
        final_block = response

    for line in final_block.split('\n'):
        line = line.strip()
        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip().upper()
        value = value.strip()

        if not value:
            continue

        try:
            if '.' in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            parsed[key] = value.strip('"\'')

    return parsed


def compute_per_primitive_accuracy(
    parsed: Dict,
    ground_truth: Dict,
    tolerances: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Compute accuracy for each primitive with details.

    Returns dict mapping field -> {expected, predicted, correct, tolerance, level}
    """
    results = {}

    for field, tol in tolerances.items():
        if field == 'final_risk_score':
            continue  # Skip final score (captured by AUPRC)

        gt_val = ground_truth.get(field)
        pred_val = parsed.get(field.upper())

        level = PRIMITIVE_LEVELS.get(field, 2)

        if gt_val is None:
            results[field] = {
                'expected': None,
                'predicted': pred_val,
                'correct': None,
                'tolerance': tol,
                'level': level,
                'error': 'No GT'
            }
            continue

        if pred_val is None:
            results[field] = {
                'expected': gt_val,
                'predicted': None,
                'correct': False,
                'tolerance': tol,
                'level': level,
                'error': 'Not predicted'
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
    task_id: str,
    restaurant: Dict,
    k: int,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single restaurant asynchronously with semaphore."""
    async with semaphore:
        name = restaurant['business']['name']
        task = get_task(task_id)
        version = task.get('version', 'v1')

        prompt = build_prompt(task_id, restaurant, k)

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

        try:
            response = await call_llm_async(prompt)
            parsed = parse_response(response)
        except Exception as e:
            return {
                'restaurant': name,
                'k': k,
                'error': str(e),
                'gt_verdict': gt.verdict,
                'gt_score': gt.final_risk_score,
                'pred_verdict': None,
                'pred_score': None,
                'ground_truth': asdict(gt),
            }

        # Compute per-primitive accuracy
        gt_dict = asdict(gt)
        tolerances = task.get('tolerances', DEFAULT_TOLERANCES_V2)
        primitive_results = compute_per_primitive_accuracy(parsed, gt_dict, tolerances)

        return {
            'restaurant': name,
            'k': k,
            'n_reviews': len(restaurant['reviews']),
            'gt_verdict': gt.verdict,
            'gt_score': gt.final_risk_score,
            'pred_verdict': parsed.get('VERDICT'),
            'pred_score': parsed.get('FINAL_RISK_SCORE'),
            'parsed': parsed,
            'ground_truth': gt_dict,
            'primitive_results': primitive_results,
        }


def aggregate_primitive_stats(outputs: List[Dict]) -> Dict[str, Dict]:
    """Aggregate per-primitive statistics across all restaurants."""
    stats = {}

    for output in outputs:
        prim_results = output.get('primitive_results', {})
        for field, result in prim_results.items():
            if field not in stats:
                stats[field] = {
                    'correct': 0,
                    'total': 0,
                    'level': result.get('level', 2),
                    'errors': []
                }

            if result.get('correct') is not None:
                stats[field]['total'] += 1
                if result['correct']:
                    stats[field]['correct'] += 1
                else:
                    stats[field]['errors'].append({
                        'restaurant': output['restaurant'],
                        'expected': result['expected'],
                        'predicted': result['predicted'],
                    })

    # Compute accuracy per primitive
    for field, s in stats.items():
        s['accuracy'] = s['correct'] / s['total'] if s['total'] > 0 else 0.0

    return stats


def print_primitive_breakdown(stats: Dict[str, Dict]):
    """Print per-primitive accuracy breakdown by level."""
    print("\n" + "=" * 80)
    print("PER-PRIMITIVE ACCURACY BREAKDOWN")
    print("=" * 80)

    # Group by level
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

        level_correct = 0
        level_total = 0

        for field, s in sorted(by_level[level]):
            acc = s['accuracy']
            correct = s['correct']
            total = s['total']

            level_correct += correct
            level_total += total
            total_correct += correct
            total_count += total

            marker = "  " if acc >= 0.8 else "* " if acc >= 0.5 else "**"
            print(f"{marker}{field:<28} {acc:>10.1%} {correct:>10} {total:>10}")

        level_acc = level_correct / level_total if level_total > 0 else 0
        print(f"{'Level ' + str(level) + ' Total':<30} {level_acc:>10.1%} {level_correct:>10} {level_total:>10}")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print("\n" + "=" * 80)
    print(f"{'OVERALL PRIMITIVE ACCURACY':<30} {overall_acc:>10.1%} {total_correct:>10} {total_count:>10}")
    print("=" * 80)

    return overall_acc


async def run_eval_async(
    task_id: str,
    k: int,
    limit: Optional[int] = None,
    workers: int = 200,
    verbose: bool = True
) -> Dict:
    """
    Run evaluation with configurable parallelism.

    Args:
        task_id: Task identifier (e.g., 'G1a-v2')
        k: Number of reviews (K value)
        limit: Limit number of restaurants (None for all)
        workers: Number of parallel workers (default 200)
        verbose: Print per-restaurant results
    """
    restaurants = load_dataset(k)
    if limit:
        restaurants = restaurants[:limit]

    print(f"\n{'='*80}")
    print(f"Evaluating {task_id} with K={k} on {len(restaurants)} restaurants")
    print(f"Using {workers} parallel workers")
    print(f"{'='*80}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(workers)

    # Run all evaluations
    tasks = [eval_restaurant_async(task_id, r, k, semaphore) for r in restaurants]
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
                print(f"SKIP: {result['restaurant']} - {result['error']}")
            continue

        outputs.append(result)

        gt_verdict = result['gt_verdict']
        pred_score = result['pred_score']

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

        if verbose:
            verdict_match = "Y" if result['pred_verdict'] == result['gt_verdict'] else "X"
            print(f"{result['restaurant'][:40]:<40} | GT: {result['gt_verdict']:<13} | Pred: {str(result['pred_verdict']):<13} {verdict_match}")

    # Aggregate primitive statistics
    prim_stats = aggregate_primitive_stats(outputs)
    overall_prim_acc = print_primitive_breakdown(prim_stats)

    # Calculate AUPRC metrics
    if len(y_true_ordinal) > 0:
        y_true_ordinal = np.array(y_true_ordinal)
        y_scores = np.array(y_scores)
        metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

        # Add primitive accuracy metrics
        metrics['avg_primitive_accuracy'] = overall_prim_acc
        metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * overall_prim_acc
        metrics['adjusted_nap'] = metrics['ordinal_nap'] * overall_prim_acc

        # Per-primitive accuracies
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
    run_dir = RESULTS_DIR / f"{task_id}_k{k}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed = {
        'timestamp': datetime.now().isoformat(),
        'task': task_id,
        'k': k,
        'workers': workers,
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

    # Update latest symlink
    latest_link = RESULTS_DIR / "latest"
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
    """Synchronous wrapper for async evaluation."""
    return asyncio.run(run_eval_async(task_id, k, limit, workers, verbose))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1a-v2 Evaluation with Per-Primitive Breakdown")
    parser.add_argument("--task", default="G1a-v2", help="Task ID (default: G1a-v2)")
    parser.add_argument("--k", type=int, default=200, help="Number of reviews (K value)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of restaurants")
    parser.add_argument("--workers", type=int, default=200, help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    configure(temperature=0.0)

    run_eval(args.task, args.k, limit=args.limit, workers=args.workers, verbose=not args.quiet)
