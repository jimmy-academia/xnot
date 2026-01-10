#!/usr/bin/env python3
"""
G1a Evaluation Framework with Cumulative Ordinal AUPRC Scoring.

Uses dynamic GT per K - ground truth computed from same reviews model sees.

Usage:
    python eval.py --task G1a --limit 10    # Quick test (K=200 default)
    python eval.py --task G1a               # Full eval on all 100 restaurants
    python eval.py --task G1a --k 50        # Use K=50 instead
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
from utils.llm import call_llm, call_llm_async, configure
from g1_allergy import TASK_REGISTRY, get_task, list_tasks
from g1_gt_compute import compute_gt_for_k
from score_auprc import calculate_ordinal_auprc, print_report, CLASS_ORDER

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


SYSTEM_PROMPT_TEMPLATE = '''You are an expert data analyst. Your task is to analyze restaurant review data and compute specific metrics.

## INSTRUCTIONS

1. Read all the data carefully before computing answers
2. Show your intermediate calculations for each metric
3. Be precise with decimal places as specified in the task
4. Complete this analysis in a single response
5. Do NOT ask questions or request clarification - make reasonable assumptions for edge cases

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
    reviews = restaurant['reviews']  # Already limited to K by dataset

    # Format restaurant metadata
    restaurant_data = f"""Name: {business['name']}
Categories: {business.get('categories', 'N/A')}
Stars: {business.get('stars', 'N/A')}
Review Count: {len(reviews)}"""

    # Format reviews with index
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

    # Extract final answers block
    start_match = re.search(r'===\s*FINAL\s*ANSWERS\s*===', response, re.IGNORECASE)
    if start_match:
        remaining = response[start_match.end():]
        end_match = re.search(r'===\s*END\s*===', remaining, re.IGNORECASE)
        final_block = remaining[:end_match.start()] if end_match else remaining
    else:
        final_block = response

    # Parse lines
    for line in final_block.split('\n'):
        line = line.strip()
        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip().upper()
        value = value.strip()

        if not value:
            continue

        # Try to parse as number
        try:
            if '.' in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            parsed[key] = value.strip('"\'')

    return parsed


async def eval_restaurant_async(
    task_id: str,
    restaurant: Dict,
    k: int
) -> Dict:
    """Evaluate a single restaurant asynchronously."""
    name = restaurant['business']['name']
    task = get_task(task_id)

    # Build prompt
    prompt = build_prompt(task_id, restaurant, k)

    # Get dynamic GT for this K
    try:
        gt = compute_gt_for_k(name, k=k)
    except ValueError:
        # No judgments for this restaurant
        return {
            'restaurant': name,
            'k': k,
            'error': 'No GT available',
            'gt_verdict': 'Low Risk',
            'gt_score': 2.75,
            'pred_verdict': None,
            'pred_score': None,
        }

    # Call LLM
    response = await call_llm_async(prompt)

    # Parse response
    parsed = parse_response(response)

    return {
        'restaurant': name,
        'k': k,
        'n_reviews': len(restaurant['reviews']),
        'gt_verdict': gt.verdict,
        'gt_score': gt.final_risk_score,
        'gt_incidents': gt.n_total_incidents,
        'pred_verdict': parsed.get('VERDICT'),
        'pred_score': parsed.get('FINAL_RISK_SCORE'),
        'pred_incidents': parsed.get('N_TOTAL_INCIDENTS'),
        'parsed': parsed,
        'ground_truth': asdict(gt),
    }


def run_eval(task_id: str, k: int, limit: Optional[int] = None, verbose: bool = True) -> Dict:
    """
    Run evaluation and compute AUPRC metrics.

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

    print(f"\n{'='*70}")
    print(f"Evaluating {task_id} with K={k} on {len(restaurants)} restaurants")
    print(f"{'='*70}")

    # Run async evaluation
    async def run_all():
        tasks = [eval_restaurant_async(task_id, r, k) for r in restaurants]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_all())

    # Process results
    outputs = []
    y_true_ordinal = []
    y_scores = []

    for result in results:
        if isinstance(result, Exception):
            print(f"ERROR: {result}")
            continue
        if result.get('error'):
            continue

        outputs.append(result)

        # Collect for AUPRC
        gt_verdict = result['gt_verdict']
        pred_score = result['pred_score']

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

        # Print per-restaurant result
        if verbose:
            verdict_match = "✓" if result['pred_verdict'] == result['gt_verdict'] else "✗"
            print(f"{result['restaurant'][:35]:<35} | GT: {result['gt_verdict']:<13} | Pred: {str(result['pred_verdict']):<13} {verdict_match}")

    # Calculate AUPRC metrics
    if len(y_true_ordinal) > 0:
        y_true_ordinal = np.array(y_true_ordinal)
        y_scores = np.array(y_scores)
        metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)
    else:
        metrics = {'error': 'No valid predictions'}

    # Print AUPRC report
    print()
    print_report(metrics)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"{task_id}_k{k}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results (compatible with score_auprc.py)
    detailed = {
        'timestamp': datetime.now().isoformat(),
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
        'outputs': outputs
    }


def run_comparison(task_id: str, limit: int = 10):
    """Compare performance across different K values."""
    print("\n" + "="*70)
    print(f"{task_id} PERFORMANCE COMPARISON ACROSS K VALUES")
    print("="*70)

    all_results = []
    for k in [25, 50, 100, 200]:
        print(f"\n{'='*70}")
        print(f"K = {k}")
        print("="*70)
        result = run_eval(task_id, k, limit=limit, verbose=False)
        all_results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'K':<8} {'N':<6} {'AUPRC≥High':<12} {'AUPRC≥Crit':<12} {'Ordinal AUPRC':<14} {'Ordinal nAP':<12}")
    print("-"*70)

    for r in all_results:
        m = r['metrics']
        print(f"{r['k']:<8} {r['n']:<6} {m.get('auprc_ge_1', 0):<12.3f} {m.get('auprc_ge_2', 0):<12.3f} {m.get('ordinal_auprc', 0):<14.3f} {m.get('ordinal_nap', 0):<12.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation with Ordinal AUPRC")
    parser.add_argument("--task", default="G1a", help="Task ID (e.g., G1a)")
    parser.add_argument("--k", type=int, default=200, help="Number of reviews (K value)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of restaurants")
    parser.add_argument("--compare", action="store_true", help="Compare across K values")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    configure(temperature=0.0)

    if args.compare:
        run_comparison(args.task, limit=args.limit or 10)
    else:
        run_eval(args.task, args.k, limit=args.limit, verbose=not args.quiet)
