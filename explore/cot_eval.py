#!/usr/bin/env python3
"""
Chain of Thought Evaluation for G1a

Tests whether explicit CoT prompting improves primitive accuracy.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm_async, configure

from g1_gt_compute import compute_gt_for_k
from g1_allergy import get_task, TASK_G1_PROMPT
from score_auprc import (
    calculate_ordinal_auprc, compute_avg_primitive_accuracy,
    print_report, CLASS_ORDER
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

COT_SYSTEM_PROMPT = '''You are an expert data analyst evaluating restaurant peanut allergy safety.

Your task is to analyze reviews and compute specific metrics using CHAIN OF THOUGHT reasoning.

## IMPORTANT: Show your work step by step!

For EACH step, explicitly:
1. Quote relevant review excerpts
2. Classify the signal (severity, account type, interaction)
3. Keep running counts
4. Show arithmetic calculations

## RESTAURANT METADATA

{restaurant_data}

## REVIEWS ({n_reviews} total)

{reviews_data}

## TASK

{task_prompt}

## CHAIN OF THOUGHT FORMAT

Think step by step:

STEP 1: SCAN FOR ALLERGY KEYWORDS
- List reviews mentioning allergies, reactions, dietary restrictions
- For each match, quote the relevant excerpt

STEP 2: CLASSIFY EACH RELEVANT REVIEW
- For each: incident_severity (none/mild/moderate/severe)
- For each: account_type (none/firsthand/secondhand/hypothetical)
- For each: safety_interaction (none/positive/negative/betrayal)

STEP 3: AGGREGATE COUNTS
- N_MILD = <count firsthand mild incidents>
- N_MODERATE = <count firsthand moderate incidents>
- N_SEVERE = <count firsthand severe incidents>
- N_TOTAL_INCIDENTS = N_MILD + N_MODERATE + N_SEVERE
- N_POSITIVE, N_NEGATIVE, N_BETRAYAL = <count safety interactions>

STEP 4: COMPUTE DERIVED VALUES
- Show INCIDENT_SCORE calculation: (N_MILD × 2) + (N_MODERATE × 5) + (N_SEVERE × 15) = ?
- Show RECENCY_DECAY calculation: max(0.3, 1.0 - (age × 0.15)) = ?
- Show CREDIBILITY_FACTOR calculation: total_weight / max(N_TOTAL, 1) = ?

STEP 5: COMPUTE FINAL SCORE
- Show full formula calculation
- FINAL_RISK_SCORE = ?
- VERDICT = ?

## OUTPUT FORMAT

After your analysis, output your final answers in EXACTLY this format:

===FINAL ANSWERS===
{output_fields}
===END===
'''


def load_dataset(k: int) -> List[Dict]:
    """Load dataset for given K value."""
    dataset_file = DATA_DIR / f"dataset_K{k}.jsonl"
    restaurants = []
    with open(dataset_file) as f:
        for line in f:
            restaurants.append(json.loads(line))
    return restaurants


def build_cot_prompt(restaurant: Dict, task_prompt: str) -> str:
    """Build CoT evaluation prompt for a restaurant."""
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

    output_fields = """N_TOTAL_INCIDENTS: [integer]
INCIDENT_SCORE: [decimal]
RECENCY_DECAY: [decimal]
CREDIBILITY_FACTOR: [decimal]
FINAL_RISK_SCORE: [decimal]
VERDICT: [string]"""

    return COT_SYSTEM_PROMPT.format(
        restaurant_data=restaurant_data,
        n_reviews=len(reviews),
        reviews_data=reviews_data,
        task_prompt=task_prompt,
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


async def eval_restaurant_cot(restaurant: Dict, k: int, task_prompt: str) -> Dict:
    """Evaluate a single restaurant using Chain of Thought."""
    name = restaurant['business']['name']

    try:
        gt = compute_gt_for_k(name, k=k)
    except ValueError:
        return {
            'restaurant': name,
            'error': 'No GT available',
            'gt_verdict': 'Low Risk',
            'gt_score': 2.75,
        }

    prompt = build_cot_prompt(restaurant, task_prompt)
    response = await call_llm_async(prompt)
    parsed = parse_response(response)

    return {
        'restaurant': name,
        'k': k,
        'n_reviews': len(restaurant['reviews']),
        'gt_verdict': gt.verdict,
        'gt_score': gt.final_risk_score,
        'pred_verdict': parsed.get('VERDICT'),
        'pred_score': parsed.get('FINAL_RISK_SCORE'),
        'parsed': parsed,
        'ground_truth': asdict(gt),
    }


async def run_cot_eval(k: int = 200, limit: Optional[int] = None, verbose: bool = True) -> Dict:
    """Run Chain of Thought evaluation."""
    restaurants = load_dataset(k)
    if limit:
        restaurants = restaurants[:limit]

    task = get_task("G1a")
    task_prompt = task['prompt']

    print(f"\n{'='*70}")
    print(f"Chain of Thought Evaluation: G1a with K={k} on {len(restaurants)} restaurants")
    print(f"{'='*70}")

    # Run evaluations
    tasks = [eval_restaurant_cot(r, k, task_prompt) for r in restaurants]
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
            continue

        outputs.append(result)

        gt_verdict = result['gt_verdict']
        pred_score = result['pred_score']

        if gt_verdict in CLASS_ORDER and pred_score is not None:
            y_true_ordinal.append(CLASS_ORDER[gt_verdict])
            y_scores.append(float(pred_score))

        if verbose:
            verdict_match = "OK" if result['pred_verdict'] == result['gt_verdict'] else "X"
            print(f"{result['restaurant'][:40]:<40} | GT: {result['gt_verdict']:<13} | Pred: {str(result['pred_verdict']):<13} {verdict_match}")

    # Calculate AUPRC
    if len(y_true_ordinal) > 0:
        y_true_ordinal = np.array(y_true_ordinal)
        y_scores = np.array(y_scores)
        metrics = calculate_ordinal_auprc(y_true_ordinal, y_scores)

        runs_for_prim = [
            {'ground_truth': r.get('ground_truth', {}), 'parsed': r.get('parsed', {})}
            for r in outputs
        ]
        avg_prim_acc, per_restaurant_acc = compute_avg_primitive_accuracy(runs_for_prim)

        metrics['avg_primitive_accuracy'] = avg_prim_acc
        metrics['adjusted_auprc'] = metrics['ordinal_auprc'] * avg_prim_acc
        metrics['adjusted_nap'] = metrics['ordinal_nap'] * avg_prim_acc
        metrics['primitive_accuracy_std'] = float(np.std(per_restaurant_acc)) if per_restaurant_acc else 0.0
        metrics['primitive_accuracy_min'] = float(min(per_restaurant_acc)) if per_restaurant_acc else 0.0
        metrics['primitive_accuracy_max'] = float(max(per_restaurant_acc)) if per_restaurant_acc else 0.0
    else:
        metrics = {'error': 'No valid predictions'}

    print()
    print_report(metrics)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"G1a_cot_k{k}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed = {
        'timestamp': datetime.now().isoformat(),
        'method': 'chain_of_thought',
        'task': 'G1a',
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

    print(f"\nResults saved to: {run_dir}")

    return {'k': k, 'n': len(outputs), 'metrics': metrics, 'outputs': outputs}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chain of Thought Evaluation")
    parser.add_argument("--k", type=int, default=200, help="Number of reviews")
    parser.add_argument("--limit", type=int, default=None, help="Limit restaurants")
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    configure(temperature=0.0)
    asyncio.run(run_cot_eval(args.k, args.limit, verbose=not args.quiet))
