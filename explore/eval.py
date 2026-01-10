#!/usr/bin/env python3
"""
L2 Evaluation Framework - Tests LLM ability to compute derived metrics.

Usage:
    python explore/eval.py --task A --restaurant Acme
    python explore/eval.py --task all --restaurant all
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, call_llm_async, configure
from tasks import TASK_REGISTRY, get_task, list_tasks

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Available restaurants - loaded dynamically
RESTAURANTS = sorted([f.name for f in DATA_DIR.glob("*.jsonl") if not f.name.startswith("index")])



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

{task_prompt}

## OUTPUT FORMAT

After your analysis, output your final answers in EXACTLY this format:

===FINAL ANSWERS===
{output_fields}
===END===

Replace each field with your computed value. Do not include units or extra text in the values.
'''


def build_output_fields(ground_truth_class) -> str:
    """Build output field template from ground truth dataclass."""
    from dataclasses import fields as dc_fields
    type_map = {int: '[integer]', float: '[decimal]', str: '[string]'}
    return '\n'.join(f"{f.name.upper()}: {type_map.get(f.type, '[value]')}" for f in dc_fields(ground_truth_class))


def load_restaurant_data(filename: str, max_reviews: int = 100) -> tuple:
    """Load restaurant metadata and reviews as plain dicts."""
    with open(DATA_DIR / filename) as f:
        restaurant = json.loads(f.readline())
        reviews = [json.loads(line) for line in f]
    return restaurant, reviews[-max_reviews:] if max_reviews else reviews


def build_full_prompt(task_id: str, restaurant: dict, reviews: List[dict]) -> str:
    """Build full prompt with system template + task prompt."""
    task = get_task(task_id)

    # Format restaurant as full JSON dump
    restaurant_data = str(restaurant)

    # Format reviews as indexed JSON dumps
    reviews_data = '\n'.join(f"[R{i}] {r}" for i, r in enumerate(reviews, 1))

    # Get output fields from ground truth class
    output_fields = build_output_fields(task['ground_truth_class'])

    # Build full prompt
    return SYSTEM_PROMPT_TEMPLATE.format(
        restaurant_data=restaurant_data,
        n_reviews=len(reviews),
        reviews_data=reviews_data,
        task_prompt=task['prompt'],
        output_fields=output_fields,
    )


def detect_prompt_failure(response: str, parsed: Dict[str, Any], expected_fields: int) -> bool:
    """Return True if model failed due to prompt issues (not computation errors)."""
    numeric_count = sum(1 for v in parsed.values() if isinstance(v, (int, float)))
    if numeric_count >= expected_fields * 0.5:  # Got at least half the fields
        return False
    if '===FINAL' not in response.upper():
        return True
    if any(p in response.lower() for p in ['confirm', 'clarification', 'would you like']):
        return True
    return len(parsed) == 0


def parse_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into field values."""
    parsed = {}

    # Extract final answers block
    final_block = response
    start_match = re.search(r'===\s*FINAL\s*ANSWERS\s*===', response, re.IGNORECASE)
    if start_match:
        remaining = response[start_match.end():]
        end_match = re.search(r'===\s*END\s*===', remaining, re.IGNORECASE)
        if end_match:
            final_block = remaining[:end_match.start()]
        else:
            final_block = remaining

    # Parse lines - handle both "KEY: value" and "KEY:\nvalue" formats
    lines = final_block.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip().upper()
        value = value.strip()

        # If value is empty, check next line
        if not value and i < len(lines):
            next_line = lines[i].strip()
            if next_line and ':' not in next_line:
                value = next_line
                i += 1

        if not value:
            continue

        # Try to parse as number
        try:
            if '.' in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            # Keep as string
            parsed[key] = value.strip('"\'')

    return parsed


def score_field(predicted: Any, expected: Any, tolerance: float = 0) -> float:
    """Score a single field."""
    if predicted is None:
        return 0.0

    # String comparison
    if isinstance(expected, str):
        return 1.0 if str(predicted).upper() == expected.upper() else 0.0

    # Numeric comparison
    try:
        pred_val = float(predicted)
        exp_val = float(expected)
        error = abs(pred_val - exp_val)

        if error <= tolerance:
            return 1.0
        else:
            max_error = max(abs(exp_val), 1.0)
            return max(0.0, 1.0 - (error - tolerance) / max_error)
    except (ValueError, TypeError):
        return 0.0


def evaluate_task(parsed: Dict, ground_truth: Any, tolerances: Dict[str, float] = None) -> Dict:
    """Evaluate parsed response against ground truth."""
    if tolerances is None:
        tolerances = {}

    gt_dict = asdict(ground_truth)
    results = {}

    for field, expected in gt_dict.items():
        field_upper = field.upper()
        predicted = parsed.get(field_upper)
        tolerance = tolerances.get(field, 0.05 if isinstance(expected, float) else 0)
        score = score_field(predicted, expected, tolerance)

        results[field] = {
            'expected': expected,
            'predicted': predicted,
            'score': score,
        }

    results['_total_score'] = sum(r['score'] for r in results.values() if isinstance(r, dict)) / len(gt_dict)

    return results


def _build_task_context(task_id: str, restaurant_file: str, max_reviews: int) -> Dict:
    """Build task context without calling LLM."""
    restaurant, reviews = load_restaurant_data(restaurant_file, max_reviews)
    task = get_task(task_id)
    gt = task['compute_ground_truth'](reviews, restaurant)
    prompt = build_full_prompt(task_id, restaurant, reviews)
    return {
        'task_id': task_id, 'task': task, 'restaurant': restaurant,
        'reviews': reviews, 'gt': gt, 'prompt': prompt, 'restaurant_file': restaurant_file,
    }


def _process_response(ctx: Dict, response: str) -> Dict:
    """Process LLM response and build output data."""
    parsed = parse_response(response)
    expected_fields = len(asdict(ctx['gt']))
    prompt_failure = detect_prompt_failure(response, parsed, expected_fields)
    results = evaluate_task(parsed, ctx['gt'], ctx['task']['tolerances'])
    results['_prompt_failure'] = prompt_failure

    return {
        'timestamp': datetime.now().isoformat(),
        'task': ctx['task_id'],
        'task_name': ctx['task']['name'],
        'restaurant': ctx['restaurant']['name'],
        'n_reviews': len(ctx['reviews']),
        'prompt': ctx['prompt'],
        'llm_response': response,
        'ground_truth': asdict(ctx['gt']),
        'parsed': parsed,
        'prompt_failure': prompt_failure,
        'results': results,
    }


def _print_result(output: Dict, show_response: bool = True):
    """Print a single task result."""
    print(f"\n{'='*60}")
    print(f"Task {output['task']}: {output['task_name']} | {output['restaurant']}")
    print(f"{'='*60}")

    if show_response:
        print(f"{'▼'*20} LLM OUTPUT START {'▼'*20}")
        print(output['llm_response'])
        print(f"{'▲'*20} LLM OUTPUT END {'▲'*22}")

    if output['prompt_failure']:
        print(f"⚠️  PROMPT FAILURE - fix the prompt, not a legitimate computational failure!")

    print(f"--- Results ---")
    for field, result in output['results'].items():
        if field.startswith('_'):
            continue
        status = "✓" if result['score'] >= 0.9 else "~" if result['score'] >= 0.5 else "✗"
        print(f"  {field}: {result['expected']} vs {result['predicted']} ({result['score']:.2f}) {status}")
    print(f"Total Score: {output['results']['_total_score']:.3f}")


def run_task(task_id: str, restaurant_file: str, max_reviews: int = 100, verbose: bool = True, save_output: bool = True) -> Dict:
    """Run a single task on a single restaurant (sync)."""
    ctx = _build_task_context(task_id, restaurant_file, max_reviews)

    if verbose:
        print(f"\nTask {task_id}: {ctx['task']['name']}")
        print(f"Restaurant: {ctx['restaurant']['name']} | Reviews: {len(ctx['reviews'])} | Prompt: {len(ctx['prompt']):,} chars")

    response = call_llm(ctx['prompt'])
    output = _process_response(ctx, response)

    if verbose:
        _print_result(output)
    if save_output:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_file = RESULTS_DIR / f"single_{task_id}_{ctx['restaurant_file']}.json"
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Output: {out_file}")

    return output


async def _run_task_async(task_id: str, restaurant_file: str, max_reviews: int) -> Dict:
    """Run a single task asynchronously (no printing)."""
    ctx = _build_task_context(task_id, restaurant_file, max_reviews)
    response = await call_llm_async(ctx['prompt'])
    return _process_response(ctx, response)


def run_tasks(task_ids: List[str], max_reviews: int = 100, limit: int = None):
    """Run specified tasks on selected restaurants in parallel."""
    selected_restaurants = RESTAURANTS[:limit] if limit else RESTAURANTS
    jobs = [(tid, rf) for rf in selected_restaurants for tid in task_ids]
    n_jobs = len(jobs)

    print(f"Running {n_jobs} jobs in parallel ({len(selected_restaurants)} restaurants × {len(task_ids)} tasks: {task_ids})...")

    async def run_all():
        tasks = [_run_task_async(tid, rf, max_reviews) for tid, rf in jobs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_all())

    # Process results
    all_outputs = []
    scores = {}
    for (task_id, rf), result in zip(jobs, results):
        if isinstance(result, Exception):
            print(f"ERROR [{rf[:10]}][{task_id}]: {result}")
            scores[(rf, task_id)] = 0.0
        else:
            all_outputs.append(result)
            scores[(rf, task_id)] = result['results']['_total_score']
            _print_result(result, show_response=False)

    # Create run directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_file = run_dir / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'runs': all_outputs}, f, indent=2, default=str)

    # Generate summary table
    summary_lines = []
    summary_lines.append(f"{'='*60}\nSUMMARY\n{'='*60}")
    header = f"{'Restaurant':<25}" + "".join(f"{t:<8}" for t in task_ids) + "Avg"
    summary_lines.append(header)

    for rf in selected_restaurants:
        name = rf.replace('.jsonl', '')[:24]
        row = [scores.get((rf, t), 0) for t in task_ids]
        line = f"{name:<25}" + "".join(f"{s:.2f}    " for s in row) + f"{sum(row)/len(row):.2f}"
        summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")
    
    with open(run_dir / "summary_table.txt", 'w') as f:
        f.write(summary_text)

    # Update latest symlink
    latest_link = RESULTS_DIR / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(run_id, target_is_directory=True)
    except Exception:
        pass

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L2 Task Evaluation")
    parser.add_argument("--task", default="A", help="Task ID (A, B, C, D, F) or 'all'")
    parser.add_argument("--restaurant", default="Acme", help="Restaurant name prefix or 'all'")
    parser.add_argument("--max-reviews", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of restaurants to run")
    args = parser.parse_args()

    configure(temperature=0.0)

    if args.task.lower() == 'all' or ',' in args.task or args.restaurant.lower() == 'all':
        if args.task.lower() == 'all':
            task_ids = list_tasks()
        else:
            task_ids = [t.strip().upper() for t in args.task.split(',')]
        
        run_tasks(task_ids, args.max_reviews, args.limit)
    else:
        restaurant_file = next((f for f in RESTAURANTS if args.restaurant.lower() in f.lower()), None)
        if not restaurant_file:
            sys.exit(f"Restaurant not found: {args.restaurant}\\nAvailable: {RESTAURANTS}")
        run_task(args.task.upper(), restaurant_file, args.max_reviews)
