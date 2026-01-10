#!/usr/bin/env python3
"""
Evaluation harness for Task v4.

Tests LLM ability to follow multi-step calculation rules.
Reports per-field accuracy and overall metrics.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, configure

from task_v4 import (
    load_reviews, compute_ground_truth, build_prompt, parse_response,
    GroundTruth
)

# Configure for reasoning models
configure(max_tokens_reasoning=32000)


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class FieldResult:
    """Result for a single field."""
    name: str
    expected: any
    predicted: any
    correct: bool
    tolerance: Optional[float] = None


@dataclass
class EvalResult:
    """Complete evaluation result."""
    fields: list[FieldResult]
    verdict_correct: bool
    full_match: bool
    fields_correct: int
    fields_total: int
    raw_response: str

    @property
    def field_accuracy(self) -> float:
        return self.fields_correct / self.fields_total if self.fields_total > 0 else 0


# Tolerances for numeric fields
TOLERANCES = {
    'AVG': 0.1,
    'BASE_SCORE': 2.0,
    'VARIANCE_RATIO': 0.05,
    'TREND_SCORE': 2.0,
    'FINAL_SCORE': 3.0,
}

# Fields that must match exactly (counts and penalties)
EXACT_FIELDS = [
    'N', 'CRITICAL_COUNT', 'SERVICE_COUNT',
    'CRITICAL_PENALTY', 'SERVICE_PENALTY', 'TOTAL_PENALTY',
    'RECENT_POS', 'RECENT_NEG',
    'FIVE_STAR', 'ONE_STAR', 'CONSISTENCY_PENALTY',
    'VERDICT'
]


def evaluate_field(name: str, expected: any, predicted: any) -> FieldResult:
    """Evaluate a single field."""

    if predicted is None:
        return FieldResult(name, expected, predicted, False)

    if name == 'VERDICT':
        correct = (predicted == expected)
        return FieldResult(name, expected, predicted, correct)

    if name in TOLERANCES:
        tolerance = TOLERANCES[name]
        try:
            correct = abs(float(predicted) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            correct = False
        return FieldResult(name, expected, predicted, correct, tolerance)

    # Exact match for counts
    try:
        correct = (int(predicted) == int(expected))
    except (TypeError, ValueError):
        correct = False

    return FieldResult(name, expected, predicted, correct)


def evaluate_response(response: str, gt: GroundTruth) -> EvalResult:
    """Evaluate LLM response against ground truth."""

    parsed = parse_response(response)
    gt_dict = gt.to_dict()

    field_results = []
    for name, expected in gt_dict.items():
        predicted = parsed.get(name)
        result = evaluate_field(name, expected, predicted)
        field_results.append(result)

    fields_correct = sum(1 for r in field_results if r.correct)
    fields_total = len(field_results)

    # Verdict check
    verdict_result = next(r for r in field_results if r.name == 'VERDICT')
    verdict_correct = verdict_result.correct

    # Full match
    full_match = all(r.correct for r in field_results)

    return EvalResult(
        fields=field_results,
        verdict_correct=verdict_correct,
        full_match=full_match,
        fields_correct=fields_correct,
        fields_total=fields_total,
        raw_response=response
    )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    filename: str = "Acme_Oyster_House__ab50qdW.jsonl",
    max_reviews: int = 100,
    verbose: bool = True
) -> EvalResult:
    """Run full evaluation pipeline."""

    # Load data
    meta, reviews = load_reviews(filename, max_reviews)

    if verbose:
        print(f"{'='*60}")
        print(f"EVALUATION: Task v4")
        print(f"{'='*60}")
        print(f"Restaurant: {meta['name']}")
        print(f"Reviews: {len(reviews)}")
        print(f"Date range: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}")

    # Compute ground truth
    gt = compute_ground_truth(reviews)

    if verbose:
        print(f"\n--- Ground Truth ---")
        for k, v in gt.to_dict().items():
            print(f"  {k}: {v}")

    # Build prompt
    prompt = build_prompt(meta, reviews)

    if verbose:
        print(f"\nPrompt length: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
        print("Calling LLM...")

    # Call LLM
    response = call_llm(prompt)

    if verbose:
        print(f"\n--- LLM Response ---")
        # Show just the output section
        if 'N:' in response:
            output_start = response.find('N:')
            print(response[output_start:output_start+800])
        else:
            print(response[:800])
        print("...")

    # Evaluate
    result = evaluate_response(response, gt)

    if verbose:
        print(f"\n--- Evaluation Results ---")
        print(f"\n{'Field':<20} {'Expected':<15} {'Predicted':<15} {'Status'}")
        print("-" * 60)

        for r in result.fields:
            status = "✓" if r.correct else "✗"
            exp_str = str(r.expected)[:12]
            pred_str = str(r.predicted)[:12] if r.predicted is not None else "N/A"
            print(f"{r.name:<20} {exp_str:<15} {pred_str:<15} {status}")

        print("-" * 60)
        print(f"\nFields correct: {result.fields_correct}/{result.fields_total} ({result.field_accuracy*100:.1f}%)")
        print(f"Verdict correct: {'✓' if result.verdict_correct else '✗'}")
        print(f"Full match: {'✓✓✓ YES ✓✓✓' if result.full_match else '✗ NO'}")

    return result


def run_scale_test(
    filename: str = "Acme_Oyster_House__ab50qdW.jsonl",
    scales: list = None,
    verbose: bool = False
):
    """Run evaluation at multiple scales."""

    if scales is None:
        scales = [100, 200, 500]

    print(f"\n{'#'*60}")
    print("SCALE TEST")
    print(f"{'#'*60}")

    results = []
    for n in scales:
        print(f"\n{'='*40}")
        print(f"Scale: {n} reviews")
        print(f"{'='*40}")

        try:
            result = run_evaluation(filename, n, verbose=verbose)
            results.append({
                'scale': n,
                'fields_correct': result.fields_correct,
                'fields_total': result.fields_total,
                'field_accuracy': result.field_accuracy,
                'verdict_correct': result.verdict_correct,
                'full_match': result.full_match,
            })

            print(f"  Fields: {result.fields_correct}/{result.fields_total} ({result.field_accuracy*100:.1f}%)")
            print(f"  Verdict: {'✓' if result.verdict_correct else '✗'}")
            print(f"  Full match: {'✓' if result.full_match else '✗'}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'scale': n,
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SCALE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scale':<10} {'Fields':<15} {'Verdict':<10} {'Full Match'}")
    print("-" * 50)

    for r in results:
        if 'error' in r:
            print(f"{r['scale']:<10} ERROR: {r['error'][:30]}")
        else:
            fields_str = f"{r['fields_correct']}/{r['fields_total']} ({r['field_accuracy']*100:.0f}%)"
            verdict_str = "✓" if r['verdict_correct'] else "✗"
            full_str = "✓" if r['full_match'] else "✗"
            print(f"{r['scale']:<10} {fields_str:<15} {verdict_str:<10} {full_str}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on Task v4")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl")
    parser.add_argument("--max-reviews", type=int, default=100)
    parser.add_argument("--scale-test", action="store_true",
                       help="Run at multiple scales")
    parser.add_argument("--scales", type=str, default="100,200,500",
                       help="Comma-separated scales for scale test")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.scale_test:
        scales = [int(x) for x in args.scales.split(',')]
        run_scale_test(args.file, scales, verbose=not args.quiet)
    else:
        run_evaluation(args.file, args.max_reviews, verbose=not args.quiet)
