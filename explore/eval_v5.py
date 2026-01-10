#!/usr/bin/env python3
"""
Evaluation harness for Task v5.

Scoring: SCORE = VERDICT_CORRECT × HARD_FIELD_ACCURACY
If verdict wrong → SCORE = 0

Reports per-field accuracy and final score.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, configure

# Debug output directory
DEBUG_DIR = Path(__file__).parent / "debug"

from task_v5 import (
    load_reviews, compute_ground_truth, build_prompt, parse_response,
    GroundTruth
)

# Configure for reasoning models
configure(max_tokens_reasoning=32000)


def save_debug_log(filename: str, max_reviews: int, prompt: str, response: str,
                   gt: 'GroundTruth', parsed: dict, result: 'EvalResult'):
    """Save full debug information for later analysis."""
    DEBUG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = DEBUG_DIR / f"eval_{timestamp}_{max_reviews}reviews.json"

    debug_data = {
        "timestamp": timestamp,
        "config": {
            "filename": filename,
            "max_reviews": max_reviews,
        },
        "prompt": prompt,
        "response": response,
        "ground_truth": gt.to_dict(),
        "parsed": parsed,
        "evaluation": {
            "verdict_correct": result.verdict_correct,
            "hard_field_accuracy": result.hard_field_accuracy,
            "final_score": result.final_score,
            "field_results": [
                {
                    "name": r.name,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                }
                for r in result.fields
            ],
        },
    }

    with open(debug_file, "w") as f:
        json.dump(debug_data, f, indent=2, default=str)

    return debug_file


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
    fields: list
    verdict_correct: bool
    hard_field_accuracy: float
    final_score: float
    raw_response: str

    @property
    def summary(self) -> str:
        if not self.verdict_correct:
            return f"SCORE: 0.0 (verdict wrong)"
        return f"SCORE: {self.final_score:.2f} ({self.hard_field_accuracy*100:.1f}% hard fields)"


# Tolerances for numeric fields
TOLERANCES = {
    'CRITICAL_COUNT': 1,      # Allow ±1 for counts
    'SERVICE_COUNT': 1,
    'WEIGHTED_AVG': 0.1,      # Allow ±0.1 for float
    'RECOMMEND_SCORE': 3.0,   # Allow ±3 for derived score
}

# Hard fields that count toward scoring (exclude RECOMMEND_SCORE as it's derived)
HARD_FIELDS = [
    'CRITICAL_COUNT', 'SERVICE_COUNT', 'TREND_DIRECTION',
    'WEIGHTED_AVG', 'SERVICE_VS_FOOD', 'TOP_COMPLAINT',
    'POLARIZATION', 'RECENT_MOMENTUM', 'VETO_FLAG',
]


def evaluate_field(name: str, expected: any, predicted: any) -> FieldResult:
    """Evaluate a single field."""

    if predicted is None:
        return FieldResult(name, expected, predicted, False)

    if name in TOLERANCES:
        tolerance = TOLERANCES[name]
        try:
            correct = abs(float(predicted) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            correct = False
        return FieldResult(name, expected, predicted, correct, tolerance)

    # Exact match for categorical
    correct = (str(predicted).upper() == str(expected).upper())
    return FieldResult(name, expected, predicted, correct)


def evaluate_response(response: str, gt: GroundTruth) -> EvalResult:
    """
    Evaluate LLM response against ground truth.

    Score = VERDICT_CORRECT × HARD_FIELD_ACCURACY
    """
    parsed = parse_response(response)
    gt_dict = gt.to_dict()

    field_results = []
    for name, expected in gt_dict.items():
        predicted = parsed.get(name)
        result = evaluate_field(name, expected, predicted)
        field_results.append(result)

    # Check verdict
    verdict_result = next(r for r in field_results if r.name == 'VERDICT')
    verdict_correct = verdict_result.correct

    # Count correct hard fields
    hard_correct = sum(1 for r in field_results if r.name in HARD_FIELDS and r.correct)
    hard_total = len(HARD_FIELDS)
    hard_field_accuracy = hard_correct / hard_total

    # Final score: 0 if verdict wrong, else hard field accuracy
    final_score = hard_field_accuracy if verdict_correct else 0.0

    return EvalResult(
        fields=field_results,
        verdict_correct=verdict_correct,
        hard_field_accuracy=hard_field_accuracy,
        final_score=final_score,
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
        print(f"EVALUATION: Task v5 (10 Hard Fields)")
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
        if 'CRITICAL_COUNT:' in response:
            output_start = response.find('CRITICAL_COUNT:')
            print(response[output_start:output_start+1000])
        else:
            print(response[:1000])
        print("...")

    # Evaluate
    result = evaluate_response(response, gt)

    # Save debug log
    parsed = parse_response(response)
    debug_file = save_debug_log(filename, max_reviews, prompt, response, gt, parsed, result)

    if verbose:
        print(f"\nDebug log saved: {debug_file}")
        print(f"\n--- Evaluation Results ---")
        print(f"\n{'Field':<20} {'Expected':<20} {'Predicted':<20} {'Status'}")
        print("-" * 70)

        for r in result.fields:
            status = "✓" if r.correct else "✗"
            is_hard = "★" if r.name in HARD_FIELDS else ""
            exp_str = str(r.expected)[:18]
            pred_str = str(r.predicted)[:18] if r.predicted is not None else "N/A"
            print(f"{r.name:<20} {exp_str:<20} {pred_str:<20} {status}{is_hard}")

        print("-" * 70)
        print(f"\nVerdict correct: {'✓' if result.verdict_correct else '✗'}")
        print(f"Hard fields correct: {sum(1 for r in result.fields if r.name in HARD_FIELDS and r.correct)}/{len(HARD_FIELDS)}")
        print(f"Hard field accuracy: {result.hard_field_accuracy*100:.1f}%")
        print(f"\n{'='*60}")
        print(f"FINAL SCORE: {result.final_score:.2f}")
        print(f"{'='*60}")

        if result.final_score == 0:
            print("(Score is 0 because verdict was wrong)")

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
    print("SCALE TEST: Task v5")
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
                'verdict_correct': result.verdict_correct,
                'hard_accuracy': result.hard_field_accuracy,
                'final_score': result.final_score,
            })

            print(f"  Verdict: {'✓' if result.verdict_correct else '✗'}")
            print(f"  Hard fields: {result.hard_field_accuracy*100:.1f}%")
            print(f"  SCORE: {result.final_score:.2f}")

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
    print(f"{'Scale':<10} {'Verdict':<10} {'Hard Fields':<15} {'SCORE'}")
    print("-" * 50)

    for r in results:
        if 'error' in r:
            print(f"{r['scale']:<10} ERROR: {r['error'][:30]}")
        else:
            verdict_str = "✓" if r['verdict_correct'] else "✗"
            hard_str = f"{r['hard_accuracy']*100:.1f}%"
            print(f"{r['scale']:<10} {verdict_str:<10} {hard_str:<15} {r['final_score']:.2f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on Task v5")
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
