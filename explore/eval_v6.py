#!/usr/bin/env python3
"""
Evaluation harness for Task v6.

Proportional scoring - every field contributes.
No verdict gating.

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

from task_v6 import (
    load_annotated_reviews, compute_ground_truth, build_prompt, parse_response,
    GroundTruth, FIELD_META
)

# Configure for reasoning models
configure(max_tokens_reasoning=32000)

# Debug output directory
DEBUG_DIR = Path(__file__).parent / "debug"


@dataclass
class FieldResult:
    """Result for a single field."""
    name: str
    expected: float
    predicted: Optional[float]
    score: float
    tolerance: float
    depends: list


@dataclass
class EvalResult:
    """Complete evaluation result."""
    fields: list
    final_score: float
    raw_response: str

    @property
    def summary(self) -> str:
        return f"SCORE: {self.final_score:.3f} ({self.final_score*100:.1f}%)"


def field_score(predicted: Optional[float], expected: float, tolerance: float) -> float:
    """Score a single field prediction."""
    if predicted is None:
        return 0.0

    error = abs(predicted - expected)

    if error <= tolerance:
        return 1.0  # Within tolerance = full credit
    else:
        # Proportional penalty beyond tolerance
        # Score decreases linearly from 1.0 to 0.0 as error increases
        max_error = max(abs(expected), 1.0)  # Normalize by expected value
        return max(0.0, 1.0 - (error - tolerance) / max_error)


def evaluate_response(response: str, gt: GroundTruth) -> EvalResult:
    """
    Evaluate LLM response against ground truth.

    Final score = mean of all field scores.
    """
    parsed = parse_response(response)
    gt_dict = gt.to_dict()

    field_results = []
    for name, expected in gt_dict.items():
        meta = FIELD_META[name]
        predicted = parsed.get(name)
        tolerance = meta['tolerance']
        depends = meta['depends']

        score = field_score(predicted, expected, tolerance)

        result = FieldResult(
            name=name,
            expected=expected,
            predicted=predicted,
            score=score,
            tolerance=tolerance,
            depends=depends
        )
        field_results.append(result)

    # Final score = mean of all field scores
    final_score = sum(r.score for r in field_results) / len(field_results)

    return EvalResult(
        fields=field_results,
        final_score=final_score,
        raw_response=response
    )


def save_debug_log(filename: str, max_reviews: int, prompt: str, response: str,
                   gt: GroundTruth, parsed: dict, result: EvalResult):
    """Save full debug information for later analysis."""
    DEBUG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = DEBUG_DIR / f"eval_v6_{timestamp}_{max_reviews}reviews.json"

    debug_data = {
        "timestamp": timestamp,
        "version": "v6",
        "config": {
            "filename": filename,
            "max_reviews": max_reviews,
        },
        "prompt": prompt,
        "response": response,
        "ground_truth": gt.to_dict(),
        "parsed": parsed,
        "evaluation": {
            "final_score": result.final_score,
            "field_scores": {r.name: r.score for r in result.fields},
            "field_results": [
                {
                    "name": r.name,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "score": r.score,
                    "tolerance": r.tolerance,
                    "depends": r.depends,
                }
                for r in result.fields
            ],
        },
    }

    with open(debug_file, "w") as f:
        json.dump(debug_data, f, indent=2, default=str)

    return debug_file


def run_evaluation(
    filename: str = "Acme_Oyster_House__ab50qdW_labeled.jsonl",
    max_reviews: int = 20,
    verbose: bool = True
) -> EvalResult:
    """Run full evaluation pipeline."""

    # Load data
    reviews = load_annotated_reviews(filename, max_reviews)

    if verbose:
        print(f"{'='*60}")
        print(f"EVALUATION: Task v6 (15 Fields, Proportional Scoring)")
        print(f"{'='*60}")
        print(f"Reviews: {len(reviews)}")

    # Compute ground truth
    gt = compute_ground_truth(reviews)

    if verbose:
        print(f"\n--- Ground Truth ---")
        for k, v in gt.to_dict().items():
            print(f"  {k}: {v}")

    # Build prompt
    prompt = build_prompt(reviews)

    if verbose:
        print(f"\nPrompt length: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
        print("Calling LLM...")

    # Call LLM
    response = call_llm(prompt)

    if verbose:
        print(f"\n--- LLM Response (truncated) ---")
        # Show just the output section
        if '===FINAL ANSWERS===' in response:
            output_start = response.find('===FINAL ANSWERS===')
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
        print(f"\n{'Field':<22} {'Expected':<12} {'Predicted':<12} {'Score':<8} {'Deps'}")
        print("-" * 75)

        # Group by tier
        tiers = [
            ('Tier 1', ['N_TOTAL', 'N_1STAR', 'N_5STAR', 'AVG_ALL']),
            ('Tier 2', ['AVG_FIRST_THIRD', 'AVG_LAST_THIRD', 'TREND_DIFF']),
            ('Tier 3', ['N_WAIT_COMPLAINTS', 'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS', 'N_WOULD_NOT_RETURN']),
            ('Tier 4', ['AVG_COMPLAINERS', 'AVG_SATISFIED']),
            ('Tier 5', ['COMPLAINT_RATIO', 'SATISFACTION_GAP']),
        ]

        for tier_name, fields in tiers:
            print(f"\n{tier_name}:")
            for r in result.fields:
                if r.name in fields:
                    exp_str = f"{r.expected}"
                    pred_str = f"{r.predicted}" if r.predicted is not None else "N/A"
                    score_str = f"{r.score:.2f}"
                    deps_str = ",".join(r.depends) if r.depends else "-"
                    status = "✓" if r.score >= 0.9 else "~" if r.score >= 0.5 else "✗"
                    print(f"  {r.name:<20} {exp_str:<12} {pred_str:<12} {score_str:<8} {deps_str} {status}")

        print(f"\n{'='*60}")
        print(f"FINAL SCORE: {result.final_score:.3f} ({result.final_score*100:.1f}%)")
        print(f"{'='*60}")

        # Breakdown by tier
        tier_scores = {}
        for tier_name, fields in tiers:
            tier_results = [r for r in result.fields if r.name in fields]
            tier_avg = sum(r.score for r in tier_results) / len(tier_results)
            tier_scores[tier_name] = tier_avg

        print(f"\nPer-tier breakdown:")
        for tier_name, score in tier_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {tier_name}: {bar} {score*100:.0f}%")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on Task v6")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW_labeled.jsonl")
    parser.add_argument("--max-reviews", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_evaluation(args.file, args.max_reviews, verbose=not args.quiet)
