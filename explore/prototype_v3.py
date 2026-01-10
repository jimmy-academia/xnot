#!/usr/bin/env python3
"""
Prototype v3: Decision-based benchmark with multi-point evaluation.

Tests if LLMs can:
1. Follow complex decision rules
2. Correctly compute intermediate conditions
3. Arrive at the correct final verdict

Evaluation is multi-point: each condition is scored separately.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, configure

from scenarios import ALL_SCENARIOS, Scenario, GroundTruth

configure(max_tokens_reasoning=32000)

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ConditionResult:
    """Result for a single condition."""
    condition_id: str
    expected: any
    predicted: any
    correct: bool
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Full evaluation result for a scenario."""
    scenario: str
    conditions: list[ConditionResult]
    verdict_expected: bool
    verdict_predicted: Optional[bool]
    verdict_correct: bool
    full_match: bool  # All conditions correct AND verdict correct
    raw_response: str


def load_reviews(filename: str, max_reviews: int = None) -> tuple[dict, list[dict]]:
    """Load restaurant data."""
    filepath = DATA_DIR / filename
    with open(filepath) as f:
        meta = json.loads(f.readline())
        all_reviews = [json.loads(line) for line in f]

    if max_reviews and max_reviews < len(all_reviews):
        reviews = all_reviews[-max_reviews:]  # Most recent
    else:
        reviews = all_reviews

    return meta, reviews


def build_prompt(meta: dict, reviews: list[dict], scenario: Scenario) -> str:
    """Build the evaluation prompt."""

    # Format reviews compactly
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r['date'][:10]} | {r['stars']}★ | {r['text']}")

    prompt = f"""You are evaluating a restaurant to answer a specific question.
You MUST follow the rules exactly and provide your answer in the specified format.

## RESTAURANT
Name: {meta['name']}
Total Reviews: {len(reviews)}

## REVIEWS
{chr(10).join(reviews_text)}

## QUESTION
{scenario.question}

## {scenario.format_rules()}

IMPORTANT:
- Count carefully. Go through each review systematically.
- Show your work for each condition.
- Follow the output format exactly.
"""
    return prompt


def parse_response(response: str, scenario: Scenario) -> dict:
    """Parse the structured response from LLM."""

    parsed = {}

    # Parse each condition
    for cond in scenario.conditions:
        cond_id = cond.id
        pattern = rf"{cond_id}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            value_str = match.group(1).strip()

            if cond.output_type == "count":
                # Extract number
                nums = re.findall(r'\d+', value_str)
                parsed[cond_id] = int(nums[0]) if nums else None

            elif cond.output_type == "percentage":
                # Extract percentage
                nums = re.findall(r'(\d+\.?\d*)', value_str)
                parsed[cond_id] = float(nums[0]) if nums else None

            elif cond.output_type == "average":
                # Extract average (float)
                nums = re.findall(r'(\d+\.?\d*)', value_str)
                parsed[cond_id] = float(nums[0]) if nums else None

            elif cond.output_type == "boolean":
                # YES/NO
                value_upper = value_str.upper()
                if "YES" in value_upper:
                    parsed[cond_id] = True
                elif "NO" in value_upper:
                    parsed[cond_id] = False
                else:
                    parsed[cond_id] = None
        else:
            parsed[cond_id] = None

    # Parse verdict
    verdict_match = re.search(r"VERDICT:\s*(YES|NO)", response, re.IGNORECASE)
    if verdict_match:
        parsed["VERDICT"] = verdict_match.group(1).upper() == "YES"
    else:
        parsed["VERDICT"] = None

    return parsed


def evaluate_condition(
    cond_id: str,
    expected: any,
    predicted: any,
    output_type: str,
    tolerance: float = None
) -> ConditionResult:
    """Evaluate a single condition."""

    if predicted is None:
        return ConditionResult(
            condition_id=cond_id,
            expected=expected,
            predicted=predicted,
            correct=False,
            error="Could not parse value"
        )

    if output_type == "boolean":
        correct = (predicted == expected)
    elif output_type in ("count", "percentage", "average"):
        # Numeric comparison with tolerance
        if tolerance is None:
            if output_type == "count":
                tolerance = 2  # Allow ±2 for counts
            elif output_type == "percentage":
                tolerance = 5.0  # Allow ±5%
            elif output_type == "average":
                tolerance = 0.2  # Allow ±0.2 for averages

        correct = abs(predicted - expected) <= tolerance
    else:
        correct = (predicted == expected)

    return ConditionResult(
        condition_id=cond_id,
        expected=expected,
        predicted=predicted,
        correct=correct,
        error=None if correct else f"Expected {expected}, got {predicted}"
    )


def run_scenario(
    meta: dict,
    reviews: list[dict],
    scenario: Scenario,
    verbose: bool = True
) -> EvalResult:
    """Run evaluation for a single scenario."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.name}")
        print(f"Question: {scenario.question}")
        print(f"Reviews: {len(reviews)}")

    # Compute ground truth
    gt = scenario.compute_ground_truth(reviews)

    if verbose:
        print(f"\nGround Truth:")
        for cond_id, value in gt.conditions.items():
            print(f"  {cond_id}: {value}")
        print(f"  VERDICT: {'YES' if gt.verdict else 'NO'}")
        print(f"  Reasoning: {gt.reasoning}")

    # Build prompt and call LLM
    prompt = build_prompt(meta, reviews, scenario)

    if verbose:
        print(f"\nPrompt length: {len(prompt):,} chars")
        print("Calling LLM...")

    response = call_llm(prompt)

    if verbose:
        print(f"\n--- LLM Response ---")
        print(response[:1500] + "..." if len(response) > 1500 else response)

    # Parse response
    parsed = parse_response(response, scenario)

    if verbose:
        print(f"\n--- Parsed Values ---")
        for k, v in parsed.items():
            print(f"  {k}: {v}")

    # Evaluate each condition
    condition_results = []
    for cond in scenario.conditions:
        expected = gt.conditions[cond.id]
        predicted = parsed.get(cond.id)
        result = evaluate_condition(
            cond.id, expected, predicted, cond.output_type
        )
        condition_results.append(result)

    # Evaluate verdict
    verdict_predicted = parsed.get("VERDICT")
    verdict_correct = (verdict_predicted == gt.verdict) if verdict_predicted is not None else False

    # Full match: all conditions correct AND verdict correct
    all_conditions_correct = all(r.correct for r in condition_results)
    full_match = all_conditions_correct and verdict_correct

    # Print results
    if verbose:
        print(f"\n--- Evaluation ---")
        for r in condition_results:
            status = "✓" if r.correct else "✗"
            print(f"  {status} {r.condition_id}: expected={r.expected}, got={r.predicted}")

        verdict_status = "✓" if verdict_correct else "✗"
        print(f"  {verdict_status} VERDICT: expected={'YES' if gt.verdict else 'NO'}, got={'YES' if verdict_predicted else 'NO' if verdict_predicted is not None else 'N/A'}")

        if full_match:
            print(f"\n✓✓✓ FULL MATCH ✓✓✓")
        else:
            print(f"\n✗ NOT FULL MATCH")

    return EvalResult(
        scenario=scenario.name,
        conditions=condition_results,
        verdict_expected=gt.verdict,
        verdict_predicted=verdict_predicted,
        verdict_correct=verdict_correct,
        full_match=full_match,
        raw_response=response
    )


def run_all_scenarios(
    filename: str,
    max_reviews: int = None,
    verbose: bool = True
) -> list[EvalResult]:
    """Run all scenarios on a restaurant."""

    meta, reviews = load_reviews(filename, max_reviews)

    print(f"\n{'#'*60}")
    print(f"RESTAURANT: {meta['name']}")
    print(f"REVIEWS: {len(reviews)}")
    print(f"DATE RANGE: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}")
    print('#'*60)

    results = []
    for scenario in ALL_SCENARIOS:
        result = run_scenario(meta, reviews, scenario, verbose)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    total_conditions = 0
    correct_conditions = 0
    verdicts_correct = 0
    full_matches = 0

    for r in results:
        cond_correct = sum(1 for c in r.conditions if c.correct)
        cond_total = len(r.conditions)
        total_conditions += cond_total
        correct_conditions += cond_correct

        if r.verdict_correct:
            verdicts_correct += 1
        if r.full_match:
            full_matches += 1

        status = "FULL MATCH" if r.full_match else "PARTIAL" if r.verdict_correct else "FAIL"
        print(f"  {r.scenario}: {cond_correct}/{cond_total} conditions, verdict={'✓' if r.verdict_correct else '✗'} → {status}")

    print(f"\n  Condition accuracy: {correct_conditions}/{total_conditions} ({correct_conditions/total_conditions*100:.1f}%)")
    print(f"  Verdict accuracy: {verdicts_correct}/{len(results)} ({verdicts_correct/len(results)*100:.1f}%)")
    print(f"  Full match: {full_matches}/{len(results)} ({full_matches/len(results)*100:.1f}%)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl")
    parser.add_argument("--max-reviews", type=int, default=200,
                        help="Max reviews to include (default 200)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run single scenario by name")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.scenario:
        # Run single scenario
        from scenarios import get_scenario
        meta, reviews = load_reviews(args.file, args.max_reviews)
        scenario = get_scenario(args.scenario)
        run_scenario(meta, reviews, scenario, verbose=not args.quiet)
    else:
        # Run all scenarios
        run_all_scenarios(args.file, args.max_reviews, verbose=not args.quiet)


if __name__ == "__main__":
    main()
