#!/usr/bin/env python3
"""
Task v4: Multi-step restaurant evaluation for risk-averse customers.

This module provides:
1. Ground truth computation
2. Prompt generation
3. Response parsing
4. Validation utilities
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# CONFIGURATION
# =============================================================================

CRITICAL_KEYWORDS = [
    'sick', 'food poisoning', 'rude', 'dirty',
    'disgusting', 'worst', 'horrible', 'never again'
]

SERVICE_KEYWORDS = [
    'slow', 'waited', 'wait', 'cold food',
    'wrong order', 'forgot', 'ignored'
]

# Penalties
CRITICAL_MULTIPLIER = 5
SERVICE_MULTIPLIER = 2
TREND_MULTIPLIER = 0.5
CONSISTENCY_THRESHOLD = 0.3
CONSISTENCY_PENALTY_VALUE = 10

# Verdict thresholds
RECOMMENDED_THRESHOLD = 75
ACCEPTABLE_THRESHOLD = 60
RISKY_THRESHOLD = 45


# =============================================================================
# DATA LOADING
# =============================================================================

def load_reviews(filename: str, max_reviews: int = None) -> tuple[dict, list[dict]]:
    """Load restaurant data from JSONL file."""
    filepath = DATA_DIR / filename
    with open(filepath) as f:
        meta = json.loads(f.readline())
        all_reviews = [json.loads(line) for line in f]

    if max_reviews and max_reviews < len(all_reviews):
        # Take most recent reviews
        reviews = all_reviews[-max_reviews:]
    else:
        reviews = all_reviews

    return meta, reviews


# =============================================================================
# GROUND TRUTH COMPUTATION
# =============================================================================

@dataclass
class GroundTruth:
    """All computed values for the task."""
    # Step 1: Baseline
    N: int
    AVG: float
    BASE_SCORE: float

    # Step 2: Critical issues
    CRITICAL_COUNT: int

    # Step 3: Service issues
    SERVICE_COUNT: int

    # Step 4: Penalties
    CRITICAL_PENALTY: int
    SERVICE_PENALTY: int
    TOTAL_PENALTY: int

    # Step 5: Recent trend
    RECENT_POS: int
    RECENT_NEG: int
    TREND_SCORE: float

    # Step 6: Consistency
    FIVE_STAR: int
    ONE_STAR: int
    VARIANCE_RATIO: float
    CONSISTENCY_PENALTY: int

    # Step 7-8: Final
    FINAL_SCORE: float
    VERDICT: str

    def to_dict(self) -> dict:
        return {
            'N': self.N,
            'AVG': self.AVG,
            'BASE_SCORE': self.BASE_SCORE,
            'CRITICAL_COUNT': self.CRITICAL_COUNT,
            'SERVICE_COUNT': self.SERVICE_COUNT,
            'CRITICAL_PENALTY': self.CRITICAL_PENALTY,
            'SERVICE_PENALTY': self.SERVICE_PENALTY,
            'TOTAL_PENALTY': self.TOTAL_PENALTY,
            'RECENT_POS': self.RECENT_POS,
            'RECENT_NEG': self.RECENT_NEG,
            'TREND_SCORE': self.TREND_SCORE,
            'FIVE_STAR': self.FIVE_STAR,
            'ONE_STAR': self.ONE_STAR,
            'VARIANCE_RATIO': self.VARIANCE_RATIO,
            'CONSISTENCY_PENALTY': self.CONSISTENCY_PENALTY,
            'FINAL_SCORE': self.FINAL_SCORE,
            'VERDICT': self.VERDICT,
        }


def compute_ground_truth(reviews: list[dict]) -> GroundTruth:
    """Compute all intermediate values and final verdict."""

    # Step 1: Baseline metrics
    n = len(reviews)
    if n == 0:
        raise ValueError("No reviews provided")

    avg = round(sum(r['stars'] for r in reviews) / n, 2)
    base_score = round(avg * 20, 2)

    # Step 2: Critical issues (1-2 stars + keywords)
    critical_count = 0
    for r in reviews:
        if r['stars'] <= 2:
            text_lower = r['text'].lower()
            if any(kw in text_lower for kw in CRITICAL_KEYWORDS):
                critical_count += 1

    # Step 3: Service issues (1-3 stars + keywords)
    service_count = 0
    for r in reviews:
        if r['stars'] <= 3:
            text_lower = r['text'].lower()
            if any(kw in text_lower for kw in SERVICE_KEYWORDS):
                service_count += 1

    # Step 4: Calculate penalties
    critical_penalty = critical_count * CRITICAL_MULTIPLIER
    service_penalty = service_count * SERVICE_MULTIPLIER
    total_penalty = critical_penalty + service_penalty

    # Step 5: Recent trend (2021+)
    recent = [r for r in reviews if r['date'][:4] >= '2021']
    recent_pos = sum(1 for r in recent if r['stars'] >= 4)
    recent_neg = sum(1 for r in recent if r['stars'] <= 2)
    trend_score = round((recent_pos - recent_neg) * TREND_MULTIPLIER, 2)

    # Step 6: Consistency check
    five_star = sum(1 for r in reviews if r['stars'] == 5)
    one_star = sum(1 for r in reviews if r['stars'] == 1)
    variance_ratio = round(one_star / max(five_star, 1), 2)
    consistency_penalty = CONSISTENCY_PENALTY_VALUE if variance_ratio > CONSISTENCY_THRESHOLD else 0

    # Step 7: Final score
    final_score = round(
        base_score - total_penalty + trend_score - consistency_penalty,
        2
    )

    # Step 8: Verdict
    if final_score >= RECOMMENDED_THRESHOLD:
        verdict = 'RECOMMENDED'
    elif final_score >= ACCEPTABLE_THRESHOLD:
        verdict = 'ACCEPTABLE'
    elif final_score >= RISKY_THRESHOLD:
        verdict = 'RISKY'
    else:
        verdict = 'AVOID'

    return GroundTruth(
        N=n,
        AVG=avg,
        BASE_SCORE=base_score,
        CRITICAL_COUNT=critical_count,
        SERVICE_COUNT=service_count,
        CRITICAL_PENALTY=critical_penalty,
        SERVICE_PENALTY=service_penalty,
        TOTAL_PENALTY=total_penalty,
        RECENT_POS=recent_pos,
        RECENT_NEG=recent_neg,
        TREND_SCORE=trend_score,
        FIVE_STAR=five_star,
        ONE_STAR=one_star,
        VARIANCE_RATIO=variance_ratio,
        CONSISTENCY_PENALTY=consistency_penalty,
        FINAL_SCORE=final_score,
        VERDICT=verdict,
    )


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def build_prompt(meta: dict, reviews: list[dict]) -> str:
    """Build the evaluation prompt with all reviews."""

    # Format reviews
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r['date'][:10]} | {r['stars']}★ | {r['text']}")

    prompt = f"""You are evaluating this restaurant for a RISK-AVERSE customer who prioritizes
consistency and safety over exceptional experiences.

Apply the following rules PRECISELY. Show all intermediate calculations.

## RESTAURANT
Name: {meta['name']}
Overall Rating: {meta.get('stars', 'N/A')}

## REVIEWS ({len(reviews)} total)
{chr(10).join(reviews_text)}

═══════════════════════════════════════════════════════════════════════════════
STEP 1: BASELINE METRICS
═══════════════════════════════════════════════════════════════════════════════
1a. Count total number of reviews → N
1b. Calculate average star rating (to 2 decimal places) → AVG
1c. Calculate BASE_SCORE = AVG × 20

═══════════════════════════════════════════════════════════════════════════════
STEP 2: CRITICAL ISSUES (severe problems that heavily penalize)
═══════════════════════════════════════════════════════════════════════════════
Find reviews that have BOTH:
  - Rating of 1-2 stars AND
  - Mention ANY of: "sick", "food poisoning", "rude", "dirty", "disgusting", "worst", "horrible", "never again"

2a. Count these reviews → CRITICAL_COUNT

═══════════════════════════════════════════════════════════════════════════════
STEP 3: SERVICE ISSUES (moderate problems)
═══════════════════════════════════════════════════════════════════════════════
Find reviews that have BOTH:
  - Rating of 1-3 stars AND
  - Mention ANY of: "slow", "waited", "wait", "cold food", "wrong order", "forgot", "ignored"

3a. Count these reviews → SERVICE_COUNT

═══════════════════════════════════════════════════════════════════════════════
STEP 4: CALCULATE PENALTIES
═══════════════════════════════════════════════════════════════════════════════
4a. CRITICAL_PENALTY = CRITICAL_COUNT × 5
4b. SERVICE_PENALTY = SERVICE_COUNT × 2
4c. TOTAL_PENALTY = CRITICAL_PENALTY + SERVICE_PENALTY

═══════════════════════════════════════════════════════════════════════════════
STEP 5: RECENT TREND ANALYSIS (reviews from 2021 or later)
═══════════════════════════════════════════════════════════════════════════════
5a. Count reviews from 2021+ with 4-5 stars → RECENT_POS
5b. Count reviews from 2021+ with 1-2 stars → RECENT_NEG
5c. TREND_SCORE = (RECENT_POS - RECENT_NEG) × 0.5

═══════════════════════════════════════════════════════════════════════════════
STEP 6: CONSISTENCY CHECK
═══════════════════════════════════════════════════════════════════════════════
6a. Count 5-star reviews → FIVE_STAR
6b. Count 1-star reviews → ONE_STAR
6c. VARIANCE_RATIO = ONE_STAR / max(FIVE_STAR, 1)
6d. If VARIANCE_RATIO > 0.3: CONSISTENCY_PENALTY = 10, else CONSISTENCY_PENALTY = 0

═══════════════════════════════════════════════════════════════════════════════
STEP 7: FINAL CALCULATION
═══════════════════════════════════════════════════════════════════════════════
FINAL_SCORE = BASE_SCORE - TOTAL_PENALTY + TREND_SCORE - CONSISTENCY_PENALTY

═══════════════════════════════════════════════════════════════════════════════
STEP 8: VERDICT
═══════════════════════════════════════════════════════════════════════════════
- If FINAL_SCORE >= 75: RECOMMENDED
- If FINAL_SCORE >= 60: ACCEPTABLE
- If FINAL_SCORE >= 45: RISKY
- Otherwise: AVOID

═══════════════════════════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT (fill in all values):
═══════════════════════════════════════════════════════════════════════════════
N: ___
AVG: ___
BASE_SCORE: ___
CRITICAL_COUNT: ___
SERVICE_COUNT: ___
CRITICAL_PENALTY: ___
SERVICE_PENALTY: ___
TOTAL_PENALTY: ___
RECENT_POS: ___
RECENT_NEG: ___
TREND_SCORE: ___
FIVE_STAR: ___
ONE_STAR: ___
VARIANCE_RATIO: ___
CONSISTENCY_PENALTY: ___
FINAL_SCORE: ___
VERDICT: ___
"""
    return prompt


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response: str) -> dict:
    """Parse LLM response to extract all values."""

    fields = [
        'N', 'AVG', 'BASE_SCORE',
        'CRITICAL_COUNT', 'SERVICE_COUNT',
        'CRITICAL_PENALTY', 'SERVICE_PENALTY', 'TOTAL_PENALTY',
        'RECENT_POS', 'RECENT_NEG', 'TREND_SCORE',
        'FIVE_STAR', 'ONE_STAR', 'VARIANCE_RATIO', 'CONSISTENCY_PENALTY',
        'FINAL_SCORE', 'VERDICT'
    ]

    parsed = {}

    for field in fields:
        # Match "FIELD: value" or "FIELD = value"
        pattern = rf"{field}[:\s=]+(.+?)(?:\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            value_str = match.group(1).strip()

            if field == 'VERDICT':
                # Extract verdict keyword
                for v in ['RECOMMENDED', 'ACCEPTABLE', 'RISKY', 'AVOID']:
                    if v in value_str.upper():
                        parsed[field] = v
                        break
                else:
                    parsed[field] = None
            else:
                # Extract number
                nums = re.findall(r'-?[\d.]+', value_str)
                if nums:
                    val = float(nums[0])
                    # Convert to int for count fields
                    if field in ['N', 'CRITICAL_COUNT', 'SERVICE_COUNT',
                                 'CRITICAL_PENALTY', 'SERVICE_PENALTY', 'TOTAL_PENALTY',
                                 'RECENT_POS', 'RECENT_NEG', 'FIVE_STAR', 'ONE_STAR',
                                 'CONSISTENCY_PENALTY']:
                        parsed[field] = int(val)
                    else:
                        parsed[field] = val
                else:
                    parsed[field] = None
        else:
            parsed[field] = None

    return parsed


# =============================================================================
# VALIDATION
# =============================================================================

def validate_ground_truth(filename: str = "Acme_Oyster_House__ab50qdW.jsonl",
                         max_reviews: int = 100,
                         verbose: bool = True) -> GroundTruth:
    """Validate ground truth computation with detailed output."""

    meta, reviews = load_reviews(filename, max_reviews)
    gt = compute_ground_truth(reviews)

    if verbose:
        print(f"{'='*60}")
        print(f"GROUND TRUTH VALIDATION")
        print(f"{'='*60}")
        print(f"Restaurant: {meta['name']}")
        print(f"Reviews loaded: {len(reviews)}")
        print(f"Date range: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}")

        print(f"\n{'='*60}")
        print("COMPUTED VALUES")
        print(f"{'='*60}")

        print("\n--- Step 1: Baseline ---")
        print(f"  N = {gt.N}")
        print(f"  AVG = {gt.AVG}")
        print(f"  BASE_SCORE = {gt.AVG} × 20 = {gt.BASE_SCORE}")

        print("\n--- Step 2: Critical Issues ---")
        print(f"  CRITICAL_COUNT = {gt.CRITICAL_COUNT}")
        # Show examples
        critical_examples = []
        for r in reviews:
            if r['stars'] <= 2:
                text_lower = r['text'].lower()
                for kw in CRITICAL_KEYWORDS:
                    if kw in text_lower:
                        critical_examples.append((r['date'][:10], r['stars'], kw, r['text'][:80]))
                        break
        if critical_examples:
            print(f"  Examples ({len(critical_examples)} found):")
            for date, stars, kw, text in critical_examples[:3]:
                print(f"    [{date}] {stars}★ '{kw}': {text}...")

        print("\n--- Step 3: Service Issues ---")
        print(f"  SERVICE_COUNT = {gt.SERVICE_COUNT}")

        print("\n--- Step 4: Penalties ---")
        print(f"  CRITICAL_PENALTY = {gt.CRITICAL_COUNT} × 5 = {gt.CRITICAL_PENALTY}")
        print(f"  SERVICE_PENALTY = {gt.SERVICE_COUNT} × 2 = {gt.SERVICE_PENALTY}")
        print(f"  TOTAL_PENALTY = {gt.TOTAL_PENALTY}")

        print("\n--- Step 5: Recent Trend ---")
        recent = [r for r in reviews if r['date'][:4] >= '2021']
        print(f"  Reviews from 2021+: {len(recent)}")
        print(f"  RECENT_POS (4-5★) = {gt.RECENT_POS}")
        print(f"  RECENT_NEG (1-2★) = {gt.RECENT_NEG}")
        print(f"  TREND_SCORE = ({gt.RECENT_POS} - {gt.RECENT_NEG}) × 0.5 = {gt.TREND_SCORE}")

        print("\n--- Step 6: Consistency ---")
        print(f"  FIVE_STAR = {gt.FIVE_STAR}")
        print(f"  ONE_STAR = {gt.ONE_STAR}")
        print(f"  VARIANCE_RATIO = {gt.ONE_STAR} / max({gt.FIVE_STAR}, 1) = {gt.VARIANCE_RATIO}")
        print(f"  CONSISTENCY_PENALTY = {gt.CONSISTENCY_PENALTY} ({'> 0.3' if gt.VARIANCE_RATIO > 0.3 else '<= 0.3'})")

        print("\n--- Step 7-8: Final ---")
        print(f"  FINAL_SCORE = {gt.BASE_SCORE} - {gt.TOTAL_PENALTY} + {gt.TREND_SCORE} - {gt.CONSISTENCY_PENALTY}")
        print(f"              = {gt.FINAL_SCORE}")
        print(f"  VERDICT = {gt.VERDICT}")

        # Verification assertions
        print(f"\n{'='*60}")
        print("VERIFICATION CHECKS")
        print(f"{'='*60}")

        checks_passed = 0
        checks_total = 0

        # Check N
        checks_total += 1
        if gt.N == len(reviews):
            print(f"  ✓ N matches review count")
            checks_passed += 1
        else:
            print(f"  ✗ N mismatch: {gt.N} vs {len(reviews)}")

        # Check AVG
        checks_total += 1
        manual_avg = sum(r['stars'] for r in reviews) / len(reviews)
        if abs(gt.AVG - manual_avg) < 0.01:
            print(f"  ✓ AVG calculation correct")
            checks_passed += 1
        else:
            print(f"  ✗ AVG mismatch: {gt.AVG} vs {manual_avg}")

        # Check BASE_SCORE
        checks_total += 1
        expected_base = gt.AVG * 20
        if abs(gt.BASE_SCORE - expected_base) < 0.01:
            print(f"  ✓ BASE_SCORE calculation correct")
            checks_passed += 1
        else:
            print(f"  ✗ BASE_SCORE mismatch: {gt.BASE_SCORE} vs {expected_base}")

        # Check penalty math
        checks_total += 1
        expected_total = gt.CRITICAL_PENALTY + gt.SERVICE_PENALTY
        if gt.TOTAL_PENALTY == expected_total:
            print(f"  ✓ TOTAL_PENALTY calculation correct")
            checks_passed += 1
        else:
            print(f"  ✗ TOTAL_PENALTY mismatch: {gt.TOTAL_PENALTY} vs {expected_total}")

        # Check final score formula
        checks_total += 1
        expected_final = gt.BASE_SCORE - gt.TOTAL_PENALTY + gt.TREND_SCORE - gt.CONSISTENCY_PENALTY
        if abs(gt.FINAL_SCORE - expected_final) < 0.01:
            print(f"  ✓ FINAL_SCORE calculation correct")
            checks_passed += 1
        else:
            print(f"  ✗ FINAL_SCORE mismatch: {gt.FINAL_SCORE} vs {expected_final}")

        print(f"\n  Checks passed: {checks_passed}/{checks_total}")

    return gt


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task v4: Ground truth computation and validation")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl",
                       help="Data file to use")
    parser.add_argument("--max-reviews", type=int, default=100,
                       help="Maximum reviews to load (default: 100)")
    parser.add_argument("--show-prompt", action="store_true",
                       help="Also show the generated prompt")
    args = parser.parse_args()

    gt = validate_ground_truth(args.file, args.max_reviews)

    if args.show_prompt:
        meta, reviews = load_reviews(args.file, args.max_reviews)
        prompt = build_prompt(meta, reviews)
        print(f"\n{'='*60}")
        print("GENERATED PROMPT (first 2000 chars)")
        print(f"{'='*60}")
        print(prompt[:2000])
        print(f"\n... ({len(prompt)} total characters)")
