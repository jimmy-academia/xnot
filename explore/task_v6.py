#!/usr/bin/env python3
"""
Task v6: Semantic + numeric evaluation with proportional scoring.

15 fields across 5 tiers:
- Tier 1: Numeric basics (N_TOTAL, N_1STAR, N_5STAR, AVG_ALL)
- Tier 2: Temporal analysis (AVG_FIRST_THIRD, AVG_LAST_THIRD, TREND_DIFF)
- Tier 3: Semantic counts (N_WAIT_COMPLAINTS, N_SERVICE_COMPLAINTS, N_FOOD_COMPLAINTS, N_WOULD_NOT_RETURN)
- Tier 4: Conditional aggregation (AVG_COMPLAINERS, AVG_SATISFIED)
- Tier 5: Derived synthesis (COMPLAINT_RATIO, SATISFACTION_GAP)

Ground truth for semantic fields comes from pre-annotated labels.
Scoring is proportional - every field contributes.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"


@dataclass
class GroundTruth:
    """Ground truth for v6 task - all numeric."""
    # Tier 1: Numeric basics
    n_total: int
    n_1star: int
    n_5star: int
    avg_all: float

    # Tier 2: Temporal analysis
    avg_first_third: float
    avg_last_third: float
    trend_diff: float  # Depends on avg_first_third, avg_last_third

    # Tier 3: Semantic counts (from annotations)
    n_wait_complaints: int
    n_service_complaints: int
    n_food_complaints: int
    n_would_not_return: int

    # Tier 4: Conditional aggregation
    avg_complainers: float  # Depends on semantic counts
    avg_satisfied: float    # Depends on semantic counts

    # Tier 5: Derived synthesis
    complaint_ratio: float     # Depends on semantic counts, n_total
    satisfaction_gap: float    # Depends on avg_satisfied, avg_complainers

    def to_dict(self) -> dict:
        return {
            'N_TOTAL': self.n_total,
            'N_1STAR': self.n_1star,
            'N_5STAR': self.n_5star,
            'AVG_ALL': self.avg_all,
            'AVG_FIRST_THIRD': self.avg_first_third,
            'AVG_LAST_THIRD': self.avg_last_third,
            'TREND_DIFF': self.trend_diff,
            'N_WAIT_COMPLAINTS': self.n_wait_complaints,
            'N_SERVICE_COMPLAINTS': self.n_service_complaints,
            'N_FOOD_COMPLAINTS': self.n_food_complaints,
            'N_WOULD_NOT_RETURN': self.n_would_not_return,
            'AVG_COMPLAINERS': self.avg_complainers,
            'AVG_SATISFIED': self.avg_satisfied,
            'COMPLAINT_RATIO': self.complaint_ratio,
            'SATISFACTION_GAP': self.satisfaction_gap,
        }


# Field metadata: type, tolerance, dependencies
FIELD_META = {
    'N_TOTAL': {'type': 'int', 'tolerance': 0, 'depends': []},
    'N_1STAR': {'type': 'int', 'tolerance': 0, 'depends': []},
    'N_5STAR': {'type': 'int', 'tolerance': 0, 'depends': []},
    'AVG_ALL': {'type': 'float', 'tolerance': 0.05, 'depends': []},
    'AVG_FIRST_THIRD': {'type': 'float', 'tolerance': 0.05, 'depends': []},
    'AVG_LAST_THIRD': {'type': 'float', 'tolerance': 0.05, 'depends': []},
    'TREND_DIFF': {'type': 'float', 'tolerance': 0.1, 'depends': ['AVG_FIRST_THIRD', 'AVG_LAST_THIRD']},
    'N_WAIT_COMPLAINTS': {'type': 'int', 'tolerance': 1, 'depends': []},  # Allow ±1 for semantic
    'N_SERVICE_COMPLAINTS': {'type': 'int', 'tolerance': 1, 'depends': []},
    'N_FOOD_COMPLAINTS': {'type': 'int', 'tolerance': 1, 'depends': []},
    'N_WOULD_NOT_RETURN': {'type': 'int', 'tolerance': 1, 'depends': []},
    'AVG_COMPLAINERS': {'type': 'float', 'tolerance': 0.2, 'depends': ['N_WAIT_COMPLAINTS', 'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS']},
    'AVG_SATISFIED': {'type': 'float', 'tolerance': 0.2, 'depends': ['N_WAIT_COMPLAINTS', 'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS']},
    'COMPLAINT_RATIO': {'type': 'float', 'tolerance': 0.05, 'depends': ['N_WAIT_COMPLAINTS', 'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS', 'N_TOTAL']},
    'SATISFACTION_GAP': {'type': 'float', 'tolerance': 0.3, 'depends': ['AVG_SATISFIED', 'AVG_COMPLAINERS']},
}


def load_annotated_reviews(filename: str, max_reviews: int = None) -> list:
    """Load pre-annotated reviews."""
    filepath = ANNOTATED_DIR / filename
    reviews = []
    with open(filepath) as f:
        for line in f:
            reviews.append(json.loads(line))

    if max_reviews and max_reviews < len(reviews):
        reviews = reviews[:max_reviews]

    return reviews


def compute_ground_truth(reviews: list) -> GroundTruth:
    """Compute ground truth from annotated reviews."""
    n = len(reviews)

    # Tier 1: Numeric basics
    n_1star = sum(1 for r in reviews if r['stars'] == 1)
    n_5star = sum(1 for r in reviews if r['stars'] == 5)
    avg_all = round(sum(r['stars'] for r in reviews) / n, 2) if n > 0 else 0.0

    # Tier 2: Temporal analysis
    third = max(1, n // 3)
    first_third = reviews[:third]
    last_third = reviews[-third:]
    avg_first = round(sum(r['stars'] for r in first_third) / len(first_third), 2)
    avg_last = round(sum(r['stars'] for r in last_third) / len(last_third), 2)
    trend_diff = round(avg_last - avg_first, 2)

    # Tier 3: Semantic counts (from annotations)
    n_wait = sum(1 for r in reviews if r['labels'].get('wait_complaint', False))
    n_service = sum(1 for r in reviews if r['labels'].get('service_complaint', False))
    n_food = sum(1 for r in reviews if r['labels'].get('food_quality_complaint', False))
    n_not_return = sum(1 for r in reviews if r['labels'].get('would_not_return', False))

    # Tier 4: Conditional aggregation
    # Complainers = reviews with ANY complaint
    complainers = [r for r in reviews if (
        r['labels'].get('wait_complaint', False) or
        r['labels'].get('service_complaint', False) or
        r['labels'].get('food_quality_complaint', False)
    )]
    satisfied = [r for r in reviews if not (
        r['labels'].get('wait_complaint', False) or
        r['labels'].get('service_complaint', False) or
        r['labels'].get('food_quality_complaint', False)
    )]

    avg_complainers = round(sum(r['stars'] for r in complainers) / len(complainers), 2) if complainers else 0.0
    avg_satisfied = round(sum(r['stars'] for r in satisfied) / len(satisfied), 2) if satisfied else 0.0

    # Tier 5: Derived
    total_complaints = n_wait + n_service + n_food
    complaint_ratio = round(total_complaints / n, 3) if n > 0 else 0.0
    satisfaction_gap = round(avg_satisfied - avg_complainers, 2)

    return GroundTruth(
        n_total=n,
        n_1star=n_1star,
        n_5star=n_5star,
        avg_all=avg_all,
        avg_first_third=avg_first,
        avg_last_third=avg_last,
        trend_diff=trend_diff,
        n_wait_complaints=n_wait,
        n_service_complaints=n_service,
        n_food_complaints=n_food,
        n_would_not_return=n_not_return,
        avg_complainers=avg_complainers,
        avg_satisfied=avg_satisfied,
        complaint_ratio=complaint_ratio,
        satisfaction_gap=satisfaction_gap,
    )


def build_prompt(reviews: list) -> str:
    """Build evaluation prompt for LLM."""
    # Format reviews
    reviews_text = []
    for r in reviews:
        reviews_text.append(f"[R{r['review_id']}] {r['date'][:10]} | {r['stars']}★ | {r['text']}")

    n = len(reviews)
    third = max(1, n // 3)

    prompt = f'''Analyze these {n} restaurant reviews and compute the following metrics.

## REVIEWS
{chr(10).join(reviews_text)}

## METRICS TO COMPUTE

### Tier 1: Numeric Basics [independent]
- **N_TOTAL**: Total number of reviews
- **N_1STAR**: Count of 1-star reviews
- **N_5STAR**: Count of 5-star reviews
- **AVG_ALL**: Average rating across all reviews (2 decimal places)

### Tier 2: Temporal Analysis [independent]
Reviews are ordered from oldest to newest.
- **AVG_FIRST_THIRD**: Average rating of first {third} reviews (R1-R{third})
- **AVG_LAST_THIRD**: Average rating of last {third} reviews (R{n-third+1}-R{n})
- **TREND_DIFF**: AVG_LAST_THIRD - AVG_FIRST_THIRD [depends on above]

### Tier 3: Semantic Counts [requires judgment]
Count reviews where the reviewer:
- **N_WAIT_COMPLAINTS**: Expresses frustration about wait time, long lines, or slow service
- **N_SERVICE_COMPLAINTS**: Criticizes staff behavior, attitude, or service quality (not speed)
- **N_FOOD_COMPLAINTS**: Criticizes food taste, temperature, or preparation
- **N_WOULD_NOT_RETURN**: Explicitly says they won't return or don't recommend

### Tier 4: Conditional Aggregation [depends on Tier 3]
- **AVG_COMPLAINERS**: Average rating of reviews with ANY complaint (wait, service, or food)
- **AVG_SATISFIED**: Average rating of reviews with NO complaints

### Tier 5: Derived Metrics [depends on previous]
- **COMPLAINT_RATIO**: (N_WAIT + N_SERVICE + N_FOOD) / N_TOTAL (3 decimal places)
- **SATISFACTION_GAP**: AVG_SATISFIED - AVG_COMPLAINERS

## OUTPUT FORMAT

Compute step by step, then output your final answers.
End with EXACTLY this format:

===FINAL ANSWERS===
N_TOTAL: [integer]
N_1STAR: [integer]
N_5STAR: [integer]
AVG_ALL: [decimal, 2 places]
AVG_FIRST_THIRD: [decimal, 2 places]
AVG_LAST_THIRD: [decimal, 2 places]
TREND_DIFF: [decimal, 2 places]
N_WAIT_COMPLAINTS: [integer]
N_SERVICE_COMPLAINTS: [integer]
N_FOOD_COMPLAINTS: [integer]
N_WOULD_NOT_RETURN: [integer]
AVG_COMPLAINERS: [decimal, 2 places]
AVG_SATISFIED: [decimal, 2 places]
COMPLAINT_RATIO: [decimal, 3 places]
SATISFACTION_GAP: [decimal, 2 places]
===END===
'''
    return prompt


def parse_response(response: str) -> dict:
    """Parse LLM response into field values."""
    import re

    parsed = {}

    # Try to extract the FINAL ANSWERS block
    final_block = response
    start_match = re.search(r'===\s*FINAL\s*ANSWERS\s*===', response, re.IGNORECASE)
    if start_match:
        remaining = response[start_match.end():]
        end_match = re.search(r'===\s*END\s*===', remaining, re.IGNORECASE)
        if end_match:
            final_block = remaining[:end_match.start()]
        else:
            final_block = remaining

    # Integer fields
    int_fields = ['N_TOTAL', 'N_1STAR', 'N_5STAR', 'N_WAIT_COMPLAINTS',
                  'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS', 'N_WOULD_NOT_RETURN']
    for field in int_fields:
        pattern = rf"{field}:\s*(\d+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            parsed[field] = int(match.group(1))

    # Float fields
    float_fields = ['AVG_ALL', 'AVG_FIRST_THIRD', 'AVG_LAST_THIRD', 'TREND_DIFF',
                    'AVG_COMPLAINERS', 'AVG_SATISFIED', 'COMPLAINT_RATIO', 'SATISFACTION_GAP']
    for field in float_fields:
        pattern = rf"{field}:\s*(-?[\d.]+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            parsed[field] = float(match.group(1))

    return parsed


def validate_ground_truth(filename: str, max_reviews: int = None):
    """Validate ground truth computation with detailed breakdown."""
    reviews = load_annotated_reviews(filename, max_reviews)
    gt = compute_ground_truth(reviews)
    gt_dict = gt.to_dict()

    print(f"{'='*60}")
    print(f"VALIDATION: Task v6")
    print(f"{'='*60}")
    print(f"Reviews: {len(reviews)}")

    print(f"\n{'='*60}")
    print("GROUND TRUTH")
    print(f"{'='*60}")

    # Group by tier
    tiers = {
        'Tier 1 (Numeric)': ['N_TOTAL', 'N_1STAR', 'N_5STAR', 'AVG_ALL'],
        'Tier 2 (Temporal)': ['AVG_FIRST_THIRD', 'AVG_LAST_THIRD', 'TREND_DIFF'],
        'Tier 3 (Semantic)': ['N_WAIT_COMPLAINTS', 'N_SERVICE_COMPLAINTS', 'N_FOOD_COMPLAINTS', 'N_WOULD_NOT_RETURN'],
        'Tier 4 (Conditional)': ['AVG_COMPLAINERS', 'AVG_SATISFIED'],
        'Tier 5 (Derived)': ['COMPLAINT_RATIO', 'SATISFACTION_GAP'],
    }

    for tier_name, fields in tiers.items():
        print(f"\n{tier_name}:")
        for field in fields:
            deps = FIELD_META[field]['depends']
            dep_str = f" [depends: {', '.join(deps)}]" if deps else ""
            print(f"  {field}: {gt_dict[field]}{dep_str}")

    # Show some example reviews with complaints
    print(f"\n{'='*60}")
    print("SAMPLE COMPLAINT REVIEWS")
    print(f"{'='*60}")

    for label_key, label_name in [
        ('wait_complaint', 'Wait'),
        ('service_complaint', 'Service'),
        ('food_quality_complaint', 'Food'),
    ]:
        matches = [r for r in reviews if r['labels'].get(label_key, False)]
        print(f"\n{label_name} complaints ({len(matches)}):")
        for r in matches[:2]:
            text = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
            print(f"  R{r['review_id']} ({r['stars']}★): {text}")

    return gt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task v6 ground truth validation")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW_labeled.jsonl")
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    validate_ground_truth(args.file, args.max_reviews)
