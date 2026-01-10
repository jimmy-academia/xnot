#!/usr/bin/env python3
"""
Task v5: Hard multi-step evaluation with diverse difficulty types.

10 hard fields covering:
- Conditional counting (rating + keyword filters)
- Trend analysis (temporal comparisons)
- Weighted aggregation (date-based weights)
- Comparative analysis (subset comparisons)
- Pattern detection (categorization)
- Logical deduction (multi-step conditionals)
- Veto detection (needle in haystack)

Scoring: VERDICT_CORRECT × HARD_FIELD_ACCURACY
If verdict wrong → SCORE = 0
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class GroundTruth:
    """Ground truth for v5 task."""
    # Conditional counting
    critical_count: int
    service_count: int

    # Trend analysis
    trend_direction: str  # IMPROVING, DECLINING, STABLE

    # Weighted aggregation
    weighted_avg: float

    # Comparative analysis
    service_vs_food: str  # SERVICE_BETTER, FOOD_BETTER, EQUAL

    # Pattern detection
    top_complaint: str  # WAIT, SERVICE, FOOD, PRICE, NONE

    # Logical deduction
    polarization: str  # POLARIZED_NEGATIVE, POLARIZED_POSITIVE, POLARIZED_MIXED, BALANCED

    # Trend analysis (recent)
    recent_momentum: str  # POSITIVE, NEGATIVE, NEUTRAL

    # Veto detection
    veto_flag: str  # YES, NO

    # Final aggregation
    recommend_score: float
    verdict: str  # HIGHLY_RECOMMENDED, RECOMMENDED, ACCEPTABLE, RISKY, AVOID

    def to_dict(self) -> dict:
        return {
            'CRITICAL_COUNT': self.critical_count,
            'SERVICE_COUNT': self.service_count,
            'TREND_DIRECTION': self.trend_direction,
            'WEIGHTED_AVG': self.weighted_avg,
            'SERVICE_VS_FOOD': self.service_vs_food,
            'TOP_COMPLAINT': self.top_complaint,
            'POLARIZATION': self.polarization,
            'RECENT_MOMENTUM': self.recent_momentum,
            'VETO_FLAG': self.veto_flag,
            'RECOMMEND_SCORE': self.recommend_score,
            'VERDICT': self.verdict,
        }


def load_reviews(filename: str, max_reviews: int = None) -> tuple:
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


def matches_any(text: str, keywords: list, word_boundary: bool = False) -> bool:
    """Check if text contains any keyword."""
    text_lower = text.lower()
    for kw in keywords:
        if word_boundary:
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                return True
        else:
            if kw.lower() in text_lower:
                return True
    return False


def compute_ground_truth(reviews: list) -> GroundTruth:
    """Compute all 10 hard fields + verdict."""
    n = len(reviews)

    # =========================================================================
    # Field 1: CRITICAL_COUNT (Conditional Counting)
    # Count reviews with: 1-2 stars AND mentions critical keywords
    # =========================================================================
    critical_kw = ['sick', 'food poisoning', 'dirty', 'disgusting', 'worst', 'horrible']
    critical_count = sum(1 for r in reviews
                        if r['stars'] <= 2
                        and matches_any(r['text'], critical_kw))

    # =========================================================================
    # Field 2: SERVICE_COUNT (Conditional Counting)
    # Count reviews with: 1-3 stars AND mentions service issues
    # =========================================================================
    service_kw = ['slow', 'waited', 'ignored', 'rude', 'cold food', 'wrong order']
    service_count = sum(1 for r in reviews
                       if r['stars'] <= 3
                       and matches_any(r['text'], service_kw))

    # =========================================================================
    # Field 3: TREND_DIRECTION (Trend Analysis)
    # Compare average rating of OLDEST 1/3 vs NEWEST 1/3
    # =========================================================================
    third = max(1, n // 3)
    oldest = reviews[:third]
    newest = reviews[-third:]
    oldest_avg = sum(r['stars'] for r in oldest) / len(oldest) if oldest else 0
    newest_avg = sum(r['stars'] for r in newest) / len(newest) if newest else 0

    if newest_avg > oldest_avg + 0.3:
        trend_direction = "IMPROVING"
    elif newest_avg < oldest_avg - 0.3:
        trend_direction = "DECLINING"
    else:
        trend_direction = "STABLE"

    # =========================================================================
    # Field 4: WEIGHTED_AVG (Weighted Aggregation)
    # Date-based weights: 2021+ = 3, 2019-2020 = 2, older = 1
    # =========================================================================
    weighted_sum, weight_total = 0.0, 0
    for r in reviews:
        year = int(r['date'][:4])
        weight = 3 if year >= 2021 else 2 if year >= 2019 else 1
        weighted_sum += r['stars'] * weight
        weight_total += weight
    weighted_avg = round(weighted_sum / weight_total, 2) if weight_total > 0 else 0.0

    # =========================================================================
    # Field 5: SERVICE_VS_FOOD (Comparative Analysis)
    # Compare avg ratings of service-mentioning vs food-mentioning reviews
    # =========================================================================
    service_reviews = [r for r in reviews if matches_any(r['text'],
                       ['service', 'staff', 'waiter', 'server', 'waitress'])]
    food_reviews = [r for r in reviews if matches_any(r['text'],
                    ['food', 'dish', 'meal', 'taste', 'delicious'])]

    service_avg = (sum(r['stars'] for r in service_reviews) / len(service_reviews)
                   if service_reviews else 0)
    food_avg = (sum(r['stars'] for r in food_reviews) / len(food_reviews)
                if food_reviews else 0)

    if service_avg > food_avg + 0.2:
        service_vs_food = "SERVICE_BETTER"
    elif food_avg > service_avg + 0.2:
        service_vs_food = "FOOD_BETTER"
    else:
        service_vs_food = "EQUAL"

    # =========================================================================
    # Field 6: TOP_COMPLAINT (Pattern Detection)
    # Among 1-3 star reviews, which complaint category has most mentions?
    # =========================================================================
    low_reviews = [r for r in reviews if r['stars'] <= 3]
    complaint_cats = {
        'WAIT': ['wait', 'slow', 'long time', 'forever', 'waited'],
        'SERVICE': ['rude', 'ignored', 'attitude', 'unfriendly'],
        'FOOD': ['cold', 'bland', 'overcooked', 'undercooked', 'stale'],
        'PRICE': ['expensive', 'overpriced', 'not worth', 'rip off', 'pricey']
    }

    cat_counts = {cat: 0 for cat in complaint_cats}
    for r in low_reviews:
        text = r['text'].lower()
        for cat, keywords in complaint_cats.items():
            if any(kw in text for kw in keywords):
                cat_counts[cat] += 1

    if max(cat_counts.values()) == 0:
        top_complaint = "NONE"
    else:
        top_complaint = max(cat_counts, key=cat_counts.get)

    # =========================================================================
    # Field 7: POLARIZATION (Logical Deduction)
    # Analyze distribution of extreme ratings
    # =========================================================================
    five_star = sum(1 for r in reviews if r['stars'] == 5)
    one_star = sum(1 for r in reviews if r['stars'] == 1)
    three_star = sum(1 for r in reviews if r['stars'] == 3)
    polar_ratio = (five_star + one_star) / n if n > 0 else 0

    if polar_ratio > 0.5 and one_star > three_star:
        polarization = "POLARIZED_NEGATIVE"
    elif polar_ratio > 0.5 and five_star > three_star:
        polarization = "POLARIZED_POSITIVE"
    elif polar_ratio > 0.5:
        polarization = "POLARIZED_MIXED"
    else:
        polarization = "BALANCED"

    # =========================================================================
    # Field 8: RECENT_MOMENTUM (Trend Analysis)
    # Compare avg rating of LAST 20 reviews vs OVERALL avg
    # =========================================================================
    last_20 = reviews[-20:] if len(reviews) >= 20 else reviews
    overall_avg = sum(r['stars'] for r in reviews) / n if n > 0 else 0
    last20_avg = sum(r['stars'] for r in last_20) / len(last_20) if last_20 else 0

    if last20_avg > overall_avg + 0.3:
        recent_momentum = "POSITIVE"
    elif last20_avg < overall_avg - 0.3:
        recent_momentum = "NEGATIVE"
    else:
        recent_momentum = "NEUTRAL"

    # =========================================================================
    # Field 9: VETO_FLAG (Veto Detection / Needle in Haystack)
    # Any 1-star review mentioning severe issues?
    # =========================================================================
    veto_kw = ['health department', 'shut down', 'closed down', 'food poisoning',
               'hospitalized', 'emergency room']
    veto_found = any(r['stars'] == 1 and matches_any(r['text'], veto_kw)
                     for r in reviews)
    veto_flag = "YES" if veto_found else "NO"

    # =========================================================================
    # Field 10: RECOMMEND_SCORE (Final Aggregation)
    # Combines all previous fields into final score
    # =========================================================================
    trend_bonus = 10 if trend_direction == "IMPROVING" else -10 if trend_direction == "DECLINING" else 0
    momentum_bonus = 5 if recent_momentum == "POSITIVE" else -5 if recent_momentum == "NEGATIVE" else 0
    veto_penalty = 15 if veto_flag == "YES" else 0

    recommend_score = round(
        weighted_avg * 20
        - critical_count * 5
        - service_count * 2
        + trend_bonus
        + momentum_bonus
        - veto_penalty, 1)

    # =========================================================================
    # VERDICT
    # =========================================================================
    if recommend_score >= 80:
        verdict = "HIGHLY_RECOMMENDED"
    elif recommend_score >= 65:
        verdict = "RECOMMENDED"
    elif recommend_score >= 50:
        verdict = "ACCEPTABLE"
    elif recommend_score >= 35:
        verdict = "RISKY"
    else:
        verdict = "AVOID"

    return GroundTruth(
        critical_count=critical_count,
        service_count=service_count,
        trend_direction=trend_direction,
        weighted_avg=weighted_avg,
        service_vs_food=service_vs_food,
        top_complaint=top_complaint,
        polarization=polarization,
        recent_momentum=recent_momentum,
        veto_flag=veto_flag,
        recommend_score=recommend_score,
        verdict=verdict,
    )


def build_prompt(meta: dict, reviews: list) -> str:
    """Build evaluation prompt for LLM."""

    # Format reviews compactly
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r['date'][:10]} | {r['stars']}★ | {r['text']}")

    prompt = f"""You are evaluating a restaurant based on its reviews. Follow ALL rules exactly.

## RESTAURANT
Name: {meta['name']}
Total Reviews: {len(reviews)}
Date Range: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}

## REVIEWS
{chr(10).join(reviews_text)}

## EVALUATION RULES

You must compute these 10 fields IN ORDER:

### CRITICAL_COUNT (Conditional Counting)
Count reviews with: 1-2 stars AND mentions any of: "sick", "food poisoning", "dirty", "disgusting", "worst", "horrible"

### SERVICE_COUNT (Conditional Counting)
Count reviews with: 1-3 stars AND mentions any of: "slow", "waited", "ignored", "rude", "cold food", "wrong order"

### TREND_DIRECTION (Trend Analysis)
Split reviews into thirds by date order (oldest to newest).
Compare average rating of OLDEST 1/3 vs NEWEST 1/3.
- If newest avg > oldest avg + 0.3: "IMPROVING"
- If newest avg < oldest avg - 0.3: "DECLINING"
- Otherwise: "STABLE"

### WEIGHTED_AVG (Weighted Aggregation)
Calculate weighted average where:
- Reviews from 2021+: weight = 3
- Reviews from 2019-2020: weight = 2
- Reviews before 2019: weight = 1
Formula: Σ(rating × weight) / Σ(weight)
Round to 2 decimal places.

### SERVICE_VS_FOOD (Comparative Analysis)
- Find reviews mentioning: "service", "staff", "waiter", "server", "waitress" → calculate avg rating
- Find reviews mentioning: "food", "dish", "meal", "taste", "delicious" → calculate avg rating
- If service_avg > food_avg + 0.2: "SERVICE_BETTER"
- If food_avg > service_avg + 0.2: "FOOD_BETTER"
- Otherwise: "EQUAL"

### TOP_COMPLAINT (Pattern Detection)
Among 1-3 star reviews, which category has the MOST mentions?
Categories:
- WAIT: "wait", "slow", "long time", "forever", "waited"
- SERVICE: "rude", "ignored", "attitude", "unfriendly"
- FOOD: "cold", "bland", "overcooked", "undercooked", "stale"
- PRICE: "expensive", "overpriced", "not worth", "rip off", "pricey"
Output the category with most matches, or "NONE" if no matches.

### POLARIZATION (Logical Deduction)
Calculate: polar_ratio = (count of 5-star + count of 1-star) / total reviews
- If polar_ratio > 0.5 AND (1-star count) > (3-star count): "POLARIZED_NEGATIVE"
- If polar_ratio > 0.5 AND (5-star count) > (3-star count): "POLARIZED_POSITIVE"
- If polar_ratio > 0.5: "POLARIZED_MIXED"
- Otherwise: "BALANCED"

### RECENT_MOMENTUM (Trend Analysis)
Compare avg rating of LAST 20 reviews vs OVERALL avg:
- If last20_avg > overall_avg + 0.3: "POSITIVE"
- If last20_avg < overall_avg - 0.3: "NEGATIVE"
- Otherwise: "NEUTRAL"

### VETO_FLAG (Veto Detection)
Is there ANY 1-star review that mentions: "health department", "shut down", "closed down", "food poisoning", "hospitalized", "emergency room"?
Output: "YES" or "NO"

### RECOMMEND_SCORE (Final Aggregation)
Calculate:
SCORE = WEIGHTED_AVG × 20
      - (CRITICAL_COUNT × 5)
      - (SERVICE_COUNT × 2)
      + (10 if TREND_DIRECTION="IMPROVING" else -10 if TREND_DIRECTION="DECLINING" else 0)
      + (5 if RECENT_MOMENTUM="POSITIVE" else -5 if RECENT_MOMENTUM="NEGATIVE" else 0)
      - (15 if VETO_FLAG="YES" else 0)
Round to 1 decimal place.

### VERDICT
Based on RECOMMEND_SCORE:
- >= 80: "HIGHLY_RECOMMENDED"
- >= 65: "RECOMMENDED"
- >= 50: "ACCEPTABLE"
- >= 35: "RISKY"
- < 35: "AVOID"

## REQUIRED OUTPUT FORMAT

Show your work step by step, then provide your final answers.

IMPORTANT: You MUST end your response with a clearly marked FINAL ANSWERS block.
Use EXACTLY this format with the === markers:

===FINAL ANSWERS===
CRITICAL_COUNT: [number]
SERVICE_COUNT: [number]
TREND_DIRECTION: [IMPROVING/DECLINING/STABLE]
WEIGHTED_AVG: [number with 2 decimals]
SERVICE_VS_FOOD: [SERVICE_BETTER/FOOD_BETTER/EQUAL]
TOP_COMPLAINT: [WAIT/SERVICE/FOOD/PRICE/NONE]
POLARIZATION: [POLARIZED_NEGATIVE/POLARIZED_POSITIVE/POLARIZED_MIXED/BALANCED]
RECENT_MOMENTUM: [POSITIVE/NEGATIVE/NEUTRAL]
VETO_FLAG: [YES/NO]
RECOMMEND_SCORE: [number with 1 decimal]
VERDICT: [HIGHLY_RECOMMENDED/RECOMMENDED/ACCEPTABLE/RISKY/AVOID]
===END===
"""
    return prompt


def parse_response(response: str) -> dict:
    """Parse LLM response into field values.

    Looks for the ===FINAL ANSWERS=== block first; falls back to full response.
    """
    parsed = {}

    # Try to extract the FINAL ANSWERS block
    final_block = response
    markers = [
        (r'===\s*FINAL\s*ANSWERS\s*===', r'===\s*END\s*==='),
        (r'FINAL\s*ANSWERS:', r'$'),
        (r'###\s*FINAL', r'$'),
    ]

    for start_pattern, end_pattern in markers:
        start_match = re.search(start_pattern, response, re.IGNORECASE)
        if start_match:
            remaining = response[start_match.end():]
            end_match = re.search(end_pattern, remaining, re.IGNORECASE)
            if end_match:
                final_block = remaining[:end_match.start()]
            else:
                final_block = remaining
            break

    # Integer fields
    for field in ['CRITICAL_COUNT', 'SERVICE_COUNT']:
        pattern = rf"{field}:\s*(\d+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            parsed[field] = int(match.group(1))

    # Float fields
    for field in ['WEIGHTED_AVG']:
        pattern = rf"{field}:\s*([\d.]+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            parsed[field] = float(match.group(1))

    for field in ['RECOMMEND_SCORE']:
        pattern = rf"{field}:\s*(-?[\d.]+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            parsed[field] = float(match.group(1))

    # Categorical fields
    categorical = {
        'TREND_DIRECTION': ['IMPROVING', 'DECLINING', 'STABLE'],
        'SERVICE_VS_FOOD': ['SERVICE_BETTER', 'FOOD_BETTER', 'EQUAL'],
        'TOP_COMPLAINT': ['WAIT', 'SERVICE', 'FOOD', 'PRICE', 'NONE'],
        'POLARIZATION': ['POLARIZED_NEGATIVE', 'POLARIZED_POSITIVE', 'POLARIZED_MIXED', 'BALANCED'],
        'RECENT_MOMENTUM': ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
        'VETO_FLAG': ['YES', 'NO'],
        'VERDICT': ['HIGHLY_RECOMMENDED', 'RECOMMENDED', 'ACCEPTABLE', 'RISKY', 'AVOID'],
    }

    for field, options in categorical.items():
        pattern = rf"{field}:\s*(\S+)"
        match = re.search(pattern, final_block, re.IGNORECASE)
        if match:
            value = match.group(1).upper().strip('[]().,')
            # Find closest match
            for opt in options:
                if opt in value or value in opt:
                    parsed[field] = opt
                    break
            if field not in parsed:
                parsed[field] = value  # Keep original if no match

    return parsed


def validate_ground_truth(filename: str = "Acme_Oyster_House__ab50qdW.jsonl",
                          max_reviews: int = 100):
    """Validate ground truth computation with detailed breakdown."""

    meta, reviews = load_reviews(filename, max_reviews)
    gt = compute_ground_truth(reviews)
    gt_dict = gt.to_dict()

    print(f"{'='*60}")
    print(f"VALIDATION: Task v5")
    print(f"{'='*60}")
    print(f"Restaurant: {meta['name']}")
    print(f"Reviews: {len(reviews)}")
    print(f"Date range: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}")

    print(f"\n{'='*60}")
    print("GROUND TRUTH")
    print(f"{'='*60}")

    for field, value in gt_dict.items():
        print(f"  {field}: {value}")

    # Show detailed breakdown for verification
    print(f"\n{'='*60}")
    print("DETAILED BREAKDOWN")
    print(f"{'='*60}")

    # Critical count breakdown
    critical_kw = ['sick', 'food poisoning', 'dirty', 'disgusting', 'worst', 'horrible']
    critical_reviews = [(i, r) for i, r in enumerate(reviews, 1)
                        if r['stars'] <= 2 and matches_any(r['text'], critical_kw)]
    print(f"\nCRITICAL_COUNT ({gt.critical_count}):")
    for i, r in critical_reviews[:5]:
        matched = [kw for kw in critical_kw if kw in r['text'].lower()]
        print(f"  R{i}: {r['stars']}★, matched: {matched}")
    if len(critical_reviews) > 5:
        print(f"  ... and {len(critical_reviews) - 5} more")

    # Trend breakdown
    n = len(reviews)
    third = max(1, n // 3)
    oldest = reviews[:third]
    newest = reviews[-third:]
    oldest_avg = sum(r['stars'] for r in oldest) / len(oldest)
    newest_avg = sum(r['stars'] for r in newest) / len(newest)
    print(f"\nTREND_DIRECTION ({gt.trend_direction}):")
    print(f"  Oldest 1/3 (R1-R{third}): avg = {oldest_avg:.2f}")
    print(f"  Newest 1/3 (R{n-third+1}-R{n}): avg = {newest_avg:.2f}")
    print(f"  Difference: {newest_avg - oldest_avg:.2f}")

    # Polarization breakdown
    five_star = sum(1 for r in reviews if r['stars'] == 5)
    one_star = sum(1 for r in reviews if r['stars'] == 1)
    three_star = sum(1 for r in reviews if r['stars'] == 3)
    print(f"\nPOLARIZATION ({gt.polarization}):")
    print(f"  5-star: {five_star}, 1-star: {one_star}, 3-star: {three_star}")
    print(f"  Polar ratio: {(five_star + one_star) / n:.2f}")

    # Score breakdown
    print(f"\nRECOMMEND_SCORE calculation:")
    print(f"  WEIGHTED_AVG × 20 = {gt.weighted_avg} × 20 = {gt.weighted_avg * 20:.1f}")
    print(f"  - CRITICAL_COUNT × 5 = {gt.critical_count} × 5 = {gt.critical_count * 5}")
    print(f"  - SERVICE_COUNT × 2 = {gt.service_count} × 2 = {gt.service_count * 2}")
    trend_bonus = 10 if gt.trend_direction == "IMPROVING" else -10 if gt.trend_direction == "DECLINING" else 0
    print(f"  + trend_bonus = {trend_bonus}")
    momentum_bonus = 5 if gt.recent_momentum == "POSITIVE" else -5 if gt.recent_momentum == "NEGATIVE" else 0
    print(f"  + momentum_bonus = {momentum_bonus}")
    veto_penalty = 15 if gt.veto_flag == "YES" else 0
    print(f"  - veto_penalty = {veto_penalty}")
    print(f"  = {gt.recommend_score}")

    return gt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task v5 ground truth validation")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl")
    parser.add_argument("--max-reviews", type=int, default=100)
    args = parser.parse_args()

    validate_ground_truth(args.file, args.max_reviews)
