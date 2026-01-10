#!/usr/bin/env python3
"""
L2 Evaluation Framework

Tests LLM ability to compute derived metrics that require multi-step reasoning.

Tasks:
- A: Recent vs Historical comparison
- B: Credibility Weighting (using useful votes, date)
- C: Rating-Text Alignment (star vs sentiment, plus reviewer patterns)
- D: Cross-Aspect Correlation (aspect co-occurrence)
- F: Expectation Calibration (relative to restaurant type)
- Combined: All tasks together

Usage:
    python explore/l2_eval.py --task A --restaurant Acme
    python explore/l2_eval.py --task all --restaurant all
"""

import json
import re
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, configure

DATA_DIR = Path(__file__).parent / "data"
DEBUG_DIR = Path(__file__).parent / "debug" / "l2"

# Available restaurants
RESTAURANTS = [
    "Acme_Oyster_House__ab50qdW.jsonl",
    "Cochon_6a4gLLFS.jsonl",
    "Commander_s_Palace__C7QiQQc.jsonl",
    "Oceana_Grill_ac1AeYqs.jsonl",
    "Royal_House_VQcCL9Pi.jsonl",
]


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class Review:
    """Parsed review with all relevant fields."""
    review_id: str
    date: str
    stars: float
    text: str
    useful: int
    funny: int
    cool: int

    @property
    def year(self) -> int:
        return int(self.date[:4])

    @property
    def datetime(self) -> datetime:
        return datetime.strptime(self.date[:10], "%Y-%m-%d")


@dataclass
class Restaurant:
    """Restaurant metadata."""
    name: str
    stars: float
    review_count: int
    categories: str
    city: str
    attributes: Dict[str, Any]

    @property
    def price_tier(self) -> int:
        return self.attributes.get('RestaurantsPriceRange2', 2)

    @property
    def noise_level(self) -> str:
        return self.attributes.get('NoiseLevel', 'average')

    @property
    def is_good_for_kids(self) -> bool:
        return self.attributes.get('GoodForKids', False)


def load_restaurant_data(filename: str, max_reviews: int = 100) -> tuple:
    """Load restaurant metadata and reviews."""
    filepath = DATA_DIR / filename

    with open(filepath) as f:
        meta_raw = json.loads(f.readline())
        restaurant = Restaurant(
            name=meta_raw['name'],
            stars=meta_raw['stars'],
            review_count=meta_raw['review_count'],
            categories=meta_raw['categories'],
            city=meta_raw['city'],
            attributes=meta_raw.get('attributes', {})
        )

        all_reviews = []
        for line in f:
            r = json.loads(line)
            all_reviews.append(Review(
                review_id=r['review_id'],
                date=r['date'],
                stars=r['stars'],
                text=r['text'],
                useful=r.get('useful', 0),
                funny=r.get('funny', 0),
                cool=r.get('cool', 0)
            ))

    # Take most recent reviews
    if max_reviews and max_reviews < len(all_reviews):
        reviews = all_reviews[-max_reviews:]
    else:
        reviews = all_reviews

    return restaurant, reviews


# =============================================================================
# TASK A: RECENT VS HISTORICAL
# =============================================================================

@dataclass
class TaskAGroundTruth:
    """Ground truth for Task A: Recent vs Historical."""
    n_total: int
    n_recent: int  # Last 1/3
    n_historical: int  # First 2/3
    avg_recent: float
    avg_historical: float
    delta: float  # recent - historical
    direction: str  # IMPROVING, DECLINING, STABLE
    recent_5star_ratio: float
    historical_5star_ratio: float
    recent_1star_ratio: float
    historical_1star_ratio: float


def compute_task_a_ground_truth(reviews: List[Review]) -> TaskAGroundTruth:
    """Compute ground truth for Task A."""
    n = len(reviews)
    split_idx = (2 * n) // 3  # First 2/3 = historical, last 1/3 = recent

    historical = reviews[:split_idx]
    recent = reviews[split_idx:]

    avg_hist = sum(r.stars for r in historical) / len(historical) if historical else 0
    avg_recent = sum(r.stars for r in recent) / len(recent) if recent else 0
    delta = round(avg_recent - avg_hist, 2)

    if delta > 0.3:
        direction = "IMPROVING"
    elif delta < -0.3:
        direction = "DECLINING"
    else:
        direction = "STABLE"

    recent_5star = sum(1 for r in recent if r.stars == 5) / len(recent) if recent else 0
    hist_5star = sum(1 for r in historical if r.stars == 5) / len(historical) if historical else 0
    recent_1star = sum(1 for r in recent if r.stars == 1) / len(recent) if recent else 0
    hist_1star = sum(1 for r in historical if r.stars == 1) / len(historical) if historical else 0

    return TaskAGroundTruth(
        n_total=n,
        n_recent=len(recent),
        n_historical=len(historical),
        avg_recent=round(avg_recent, 2),
        avg_historical=round(avg_hist, 2),
        delta=delta,
        direction=direction,
        recent_5star_ratio=round(recent_5star, 3),
        historical_5star_ratio=round(hist_5star, 3),
        recent_1star_ratio=round(recent_1star, 3),
        historical_1star_ratio=round(hist_1star, 3),
    )


def build_task_a_prompt(restaurant: Restaurant, reviews: List[Review]) -> str:
    """Build prompt for Task A."""
    n = len(reviews)
    split_idx = (2 * n) // 3

    reviews_text = []
    for i, r in enumerate(reviews, 1):
        period = "H" if i <= split_idx else "R"  # H=Historical, R=Recent
        reviews_text.append(f"[R{i}] {r.date[:10]} | {r.stars}★ | {period}")

    return f'''Analyze temporal patterns. H=Historical (R1-R{split_idx}), R=Recent (R{split_idx+1}-R{n}).

Reviews: {chr(10).join(reviews_text)}

Compute:
- N_TOTAL, N_RECENT, N_HISTORICAL
- AVG_RECENT, AVG_HISTORICAL (2 decimals)
- DELTA = AVG_RECENT - AVG_HISTORICAL
- DIRECTION: DELTA>0.3="IMPROVING", DELTA<-0.3="DECLINING", else "STABLE"
- RECENT_5STAR_RATIO, HISTORICAL_5STAR_RATIO (3 decimals)
- RECENT_1STAR_RATIO, HISTORICAL_1STAR_RATIO (3 decimals)

===FINAL ANSWERS===
N_TOTAL:
N_RECENT:
N_HISTORICAL:
AVG_RECENT:
AVG_HISTORICAL:
DELTA:
DIRECTION:
RECENT_5STAR_RATIO:
HISTORICAL_5STAR_RATIO:
RECENT_1STAR_RATIO:
HISTORICAL_1STAR_RATIO:
===END===
'''


# =============================================================================
# TASK B: CREDIBILITY WEIGHTING
# =============================================================================

@dataclass
class TaskBGroundTruth:
    """Ground truth for Task B: Credibility Weighting."""
    n_total: int
    n_high_useful: int  # Reviews with useful >= 3
    n_low_useful: int   # Reviews with useful == 0
    avg_all: float
    avg_high_useful: float
    avg_low_useful: float
    credibility_weighted_avg: float  # Weighted by useful+1
    high_vs_low_delta: float
    divergence_score: float  # Variance in ratings


def compute_task_b_ground_truth(reviews: List[Review]) -> TaskBGroundTruth:
    """Compute ground truth for Task B."""
    n = len(reviews)

    high_useful = [r for r in reviews if r.useful >= 3]
    low_useful = [r for r in reviews if r.useful == 0]

    avg_all = sum(r.stars for r in reviews) / n if n else 0
    avg_high = sum(r.stars for r in high_useful) / len(high_useful) if high_useful else 0
    avg_low = sum(r.stars for r in low_useful) / len(low_useful) if low_useful else 0

    # Weighted average: weight = useful + 1
    total_weight = sum(r.useful + 1 for r in reviews)
    weighted_sum = sum(r.stars * (r.useful + 1) for r in reviews)
    weighted_avg = weighted_sum / total_weight if total_weight else 0

    # Variance
    mean = avg_all
    variance = sum((r.stars - mean) ** 2 for r in reviews) / n if n else 0

    return TaskBGroundTruth(
        n_total=n,
        n_high_useful=len(high_useful),
        n_low_useful=len(low_useful),
        avg_all=round(avg_all, 2),
        avg_high_useful=round(avg_high, 2),
        avg_low_useful=round(avg_low, 2),
        credibility_weighted_avg=round(weighted_avg, 2),
        high_vs_low_delta=round(avg_high - avg_low, 2) if high_useful and low_useful else 0.0,
        divergence_score=round(variance, 2),
    )


def build_task_b_prompt(restaurant: Restaurant, reviews: List[Review]) -> str:
    """Build prompt for Task B."""
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r.date[:10]} | {r.stars}★ | useful={r.useful} | {r.text[:200]}...")

    return f'''Analyze review credibility patterns for this restaurant.

## RESTAURANT: {restaurant.name}

## REVIEWS (with useful vote counts)
{chr(10).join(reviews_text)}

## TASK: Analyze how "useful" votes correlate with ratings

The "useful" field indicates how many other users found this review helpful.
- High useful (>=3): More credible reviews
- Low useful (=0): Less established reviews

Compute:

1. **N_TOTAL**: Total number of reviews
2. **N_HIGH_USEFUL**: Reviews with useful >= 3
3. **N_LOW_USEFUL**: Reviews with useful = 0
4. **AVG_ALL**: Average star rating of all reviews
5. **AVG_HIGH_USEFUL**: Average star rating of high-useful reviews (0 if none)
6. **AVG_LOW_USEFUL**: Average star rating of low-useful reviews (0 if none)
7. **CREDIBILITY_WEIGHTED_AVG**: Weighted average where weight = useful + 1
   Formula: Σ(stars × (useful+1)) / Σ(useful+1)
8. **HIGH_VS_LOW_DELTA**: AVG_HIGH_USEFUL - AVG_LOW_USEFUL (0 if either group empty)
9. **DIVERGENCE_SCORE**: Variance of star ratings = Σ(stars - mean)² / N

## OUTPUT FORMAT

===FINAL ANSWERS===
N_TOTAL: [integer]
N_HIGH_USEFUL: [integer]
N_LOW_USEFUL: [integer]
AVG_ALL: [decimal]
AVG_HIGH_USEFUL: [decimal]
AVG_LOW_USEFUL: [decimal]
CREDIBILITY_WEIGHTED_AVG: [decimal]
HIGH_VS_LOW_DELTA: [decimal]
DIVERGENCE_SCORE: [decimal]
===END===
'''


# =============================================================================
# TASK C: RATING-TEXT ALIGNMENT
# =============================================================================

@dataclass
class TaskCGroundTruth:
    """Ground truth for Task C: Rating-Text Alignment."""
    n_total: int
    n_positive_text: int  # Text with positive sentiment
    n_negative_text: int  # Text with negative sentiment
    n_aligned: int  # Positive text + high stars OR negative text + low stars
    n_misaligned: int
    alignment_ratio: float
    avg_aligned_stars: float
    avg_misaligned_stars: float


def has_positive_sentiment(text: str) -> bool:
    """Simple positive sentiment detection."""
    positive_words = ['amazing', 'excellent', 'great', 'wonderful', 'fantastic',
                     'delicious', 'perfect', 'best', 'love', 'loved', 'awesome',
                     'incredible', 'outstanding', 'superb', 'recommend']
    text_lower = text.lower()
    return any(word in text_lower for word in positive_words)


def has_negative_sentiment(text: str) -> bool:
    """Simple negative sentiment detection."""
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'bad', 'poor',
                     'disappointing', 'disappointed', 'disgusting', 'never again',
                     'waste', 'avoid', 'mediocre', 'overpriced', 'cold']
    text_lower = text.lower()
    return any(word in text_lower for word in negative_words)


def compute_task_c_ground_truth(reviews: List[Review]) -> TaskCGroundTruth:
    """Compute ground truth for Task C."""
    n = len(reviews)

    positive_text = []
    negative_text = []
    aligned = []
    misaligned = []

    for r in reviews:
        is_pos = has_positive_sentiment(r.text)
        is_neg = has_negative_sentiment(r.text)

        if is_pos and not is_neg:
            positive_text.append(r)
            if r.stars >= 4:
                aligned.append(r)
            else:
                misaligned.append(r)
        elif is_neg and not is_pos:
            negative_text.append(r)
            if r.stars <= 2:
                aligned.append(r)
            else:
                misaligned.append(r)
        # Mixed or neutral: skip for alignment calculation

    n_classified = len(positive_text) + len(negative_text)

    return TaskCGroundTruth(
        n_total=n,
        n_positive_text=len(positive_text),
        n_negative_text=len(negative_text),
        n_aligned=len(aligned),
        n_misaligned=len(misaligned),
        alignment_ratio=round(len(aligned) / n_classified, 3) if n_classified else 0.0,
        avg_aligned_stars=round(sum(r.stars for r in aligned) / len(aligned), 2) if aligned else 0.0,
        avg_misaligned_stars=round(sum(r.stars for r in misaligned) / len(misaligned), 2) if misaligned else 0.0,
    )


def build_task_c_prompt(restaurant: Restaurant, reviews: List[Review]) -> str:
    """Build prompt for Task C."""
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r.stars}★ | {r.text[:300]}...")

    return f'''Analyze rating-text alignment in these reviews.

## RESTAURANT: {restaurant.name}

## REVIEWS
{chr(10).join(reviews_text)}

## TASK: Detect alignment between text sentiment and star rating

For each review, determine:
- **Positive text**: Contains positive sentiment words (amazing, excellent, great, delicious, love, recommend, etc.)
- **Negative text**: Contains negative sentiment words (terrible, awful, horrible, disappointing, avoid, etc.)
- **Aligned**: Positive text with 4-5 stars, OR negative text with 1-2 stars
- **Misaligned**: Positive text with 1-3 stars, OR negative text with 3-5 stars

Note: Reviews with mixed or neutral sentiment are excluded from alignment calculation.

Compute:

1. **N_TOTAL**: Total reviews
2. **N_POSITIVE_TEXT**: Reviews with clearly positive sentiment
3. **N_NEGATIVE_TEXT**: Reviews with clearly negative sentiment
4. **N_ALIGNED**: Reviews where sentiment matches rating
5. **N_MISALIGNED**: Reviews where sentiment contradicts rating
6. **ALIGNMENT_RATIO**: N_ALIGNED / (N_POSITIVE_TEXT + N_NEGATIVE_TEXT)
7. **AVG_ALIGNED_STARS**: Average rating of aligned reviews
8. **AVG_MISALIGNED_STARS**: Average rating of misaligned reviews

## OUTPUT FORMAT

===FINAL ANSWERS===
N_TOTAL: [integer]
N_POSITIVE_TEXT: [integer]
N_NEGATIVE_TEXT: [integer]
N_ALIGNED: [integer]
N_MISALIGNED: [integer]
ALIGNMENT_RATIO: [decimal]
AVG_ALIGNED_STARS: [decimal]
AVG_MISALIGNED_STARS: [decimal]
===END===
'''


# =============================================================================
# TASK D: CROSS-ASPECT CORRELATION
# =============================================================================

ASPECTS = ['food', 'service', 'wait', 'value', 'ambiance']

def detect_aspects(text: str) -> Dict[str, str]:
    """Detect aspect mentions and sentiment in text."""
    text_lower = text.lower()
    aspects = {}

    # Food
    food_pos = ['delicious', 'tasty', 'fresh', 'amazing food', 'great food', 'excellent food']
    food_neg = ['bland', 'cold food', 'stale', 'undercooked', 'overcooked', 'tasteless']
    if any(w in text_lower for w in food_pos):
        aspects['food'] = 'positive'
    elif any(w in text_lower for w in food_neg):
        aspects['food'] = 'negative'

    # Service
    service_pos = ['friendly', 'attentive', 'great service', 'excellent service', 'helpful']
    service_neg = ['rude', 'slow service', 'ignored', 'terrible service', 'inattentive']
    if any(w in text_lower for w in service_pos):
        aspects['service'] = 'positive'
    elif any(w in text_lower for w in service_neg):
        aspects['service'] = 'negative'

    # Wait
    wait_pos = ['no wait', 'seated immediately', 'quick', 'fast']
    wait_neg = ['long wait', 'waited forever', 'slow', 'took forever', 'hour wait']
    if any(w in text_lower for w in wait_pos):
        aspects['wait'] = 'positive'
    elif any(w in text_lower for w in wait_neg):
        aspects['wait'] = 'negative'

    # Value
    value_pos = ['great value', 'worth it', 'reasonable price', 'good price', 'affordable']
    value_neg = ['overpriced', 'expensive', 'not worth', 'rip off', 'too pricey']
    if any(w in text_lower for w in value_pos):
        aspects['value'] = 'positive'
    elif any(w in text_lower for w in value_neg):
        aspects['value'] = 'negative'

    # Ambiance
    ambiance_pos = ['cozy', 'nice atmosphere', 'great ambiance', 'romantic', 'beautiful']
    ambiance_neg = ['loud', 'noisy', 'crowded', 'dirty', 'cramped']
    if any(w in text_lower for w in ambiance_pos):
        aspects['ambiance'] = 'positive'
    elif any(w in text_lower for w in ambiance_neg):
        aspects['ambiance'] = 'negative'

    return aspects


@dataclass
class TaskDGroundTruth:
    """Ground truth for Task D: Cross-Aspect Correlation."""
    n_total: int
    n_multi_aspect: int  # Reviews mentioning 2+ aspects
    n_food_positive: int
    n_service_positive: int
    n_wait_negative: int
    n_compensation_pattern: int  # Negative wait + positive food (worth the wait)
    n_systemic_negative: int  # 2+ negative aspects in same review
    avg_multi_aspect_stars: float
    avg_single_aspect_stars: float


def compute_task_d_ground_truth(reviews: List[Review]) -> TaskDGroundTruth:
    """Compute ground truth for Task D."""
    n = len(reviews)

    multi_aspect = []
    single_aspect = []
    food_positive = 0
    service_positive = 0
    wait_negative = 0
    compensation = 0
    systemic_negative = 0

    for r in reviews:
        aspects = detect_aspects(r.text)

        if len(aspects) >= 2:
            multi_aspect.append(r)
        elif len(aspects) == 1:
            single_aspect.append(r)

        if aspects.get('food') == 'positive':
            food_positive += 1
        if aspects.get('service') == 'positive':
            service_positive += 1
        if aspects.get('wait') == 'negative':
            wait_negative += 1

        # Compensation pattern: negative wait + positive food
        if aspects.get('wait') == 'negative' and aspects.get('food') == 'positive':
            compensation += 1

        # Systemic negative: 2+ negative aspects
        neg_count = sum(1 for v in aspects.values() if v == 'negative')
        if neg_count >= 2:
            systemic_negative += 1

    return TaskDGroundTruth(
        n_total=n,
        n_multi_aspect=len(multi_aspect),
        n_food_positive=food_positive,
        n_service_positive=service_positive,
        n_wait_negative=wait_negative,
        n_compensation_pattern=compensation,
        n_systemic_negative=systemic_negative,
        avg_multi_aspect_stars=round(sum(r.stars for r in multi_aspect) / len(multi_aspect), 2) if multi_aspect else 0.0,
        avg_single_aspect_stars=round(sum(r.stars for r in single_aspect) / len(single_aspect), 2) if single_aspect else 0.0,
    )


def build_task_d_prompt(restaurant: Restaurant, reviews: List[Review]) -> str:
    """Build prompt for Task D."""
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r.stars}★ | {r.text[:350]}...")

    return f'''Analyze cross-aspect patterns in these reviews.

## RESTAURANT: {restaurant.name}

## REVIEWS
{chr(10).join(reviews_text)}

## TASK: Detect aspect co-occurrence patterns

For each review, identify which aspects are mentioned and their sentiment:
- **food**: positive (delicious, tasty, fresh) or negative (bland, cold, stale)
- **service**: positive (friendly, attentive) or negative (rude, slow service, ignored)
- **wait**: positive (no wait, quick) or negative (long wait, slow, took forever)
- **value**: positive (great value, worth it) or negative (overpriced, expensive)
- **ambiance**: positive (cozy, nice atmosphere) or negative (loud, noisy, dirty)

Then compute:

1. **N_TOTAL**: Total reviews
2. **N_MULTI_ASPECT**: Reviews mentioning 2+ different aspects
3. **N_FOOD_POSITIVE**: Reviews with positive food mentions
4. **N_SERVICE_POSITIVE**: Reviews with positive service mentions
5. **N_WAIT_NEGATIVE**: Reviews with negative wait mentions
6. **N_COMPENSATION_PATTERN**: Reviews with BOTH negative wait AND positive food ("worth the wait")
7. **N_SYSTEMIC_NEGATIVE**: Reviews with 2+ negative aspects (systemic problem)
8. **AVG_MULTI_ASPECT_STARS**: Average rating of multi-aspect reviews
9. **AVG_SINGLE_ASPECT_STARS**: Average rating of single-aspect reviews

## OUTPUT FORMAT

===FINAL ANSWERS===
N_TOTAL: [integer]
N_MULTI_ASPECT: [integer]
N_FOOD_POSITIVE: [integer]
N_SERVICE_POSITIVE: [integer]
N_WAIT_NEGATIVE: [integer]
N_COMPENSATION_PATTERN: [integer]
N_SYSTEMIC_NEGATIVE: [integer]
AVG_MULTI_ASPECT_STARS: [decimal]
AVG_SINGLE_ASPECT_STARS: [decimal]
===END===
'''


# =============================================================================
# TASK F: EXPECTATION CALIBRATION
# =============================================================================

@dataclass
class TaskFGroundTruth:
    """Ground truth for Task F: Expectation Calibration."""
    n_total: int
    restaurant_price_tier: int
    restaurant_noise_level: str
    n_price_complaints: int
    n_noise_complaints: int
    price_complaint_ratio: float
    noise_complaint_ratio: float
    avg_stars_price_complainers: float
    price_adjusted_score: float  # Lower tiers should have fewer price complaints


def compute_task_f_ground_truth(restaurant: Restaurant, reviews: List[Review]) -> TaskFGroundTruth:
    """Compute ground truth for Task F."""
    n = len(reviews)

    price_complaints = []
    noise_complaints = []

    for r in reviews:
        text_lower = r.text.lower()

        # Price complaints
        if any(w in text_lower for w in ['expensive', 'overpriced', 'pricey', 'not worth', 'rip off']):
            price_complaints.append(r)

        # Noise complaints (only count if restaurant claims to be quiet)
        if any(w in text_lower for w in ['loud', 'noisy', 'couldn\'t hear', 'too loud']):
            noise_complaints.append(r)

    price_ratio = len(price_complaints) / n if n else 0
    noise_ratio = len(noise_complaints) / n if n else 0

    # Price-adjusted score: price_tier ranges 1-4
    # Higher tier should "justify" more price complaints
    # Score = 1 - (price_ratio * (5 - price_tier))
    price_adjusted = 1 - (price_ratio * (5 - restaurant.price_tier))

    return TaskFGroundTruth(
        n_total=n,
        restaurant_price_tier=restaurant.price_tier,
        restaurant_noise_level=str(restaurant.noise_level),
        n_price_complaints=len(price_complaints),
        n_noise_complaints=len(noise_complaints),
        price_complaint_ratio=round(price_ratio, 3),
        noise_complaint_ratio=round(noise_ratio, 3),
        avg_stars_price_complainers=round(sum(r.stars for r in price_complaints) / len(price_complaints), 2) if price_complaints else 0.0,
        price_adjusted_score=round(max(0, price_adjusted), 3),
    )


def build_task_f_prompt(restaurant: Restaurant, reviews: List[Review]) -> str:
    """Build prompt for Task F."""
    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(f"[R{i}] {r.stars}★ | {r.text[:300]}...")

    return f'''Analyze reviews relative to restaurant expectations.

## RESTAURANT: {restaurant.name}
Categories: {restaurant.categories}
Price Tier: {restaurant.price_tier} (1=cheap, 4=expensive)
Noise Level: {restaurant.noise_level}
Good for Kids: {restaurant.is_good_for_kids}

## REVIEWS
{chr(10).join(reviews_text)}

## TASK: Evaluate complaints relative to restaurant type

Compute:

1. **N_TOTAL**: Total reviews
2. **RESTAURANT_PRICE_TIER**: The restaurant's price tier (given above)
3. **RESTAURANT_NOISE_LEVEL**: The restaurant's noise level (given above)
4. **N_PRICE_COMPLAINTS**: Reviews mentioning price issues (expensive, overpriced, pricey, not worth, rip off)
5. **N_NOISE_COMPLAINTS**: Reviews mentioning noise issues (loud, noisy, couldn't hear)
6. **PRICE_COMPLAINT_RATIO**: N_PRICE_COMPLAINTS / N_TOTAL (3 decimal places)
7. **NOISE_COMPLAINT_RATIO**: N_NOISE_COMPLAINTS / N_TOTAL (3 decimal places)
8. **AVG_STARS_PRICE_COMPLAINERS**: Average rating of reviews with price complaints (0 if none)
9. **PRICE_ADJUSTED_SCORE**: 1 - (PRICE_COMPLAINT_RATIO × (5 - PRICE_TIER))
   This penalizes cheap restaurants more for price complaints (they should be affordable)

## OUTPUT FORMAT

===FINAL ANSWERS===
N_TOTAL: [integer]
RESTAURANT_PRICE_TIER: [integer]
RESTAURANT_NOISE_LEVEL: [string]
N_PRICE_COMPLAINTS: [integer]
N_NOISE_COMPLAINTS: [integer]
PRICE_COMPLAINT_RATIO: [decimal]
NOISE_COMPLAINT_RATIO: [decimal]
AVG_STARS_PRICE_COMPLAINERS: [decimal]
PRICE_ADJUSTED_SCORE: [decimal]
===END===
'''


# =============================================================================
# PARSING & EVALUATION
# =============================================================================

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


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_task(task_id: str, restaurant_file: str, max_reviews: int = 100, verbose: bool = True) -> Dict:
    """Run a single task on a single restaurant."""
    restaurant, reviews = load_restaurant_data(restaurant_file, max_reviews)

    # Get task-specific functions
    if task_id == 'A':
        gt = compute_task_a_ground_truth(reviews)
        prompt = build_task_a_prompt(restaurant, reviews)
        tolerances = {'delta': 0.1, 'avg_recent': 0.1, 'avg_historical': 0.1}
    elif task_id == 'B':
        gt = compute_task_b_ground_truth(reviews)
        prompt = build_task_b_prompt(restaurant, reviews)
        tolerances = {'avg_all': 0.1, 'avg_high_useful': 0.1, 'avg_low_useful': 0.1,
                     'credibility_weighted_avg': 0.1, 'divergence_score': 0.1}
    elif task_id == 'C':
        gt = compute_task_c_ground_truth(reviews)
        prompt = build_task_c_prompt(restaurant, reviews)
        tolerances = {'alignment_ratio': 0.1, 'avg_aligned_stars': 0.2, 'avg_misaligned_stars': 0.2}
    elif task_id == 'D':
        gt = compute_task_d_ground_truth(reviews)
        prompt = build_task_d_prompt(restaurant, reviews)
        tolerances = {'avg_multi_aspect_stars': 0.2, 'avg_single_aspect_stars': 0.2}
    elif task_id == 'F':
        gt = compute_task_f_ground_truth(restaurant, reviews)
        prompt = build_task_f_prompt(restaurant, reviews)
        tolerances = {'price_adjusted_score': 0.1}
    else:
        raise ValueError(f"Unknown task: {task_id}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {restaurant.name}")
        print(f"{'='*60}")
        print(f"Reviews: {len(reviews)}")
        print(f"Prompt length: {len(prompt):,} chars")

    # Call LLM
    if verbose:
        print("Calling LLM...")

    response = call_llm(prompt)
    parsed = parse_response(response)
    results = evaluate_task(parsed, gt, tolerances)

    # Save debug
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = DEBUG_DIR / f"task_{task_id}_{restaurant.name[:20]}_{timestamp}.json"

    debug_data = {
        'task': task_id,
        'restaurant': restaurant.name,
        'n_reviews': len(reviews),
        'prompt': prompt,
        'response': response,
        'ground_truth': asdict(gt),
        'parsed': parsed,
        'results': results,
    }

    with open(debug_file, 'w') as f:
        json.dump(debug_data, f, indent=2, default=str)

    if verbose:
        print(f"\n--- Results ---")
        for field, result in results.items():
            if field.startswith('_'):
                continue
            status = "✓" if result['score'] >= 0.9 else "~" if result['score'] >= 0.5 else "✗"
            print(f"  {field}: {result['expected']} vs {result['predicted']} ({result['score']:.2f}) {status}")

        print(f"\nTotal Score: {results['_total_score']:.3f}")
        print(f"Debug saved: {debug_file}")

    return results


def run_all_tasks(max_reviews: int = 100):
    """Run all tasks on all restaurants."""
    tasks = ['A', 'B', 'C', 'D', 'F']

    all_results = {}

    for restaurant_file in RESTAURANTS:
        restaurant_name = restaurant_file.split('_')[0]
        all_results[restaurant_name] = {}

        for task_id in tasks:
            print(f"\n{'#'*60}")
            print(f"Running Task {task_id} on {restaurant_name}")
            print(f"{'#'*60}")

            try:
                results = run_task(task_id, restaurant_file, max_reviews, verbose=True)
                all_results[restaurant_name][task_id] = results['_total_score']
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[restaurant_name][task_id] = 0.0

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Restaurant':<20}", end='')
    for task in tasks:
        print(f"Task {task:<6}", end='')
    print("Avg")
    print("-" * (20 + 8 * len(tasks) + 6))

    for restaurant_name, task_scores in all_results.items():
        print(f"{restaurant_name:<20}", end='')
        scores = []
        for task in tasks:
            score = task_scores.get(task, 0)
            scores.append(score)
            print(f"{score:.2f}    ", end='')
        print(f"{sum(scores)/len(scores):.2f}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L2 Task Evaluation")
    parser.add_argument("--task", default="A", help="Task ID (A, B, C, D, F) or 'all'")
    parser.add_argument("--restaurant", default="Acme", help="Restaurant name prefix or 'all'")
    parser.add_argument("--max-reviews", type=int, default=100)
    args = parser.parse_args()

    configure(temperature=0.0)

    if args.task.lower() == 'all' or args.restaurant.lower() == 'all':
        run_all_tasks(args.max_reviews)
    else:
        # Find matching restaurant file
        restaurant_file = None
        for f in RESTAURANTS:
            if args.restaurant.lower() in f.lower():
                restaurant_file = f
                break

        if not restaurant_file:
            print(f"Restaurant not found: {args.restaurant}")
            print(f"Available: {RESTAURANTS}")
            sys.exit(1)

        run_task(args.task.upper(), restaurant_file, args.max_reviews)
