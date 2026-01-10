#!/usr/bin/env python3
"""
L2 Task Definitions

Each task defines:
- GroundTruth dataclass
- compute_ground_truth(reviews, restaurant) -> GroundTruth
- PROMPT: str (pure task instructions, no formatting)
- TOLERANCES: dict (field tolerances for scoring)
"""

from dataclasses import dataclass
from typing import List, Dict, Any




@dataclass
class TaskCGroundTruth:
    n_total: int
    n_positive_text: int
    n_negative_text: int
    n_aligned: int
    n_misaligned: int
    alignment_ratio: float
    avg_aligned_stars: float
    avg_misaligned_stars: float


def _has_positive_sentiment(text: str) -> bool:
    positive_words = ['amazing', 'excellent', 'great', 'wonderful', 'fantastic',
                     'delicious', 'perfect', 'best', 'love', 'loved', 'awesome',
                     'incredible', 'outstanding', 'superb', 'recommend']
    text_lower = text.lower()
    return any(word in text_lower for word in positive_words)


def _has_negative_sentiment(text: str) -> bool:
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'bad', 'poor',
                     'disappointing', 'disappointed', 'disgusting', 'never again',
                     'waste', 'avoid', 'mediocre', 'overpriced', 'cold']
    text_lower = text.lower()
    return any(word in text_lower for word in negative_words)


def compute_task_c_ground_truth(reviews: List[Any], restaurant: Any = None) -> TaskCGroundTruth:
    n = len(reviews)

    positive_text = []
    negative_text = []
    aligned = []
    misaligned = []

    for r in reviews:
        is_pos = _has_positive_sentiment(r['text'])
        is_neg = _has_negative_sentiment(r['text'])

        if is_pos and not is_neg:
            positive_text.append(r)
            if r['stars'] >= 4:
                aligned.append(r)
            else:
                misaligned.append(r)
        elif is_neg and not is_pos:
            negative_text.append(r)
            if r['stars'] <= 2:
                aligned.append(r)
            else:
                misaligned.append(r)

    n_classified = len(positive_text) + len(negative_text)

    return TaskCGroundTruth(
        n_total=n,
        n_positive_text=len(positive_text),
        n_negative_text=len(negative_text),
        n_aligned=len(aligned),
        n_misaligned=len(misaligned),
        alignment_ratio=round(len(aligned) / n_classified, 3) if n_classified else 0.0,
        avg_aligned_stars=round(sum(r['stars'] for r in aligned) / len(aligned), 2) if aligned else 0.0,
        avg_misaligned_stars=round(sum(r['stars'] for r in misaligned) / len(misaligned), 2) if misaligned else 0.0,
    )


TASK_C_PROMPT = """Analyze rating-text alignment in the reviews.

For each review, determine:
- Positive text: Contains positive sentiment words (amazing, excellent, great, delicious, love, recommend, etc.)
- Negative text: Contains negative sentiment words (terrible, awful, horrible, disappointing, avoid, etc.)
- Aligned: Positive text with 4-5 stars, OR negative text with 1-2 stars
- Misaligned: Positive text with 1-3 stars, OR negative text with 3-5 stars

Note: Reviews with mixed sentiment (both positive and negative words) or neutral sentiment are excluded.

Compute:
1. N_TOTAL: Total reviews
2. N_POSITIVE_TEXT: Reviews with clearly positive sentiment (no negative words)
3. N_NEGATIVE_TEXT: Reviews with clearly negative sentiment (no positive words)
4. N_ALIGNED: Reviews where sentiment matches rating
5. N_MISALIGNED: Reviews where sentiment contradicts rating
6. ALIGNMENT_RATIO: N_ALIGNED / (N_POSITIVE_TEXT + N_NEGATIVE_TEXT) (3 decimal places)
7. AVG_ALIGNED_STARS: Average rating of aligned reviews (0 if none, 2 decimal places)
8. AVG_MISALIGNED_STARS: Average rating of misaligned reviews (0 if none, 2 decimal places)"""

TASK_C_TOLERANCES = {'alignment_ratio': 0.1, 'avg_aligned_stars': 0.2, 'avg_misaligned_stars': 0.2}


# =============================================================================
# TASK D: CROSS-ASPECT CORRELATION
# =============================================================================

def _detect_aspects(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    aspects = {}

    food_pos = ['delicious', 'tasty', 'fresh', 'amazing food', 'great food', 'excellent food']
    food_neg = ['bland', 'cold food', 'stale', 'undercooked', 'overcooked', 'tasteless']
    if any(w in text_lower for w in food_pos):
        aspects['food'] = 'positive'
    elif any(w in text_lower for w in food_neg):
        aspects['food'] = 'negative'

    service_pos = ['friendly', 'attentive', 'great service', 'excellent service', 'helpful']
    service_neg = ['rude', 'slow service', 'ignored', 'terrible service', 'inattentive']
    if any(w in text_lower for w in service_pos):
        aspects['service'] = 'positive'
    elif any(w in text_lower for w in service_neg):
        aspects['service'] = 'negative'

    wait_pos = ['no wait', 'seated immediately', 'quick', 'fast']
    wait_neg = ['long wait', 'waited forever', 'slow', 'took forever', 'hour wait']
    if any(w in text_lower for w in wait_pos):
        aspects['wait'] = 'positive'
    elif any(w in text_lower for w in wait_neg):
        aspects['wait'] = 'negative'

    value_pos = ['great value', 'worth it', 'reasonable price', 'good price', 'affordable']
    value_neg = ['overpriced', 'expensive', 'not worth', 'rip off', 'too pricey']
    if any(w in text_lower for w in value_pos):
        aspects['value'] = 'positive'
    elif any(w in text_lower for w in value_neg):
        aspects['value'] = 'negative'

    ambiance_pos = ['cozy', 'nice atmosphere', 'great ambiance', 'romantic', 'beautiful']
    ambiance_neg = ['loud', 'noisy', 'crowded', 'dirty', 'cramped']
    if any(w in text_lower for w in ambiance_pos):
        aspects['ambiance'] = 'positive'
    elif any(w in text_lower for w in ambiance_neg):
        aspects['ambiance'] = 'negative'

    return aspects


@dataclass
class TaskDGroundTruth:
    n_total: int
    n_multi_aspect: int
    n_food_positive: int
    n_service_positive: int
    n_wait_negative: int
    n_compensation_pattern: int
    n_systemic_negative: int
    avg_multi_aspect_stars: float
    avg_single_aspect_stars: float


def compute_task_d_ground_truth(reviews: List[Any], restaurant: Any = None) -> TaskDGroundTruth:
    n = len(reviews)

    multi_aspect = []
    single_aspect = []
    food_positive = 0
    service_positive = 0
    wait_negative = 0
    compensation = 0
    systemic_negative = 0

    for r in reviews:
        aspects = _detect_aspects(r['text'])

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

        if aspects.get('wait') == 'negative' and aspects.get('food') == 'positive':
            compensation += 1

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
        avg_multi_aspect_stars=round(sum(r['stars'] for r in multi_aspect) / len(multi_aspect), 2) if multi_aspect else 0.0,
        avg_single_aspect_stars=round(sum(r['stars'] for r in single_aspect) / len(single_aspect), 2) if single_aspect else 0.0,
    )


TASK_D_PROMPT = """Analyze cross-aspect patterns in the reviews.

For each review, identify which aspects are mentioned and their sentiment:
- food: positive (delicious, tasty, fresh) or negative (bland, cold, stale)
- service: positive (friendly, attentive) or negative (rude, slow service, ignored)
- wait: positive (no wait, quick) or negative (long wait, slow, took forever)
- value: positive (great value, worth it) or negative (overpriced, expensive)
- ambiance: positive (cozy, nice atmosphere) or negative (loud, noisy, dirty)

Compute:
1. N_TOTAL: Total reviews
2. N_MULTI_ASPECT: Reviews mentioning 2+ different aspects with sentiment
3. N_FOOD_POSITIVE: Reviews with positive food mentions
4. N_SERVICE_POSITIVE: Reviews with positive service mentions
5. N_WAIT_NEGATIVE: Reviews with negative wait mentions
6. N_COMPENSATION_PATTERN: Reviews with BOTH negative wait AND positive food ("worth the wait")
7. N_SYSTEMIC_NEGATIVE: Reviews with 2+ negative aspects (systemic problem)
8. AVG_MULTI_ASPECT_STARS: Average rating of multi-aspect reviews (0 if none, 2 decimal places)
9. AVG_SINGLE_ASPECT_STARS: Average rating of single-aspect reviews (0 if none, 2 decimal places)"""

TASK_D_TOLERANCES = {'avg_multi_aspect_stars': 0.2, 'avg_single_aspect_stars': 0.2}


# =============================================================================
# TASK F: EXPECTATION CALIBRATION
# =============================================================================

@dataclass
class TaskFGroundTruth:
    n_total: int
    restaurant_price_tier: int
    restaurant_noise_level: str
    n_price_complaints: int
    n_noise_complaints: int
    price_complaint_ratio: float
    noise_complaint_ratio: float
    avg_stars_price_complainers: float
    price_adjusted_score: float


@dataclass
class TaskGGroundTruth:
    n_reviews_with_wait: int
    n_worth_it: int
    n_not_worth_it: int
    wait_redeemability_ratio: float
    most_common_wait_complaint: str


@dataclass
class TaskHGroundTruth:
    top_staff_member: str
    n_mentions: int
    avg_stars_with_staff: float
    staff_sentiment_summary: str



def compute_task_f_ground_truth(reviews: List[Any], restaurant: Any) -> TaskFGroundTruth:
    n = len(reviews)

    price_complaints = []
    noise_complaints = []

    for r in reviews:
        text_lower = r['text'].lower()

        if any(w in text_lower for w in ['expensive', 'overpriced', 'pricey', 'not worth', 'rip off']):
            price_complaints.append(r)

        if any(w in text_lower for w in ['loud', 'noisy', 'couldn\'t hear', 'too loud']):
            noise_complaints.append(r)

    price_ratio = len(price_complaints) / n if n else 0
    noise_ratio = len(noise_complaints) / n if n else 0

    attrs = restaurant.get('attributes', {})
    price_tier = int(attrs.get('RestaurantsPriceRange2', 2))
    noise_level = attrs.get('NoiseLevel', 'average')
    price_adjusted = 1 - (price_ratio * (5 - price_tier))

    return TaskFGroundTruth(
        n_total=n,
        restaurant_price_tier=price_tier,
        restaurant_noise_level=str(noise_level),
        n_price_complaints=len(price_complaints),
        n_noise_complaints=len(noise_complaints),
        price_complaint_ratio=round(price_ratio, 3),
        noise_complaint_ratio=round(noise_ratio, 3),
        avg_stars_price_complainers=round(sum(r['stars'] for r in price_complaints) / len(price_complaints), 2) if price_complaints else 0.0,
        price_adjusted_score=round(max(0, price_adjusted), 3),
    )


def compute_task_g_ground_truth(reviews: List[Any], restaurant: Any) -> TaskGGroundTruth:
    """
    Task G: Conditional Logic (Wait Times & "Worth It").

    Logic:
    1. Identify reviews mentioning "wait", "line", "queue", "table".
    2. Classify as "Worth It" if they contain: "worth it", "worth the wait", "moved fast", "didn't mind".
    3. Classify as "Not Worth It" if they contain: "too long", "ridiculous", "never again", "slow", "forever" AND NOT in "Worth It" set.
    """
    wait_reviews = []
    worth_it = []
    not_worth_it = []
    
    wait_keywords = ['wait', 'line', 'queue', 'table', 'reservation']
    worth_keywords = ['worth it', 'worth the', 'moved fast', "didn't mind", 'not bad', 'quick']
    neg_keywords = ['too long', 'ridiculous', 'never again', 'slow', 'forever', 'awful', 'rude']

    for r in reviews:
        text = r['text'].lower()
        if any(w in text for w in wait_keywords):
            wait_reviews.append(r)
            if any(w in text for w in worth_keywords):
                worth_it.append(r)
            elif any(w in text for w in neg_keywords):
                not_worth_it.append(r)
                
    n_wait = len(wait_reviews)
    n_worth = len(worth_it)
    n_not = len(not_worth_it)
    
    ratio = n_worth / n_wait if n_wait > 0 else 0.0
    
    # Simple heuristic for most common complaint word in not_worth_it
    complaint_counts = {k: 0 for k in neg_keywords}
    for r in not_worth_it:
        text = r['text'].lower()
        for k in neg_keywords:
            if k in text:
                complaint_counts[k] += 1
    
    most_common = max(complaint_counts.items(), key=lambda x: x[1])[0] if n_not > 0 else "none"

    return TaskGGroundTruth(
        n_reviews_with_wait=n_wait,
        n_worth_it=n_worth,
        n_not_worth_it=n_not,
        wait_redeemability_ratio=round(ratio, 3),
        most_common_wait_complaint=most_common
    )


@dataclass
class TaskIGroundTruth:
    top_user_name: str
    top_user_useful_count: int
    targeted_dish: str
    n_other_reviews_of_dish: int
    avg_stars_contradicting_dish: float


def compute_task_i_ground_truth(reviews: List[Any], restaurant: Any) -> TaskIGroundTruth:
    """
    Task I: Multi-Hop Quantitative Reasoning.
    1. Find user with highest 'useful' count.
    2. Identify dish they mentioned (heuristic: first capitalized word after 'order' or 'had').
    3. Find all OTHER reviews mentioning that dish.
    4. Calculate avg stars of those other reviews.
    """
    if not reviews:
        return TaskIGroundTruth("none", 0, "none", 0, 0.0)
        
    # 1. Top useful user
    top_user_rev = max(reviews, key=lambda x: x.get('useful', 0))
    user_id = top_user_rev.get('user_id', 'unknown')
    # Since we don't have user names in the raw reviews dict sometimes, we use user_id or 'User [ID]'
    user_display = f"User {user_id[:5]}"
    useful_count = top_user_rev.get('useful', 0)
    
    # 2. Targeted dish (very simple extraction for GT)
    # We look for common dishes in philly cafes data
    dishes = ['coffee', 'espresso', 'latte', 'gelato', 'pasta', 'sandwich', 'salad', 'gnocchi', 'ravioli', 'meatball', 'octopus', 'steak']
    target_dish = "none"
    text_lower = top_user_rev['text'].lower()
    for d in dishes:
        if d in text_lower:
            target_dish = d
            break
            
    # 3. Other reviews mentioning dish
    other_revs = [r for r in reviews if r['review_id'] != top_user_rev['review_id'] and target_dish in r['text'].lower()]
    
    avg_stars = sum(r['stars'] for r in other_revs) / len(other_revs) if other_revs else 0.0
    
    return TaskIGroundTruth(
        top_user_name=user_display,
        top_user_useful_count=useful_count,
        targeted_dish=target_dish,
        n_other_reviews_of_dish=len(other_revs),
        avg_stars_contradicting_dish=round(avg_stars, 2)
    )

def compute_task_h_ground_truth(reviews: List[Any], restaurant: Any) -> TaskHGroundTruth:
    """
    Task H: Entity Resolution (Staff Aggregation).
    """
    import re
    from collections import defaultdict
    patterns = [
        r'(?:server|waiter|waitress|bartender|host|hostess|manager|ask for|nice lady|shout out to)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+was our (?:server|waiter|waitress|bartender)'
    ]
    staff_mentions = defaultdict(list)
    for r in reviews:
        text = r['text']
        found_names = set()
        for p in patterns:
            matches = re.findall(p, text)
            for m in matches:
                if len(m) > 2 and m.lower() not in ['he', 'she', 'they', 'the', 'this', 'that', 'it', 'if']:
                    found_names.add(m)
        for name in found_names:
            staff_mentions[name].append(r)
            
    if not staff_mentions:
        return TaskHGroundTruth("none", 0, 0.0, "neutral")
        
    top_staff = max(staff_mentions.items(), key=lambda x: len(x[1]))
    name, revs = top_staff
    avg_stars = sum(r['stars'] for r in revs) / len(revs)
    sentiment = "positive" if avg_stars >= 4.0 else "negative" if avg_stars <= 2.5 else "mixed"
    
    return TaskHGroundTruth(
        top_staff_member=name,
        n_mentions=len(revs),
        avg_stars_with_staff=round(avg_stars, 2),
        staff_sentiment_summary=sentiment
    )


@dataclass
class TaskJGroundTruth:
    avg_useful_service: float
    avg_useful_food: float
    prioritized_aspect: str
    avg_stars_prioritized: float


def compute_task_j_ground_truth(reviews: List[Any], restaurant: Any) -> TaskJGroundTruth:
    """
    Task J: Weighted Aspect Priority.
    """
    service_revs = [r for r in reviews if any(w in r['text'].lower() for w in ['service', 'waitress', 'waiter', 'server', 'staff'])]
    food_revs = [r for r in reviews if any(w in r['text'].lower() for w in ['food', 'dish', 'meal', 'eat', 'delicious', 'tasty'])]
    
    avg_u_service = sum(r.get('useful', 0) for r in service_revs) / len(service_revs) if service_revs else 0.0
    avg_u_food = sum(r.get('useful', 0) for r in food_revs) / len(food_revs) if food_revs else 0.0
    
    if avg_u_service > avg_u_food:
        priority = "service"
        target_revs = service_revs
    elif avg_u_food > avg_u_service:
        priority = "food"
        target_revs = food_revs
    else:
        priority = "draw"
        target_revs = service_revs + food_revs
        
    avg_stars = sum(r['stars'] for r in target_revs) / len(target_revs) if target_revs else 0.0
    
    return TaskJGroundTruth(
        avg_useful_service=round(avg_u_service, 2),
        avg_useful_food=round(avg_u_food, 2),
        prioritized_aspect=priority,
        avg_stars_prioritized=round(avg_stars, 2)
    )


@dataclass
class TaskKGroundTruth:
    top_staff_member: str
    earliest_rating: float
    latest_rating: float
    drift: str


def compute_task_k_ground_truth(reviews: List[Any], restaurant: Any) -> TaskKGroundTruth:
    """
    Task K: Entity Sentiment Drift.
    """
    # Reuse staff extraction logic from Task H
    import re
    from collections import defaultdict
    patterns = [r'(?:server|waiter|waitress|bartender|host|hostess|manager|ask for|nice lady|shout out to)\s+([A-Z][a-z]+)']
    staff_mentions = defaultdict(list)
    for r in reviews:
        for p in patterns:
            matches = re.findall(p, r['text'])
            for m in matches:
                if len(m) > 2 and m.lower() not in ['he', 'she', 'they', 'the', 'this']:
                    staff_mentions[m].append(r)
                    
    if not staff_mentions:
        return TaskKGroundTruth("none", 0.0, 0.0, "stable")
        
    top_staff = max(staff_mentions.items(), key=lambda x: len(x[1]))
    name, revs = top_staff
    
    # Sort by date
    revs_sorted = sorted(revs, key=lambda x: x['date'])
    earliest = revs_sorted[0]['stars']
    latest = revs_sorted[-1]['stars']
    
    if latest > earliest:
        drift = "improving"
    elif latest < earliest:
        drift = "declining"
    else:
        drift = "stable"
        
    return TaskKGroundTruth(
        top_staff_member=name,
        earliest_rating=float(earliest),
        latest_rating=float(latest),
        drift=drift
    )


TASK_F_PROMPT = """Analyze reviews relative to restaurant expectations.

The restaurant metadata includes price tier and noise level. Evaluate how complaints relate to these attributes.

Compute:
1. N_TOTAL: Total reviews
2. RESTAURANT_PRICE_TIER: The restaurant's price tier from metadata (1-4)
3. RESTAURANT_NOISE_LEVEL: The restaurant's noise level from metadata
4. N_PRICE_COMPLAINTS: Reviews mentioning price issues (expensive, overpriced, pricey, not worth, rip off)
5. N_NOISE_COMPLAINTS: Reviews mentioning noise issues (loud, noisy, couldn't hear)
6. PRICE_COMPLAINT_RATIO: N_PRICE_COMPLAINTS / N_TOTAL (3 decimal places)
7. NOISE_COMPLAINT_RATIO: N_NOISE_COMPLAINTS / N_TOTAL (3 decimal places)
8. AVG_STARS_PRICE_COMPLAINERS: Average rating of reviews with price complaints (0 if none, 2 decimal places)
9. PRICE_ADJUSTED_SCORE: 1 - (PRICE_COMPLAINT_RATIO * (5 - PRICE_TIER)) (3 decimal places)"""

TASK_F_TOLERANCES = {'price_adjusted_score': 0.1}

TASK_G_PROMPT = """Analyze the relationship between wait times and customer satisfaction.

Compute:
1. N_REVIEWS_WITH_WAIT: Count of reviews mentioning 'wait', 'line', 'queue', 'table', 'reservation'.
2. N_WORTH_IT: From wait group, count explicitly positive wait experiences ("worth it", "moved fast").
3. N_NOT_WORTH_IT: From wait group, count negative wait experiences ("too long", "slow").
4. WAIT_REDEEMABILITY_RATIO: N_WORTH_IT / N_REVIEWS_WITH_WAIT (3 decimal places).
5. MOST_COMMON_WAIT_COMPLAINT: Most frequent complaint keyword in negative wait reviews (e.g. 'too long', 'slow')."""

TASK_G_TOLERANCES = {'wait_redeemability_ratio': 0.1}

TASK_H_PROMPT = """Identify specific staff members mentioned by name and aggregate their performance.

Compute:
1. TOP_STAFF_MEMBER: The single capitalized name most frequently mentioned after role keywords (server, waiter, etc.).
2. N_MENTIONS: Number of reviews mentioning this top staff member.
3. AVG_STARS_WITH_STAFF: Average star rating of reviews mentioning this person (2 decimal places).
4. STAFF_SENTIMENT_SUMMARY: 'positive' (avg>=4.0), 'negative' (avg<=2.5), or 'mixed'."""

TASK_H_TOLERANCES = {'avg_stars_with_staff': 0.5}


# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_I_PROMPT = """Perform multi-hop quantitative analysis to identify user influence and contradictions.

Compute:
1. TOP_USER_NAME: The name/ID of the user whose review has the highest 'useful' count. Format as 'User [First 5 chars of ID]'.
2. TOP_USER_USEFUL_COUNT: The useful count of that specific review.
3. TARGETED_DISH: A specific dish mentioned in that user's review (e.g. coffee, gelato, pasta). Pick one.
4. N_OTHER_REVIEWS_OF_DISH: Count of ALL OTHER reviews in the set that mention this same dish.
5. AVG_STARS_CONTRADICTING_DISH: Average star rating of those other reviews (2 decimal places)."""

TASK_I_TOLERANCES = {'avg_stars_contradicting_dish': 0.2}

# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_J_PROMPT = """Analyze which aspect of the restaurant is prioritized by 'useful' reviews.

Compute:
1. AVG_USEFUL_SERVICE: Average useful count for reviews mentioning service-related terms (2 decimal places).
2. AVG_USEFUL_FOOD: Average useful count for reviews mentioning food-related terms (2 decimal places).
3. PRIORITIZED_ASPECT: "service" if AVG_USEFUL_SERVICE > AVG_USEFUL_FOOD, "food" if vice-versa, else "draw".
4. AVG_STARS_PRIORITIZED: Average star rating of the reviews in the prioritized group (2 decimal places)."""

TASK_J_TOLERANCES = {'avg_useful_service': 0.5, 'avg_useful_food': 0.5, 'avg_stars_prioritized': 0.2}

TASK_K_PROMPT = """Analyze the sentiment drift for the most mentioned staff member.

Compute:
1. TOP_STAFF_MEMBER: The name of the staff member mentioned most frequently.
2. EARLIEST_RATING: The star rating of the chronologically earliest review mentioning this staff member.
3. LATEST_RATING: The star rating of the chronologically latest review mentioning this staff member.
4. DRIFT: "improving" if LATEST_RATING > EARLIEST_RATING, "declining" if LATEST_RATING < EARLIEST_RATING, else "stable"."""

TASK_K_TOLERANCES = {}

# =============================================================================
# TASK G1: SEVERE ALLERGY SAFETY
# =============================================================================

# =============================================================================
# TASK G1: SEVERE ALLERGY SAFETY (v14: SEMANTIC REASONING)
# =============================================================================

@dataclass
class TaskG1GroundTruth:
    # Semantic Primitives (Scored)
    firsthand_severe_count: int
    safety_trajectory: str
    false_assurance_count: int
    evidence_consensus: float
    
    # Final Synthesis (Scored)
    final_risk_score: float
    verdict: str


# Global cache for semantic GT
G1_SEMANTIC_GT_CACHE = None

def load_g1_semantic_gt():
    global G1_SEMANTIC_GT_CACHE
    if G1_SEMANTIC_GT_CACHE is not None:
        return G1_SEMANTIC_GT_CACHE
        
    try:
        import json
        from pathlib import Path
        gt_path = Path(__file__).parent / 'data' / 'semantic_gt' / 'all_judgments.json'
        if gt_path.exists():
            print(f"Loading semantic GT from {gt_path}")
            with open(gt_path, 'r') as f:
                data = json.load(f)
                # Index by restaurant name
                G1_SEMANTIC_GT_CACHE = {item['restaurant']: item for item in data}
        else:
            print(f"WARNING: Semantic GT file not found at {gt_path}")
            G1_SEMANTIC_GT_CACHE = {}
    except Exception as e:
        print(f"ERROR loading semantic GT: {e}")
        G1_SEMANTIC_GT_CACHE = {}
        
    return G1_SEMANTIC_GT_CACHE


def compute_task_g1_ground_truth(reviews: List[Any], restaurant: Any) -> TaskG1GroundTruth:
    """
    Task G1 v14: Severe Peanut Allergy (Semantic Reasoning).
    
    Uses LLM-generated ground truth for semantic primitives which cannot 
    be reliably computed via keyword matching.
    """
    # Try to load expert/LLM judgment from cache
    gt_cache = load_g1_semantic_gt()
    res_name = restaurant.get('name', 'Unknown')
    
    if res_name in gt_cache:
        gt = gt_cache[res_name]
        return TaskG1GroundTruth(
            firsthand_severe_count=int(gt.get('firsthand_severe_count', 0)),
            safety_trajectory=str(gt.get('safety_trajectory', 'stable')),
            false_assurance_count=int(gt.get('false_assurance_count', 0)),
            evidence_consensus=float(gt.get('evidence_consensus', 0.5)),
            final_risk_score=float(gt.get('final_risk_score', 5.0)),
            verdict=str(gt.get('verdict', 'High Risk'))
        )
    
    # Fallback for validation/unknown restaurants (Simple heuristic)
    # This ensures code doesn't crash if cache is missing or for new data
    severe = sum(1 for r in reviews if 'reaction' in r['text'].lower() or 'hospital' in r['text'].lower())
    pos = sum(1 for r in reviews if 'accommodat' in r['text'].lower())
    
    return TaskG1GroundTruth(
        firsthand_severe_count=severe,
        safety_trajectory="stable",
        false_assurance_count=0,
        evidence_consensus=0.5 if (severe > 0 and pos > 0) else 1.0,
        final_risk_score=10.0 if severe > 0 else 2.0,
        verdict="Critical Risk" if severe > 0 else "Low Risk"
    )

TASK_G1_PROMPT = """Analyze the reviews using SEMANTIC UNDERSTANDING (not just keyword matching) to assess severe peanut allergy safety.

Compute these 4 semantic primitives:

1. firsthand_severe_count (int):
   Count reviews describing a PERSONAL severe allergic reaction (I/we/my child).
   EXCLUDE hearsay ("I heard...") or warnings.

2. safety_trajectory (str):
   Assess if safety is 'improving', 'stable', or 'worsening' over time.
   Consider management changes and recent vs old incidents.

3. false_assurance_count (int):
   Count reviews where the restaurant CLAIMED safety (e.g. "allergy menu") BUT the customer still had issues.

4. evidence_consensus (float 0.0-1.0):
   Among reviews discussing allergy, assess agreement.
   0.0 = total conflict (safe vs unsafe), 1.0 = total consensus.

Finally, determine:
- final_risk_score (float 0-20): 0-4=Low, 4-8=High, 8+=Critical
- verdict (str): "Low Risk", "High Risk", or "Critical Risk"

Output JSON."""

TASK_G1_TOLERANCES = {
    'firsthand_severe_count': 0,  # Strict count
    'safety_trajectory': 0,  # Exact string match
    'false_assurance_count': 0,  # Strict count
    'evidence_consensus': 0.2,  # Float tolerance
    'final_risk_score': 2.0,  # Score tolerance
}

# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_REGISTRY = {
    'C': {
        'name': 'Rating-Text Alignment',
        'ground_truth_class': TaskCGroundTruth,
        'compute_ground_truth': compute_task_c_ground_truth,
        'prompt': TASK_C_PROMPT,
        'tolerances': TASK_C_TOLERANCES,
    },
    'D': {
        'name': 'Cross-Aspect Correlation',
        'ground_truth_class': TaskDGroundTruth,
        'compute_ground_truth': compute_task_d_ground_truth,
        'prompt': TASK_D_PROMPT,
        'tolerances': TASK_D_TOLERANCES,
    },
    'F': {
        'name': 'Expectation Calibration',
        'ground_truth_class': TaskFGroundTruth,
        'compute_ground_truth': compute_task_f_ground_truth,
        'prompt': TASK_F_PROMPT,
        'tolerances': TASK_F_TOLERANCES,
    },
    'G': {
        'name': 'Conditional Sentiment',
        'ground_truth_class': TaskGGroundTruth,
        'compute_ground_truth': compute_task_g_ground_truth,
        'prompt': TASK_G_PROMPT,
        'tolerances': TASK_G_TOLERANCES,
    },
    'H': {
        'name': 'Staff Entity Resolution',
        'ground_truth_class': TaskHGroundTruth,
        'compute_ground_truth': compute_task_h_ground_truth,
        'prompt': TASK_H_PROMPT,
        'tolerances': TASK_H_TOLERANCES,
    },
    'I': {
        'name': 'Multi-Hop Reasoning',
        'ground_truth_class': TaskIGroundTruth,
        'compute_ground_truth': compute_task_i_ground_truth,
        'prompt': TASK_I_PROMPT,
        'tolerances': TASK_I_TOLERANCES,
    },
    'J': {
        'name': 'Weighted Aspect Priority',
        'ground_truth_class': TaskJGroundTruth,
        'compute_ground_truth': compute_task_j_ground_truth,
        'prompt': TASK_J_PROMPT,
        'tolerances': TASK_J_TOLERANCES,
    },
    'K': {
        'name': 'Entity Sentiment Drift',
        'ground_truth_class': TaskKGroundTruth,
        'compute_ground_truth': compute_task_k_ground_truth,
        'prompt': TASK_K_PROMPT,
        'tolerances': TASK_K_TOLERANCES,
    },
    'G1': {
        'name': 'Severe Allergy Safety',
        'ground_truth_class': TaskG1GroundTruth,
        'compute_ground_truth': compute_task_g1_ground_truth,
        'prompt': TASK_G1_PROMPT,
        'tolerances': TASK_G1_TOLERANCES,
        'scoring_fields': [
            'firsthand_severe_count',
            'safety_trajectory', 
            'false_assurance_count',
            'evidence_consensus',
            'final_risk_score'
        ],
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
