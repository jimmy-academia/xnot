"""
Tools for G1a-ANoT phases.

These tools enable selective data access and computation:
- keyword_search: Find allergy-relevant reviews
- get_review_metadata: Get date, stars, useful for a review
- get_cuisine_modifier: Lookup cuisine risk from categories
"""

import re
import math
from typing import List, Dict, Any, Tuple


# Allergy-related keywords for filtering reviews
ALLERGY_KEYWORDS = [
    'allergy', 'allergic', 'allergies',
    'peanut', 'nut', 'nuts', 'tree nut',
    'epipen', 'epi-pen', 'epinephrine',
    'anaphyl', 'anaphylaxis', 'anaphylactic',
    'reaction', 'hives', 'swell', 'swelling', 'swollen',
    'gluten', 'celiac', 'coeliac',
    'dietary', 'restriction', 'accommodate', 'accommodation',
    'sensitive', 'sensitivity', 'intoleran', 'intolerance',
]

# Cuisine risk modifiers (higher = more nut usage)
CUISINE_RISK_BASE = {
    'Thai': 2.0,
    'Vietnamese': 1.8,
    'Chinese': 1.5,
    'Asian': 1.5,
    'Asian Fusion': 1.5,
    'Indian': 1.3,
    'Japanese': 1.2,
    'Korean': 1.2,
    'Mexican': 1.0,
    'Italian': 0.5,
    'American': 0.5,
    'Pizza': 0.5,
    'Sandwiches': 0.5,
    'default': 1.0
}


def keyword_search(reviews: List[Dict], keywords: List[str] = None) -> List[int]:
    """
    Find review indices containing allergy-related keywords.

    Args:
        reviews: List of review dicts with 'text' field
        keywords: Optional custom keywords (defaults to ALLERGY_KEYWORDS)

    Returns:
        List of review indices (0-indexed) containing keywords
    """
    if keywords is None:
        keywords = ALLERGY_KEYWORDS

    # Compile pattern for efficiency
    pattern = re.compile('|'.join(re.escape(k) for k in keywords), re.IGNORECASE)

    relevant = []
    for idx, review in enumerate(reviews):
        text = review.get('text', '')
        if pattern.search(text):
            relevant.append(idx)

    return relevant


def get_review_metadata(review: Dict) -> Dict[str, Any]:
    """
    Extract metadata from a review for computation.

    Args:
        review: Review dict

    Returns:
        Dict with date, year, stars, useful
    """
    date = review.get('date', '2020-01-01')
    try:
        year = int(date[:4])
    except (ValueError, TypeError):
        year = 2020

    return {
        'date': date,
        'year': year,
        'stars': review.get('stars', 3),
        'useful': review.get('useful', 0),
    }


def get_cuisine_modifier(categories: str) -> float:
    """
    Get cuisine risk modifier from restaurant categories.

    Args:
        categories: Comma-separated category string

    Returns:
        Highest matching cuisine risk modifier
    """
    if not categories:
        return CUISINE_RISK_BASE['default']

    cats = [c.strip() for c in categories.split(',')]
    max_risk = CUISINE_RISK_BASE['default']

    for cat in cats:
        for cuisine, risk in CUISINE_RISK_BASE.items():
            if cuisine.lower() in cat.lower():
                max_risk = max(max_risk, risk)

    return max_risk


def compute_incident_weight(stars: int, useful: int) -> float:
    """
    Compute credibility weight for an incident review.

    Formula: (5 - stars) + log(useful + 1)

    Args:
        stars: Review star rating (1-5)
        useful: Useful vote count

    Returns:
        Weight value
    """
    return (5 - stars) + math.log(useful + 1)


def compute_recency_decay(incident_year: int, current_year: int = 2025) -> float:
    """
    Compute recency decay factor for incident.

    Formula: max(0.3, 1.0 - (age * 0.15))

    Args:
        incident_year: Year of most recent incident
        current_year: Current year (default 2025)

    Returns:
        Decay factor (0.3 to 1.0)
    """
    age = current_year - incident_year
    return max(0.3, 1.0 - (age * 0.15))


def compute_risk_score(
    n_mild: int,
    n_moderate: int,
    n_severe: int,
    n_positive: int,
    n_negative: int,
    n_betrayal: int,
    recency_decay: float,
    credibility_factor: float,
    cuisine_modifier: float,
    n_allergy_reviews: int
) -> Tuple[float, str]:
    """
    Compute final risk score and verdict using V1 formula.

    Returns:
        (final_risk_score, verdict)
    """
    # Layer 2: Derived values
    incident_score = (n_mild * 2) + (n_moderate * 5) + (n_severe * 15)
    safety_credit = (n_positive * 1.0) - (n_negative * 0.5) - (n_betrayal * 5.0)

    review_density = min(1.0, n_allergy_reviews / 10.0)
    confidence_penalty = 1.0 - (0.3 * (1 - review_density))

    # Layer 3: Final score
    BASE_RISK = 2.5

    incident_impact = incident_score * recency_decay * credibility_factor
    safety_impact = safety_credit * confidence_penalty
    cuisine_impact = cuisine_modifier * 0.5

    raw_risk = (BASE_RISK
                + incident_impact
                - safety_impact
                + cuisine_impact
                - (n_betrayal * 3.0))

    final_risk_score = max(0.0, min(20.0, raw_risk))

    # Verdict
    if final_risk_score < 4.0:
        verdict = "Low Risk"
    elif final_risk_score < 8.0:
        verdict = "High Risk"
    else:
        verdict = "Critical Risk"

    return round(final_risk_score, 2), verdict


def format_review_preview(review: Dict, max_length: int = 200) -> str:
    """
    Get a preview of review text for LLM extraction.

    Args:
        review: Review dict
        max_length: Max characters to include

    Returns:
        Truncated review text
    """
    text = review.get('text', '')
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text


def prepare_review_data(reviews: List[Dict], indices: List[int]) -> List[Dict]:
    """
    Prepare review data for Phase 2 prompt.

    Args:
        reviews: All reviews
        indices: Indices of relevant reviews

    Returns:
        List of review data dicts with preview and metadata
    """
    result = []
    for idx in indices:
        if idx < len(reviews):
            review = reviews[idx]
            meta = get_review_metadata(review)
            result.append({
                'idx': idx,
                'text_preview': format_review_preview(review),
                **meta
            })
    return result
