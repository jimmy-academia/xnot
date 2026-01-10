#!/usr/bin/env python3
"""
Deterministic Ground Truth Computation for G1.1 (Peanut Allergy Safety)

Computes GT values from stored per-review judgments using explicit formulas.
NO LLM calls - pure arithmetic on stored semantic judgments.

Usage:
    # Compute GT for one restaurant
    from g1_gt_compute import compute_gt_from_judgments
    gt = compute_gt_from_judgments("Restaurant Name")

    # Compute all and save
    python g1_gt_compute.py
    python g1_gt_compute.py --restaurant "Table 31"  # Single restaurant
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

DATA_DIR = Path(__file__).parent / "data"
JUDGMENTS_FILE = DATA_DIR / "semantic_gt" / "task_G1" / "judgments.json"
OUTPUT_FILE = DATA_DIR / "semantic_gt" / "task_G1" / "computed_gt.json"

# Cuisine risk modifiers (peanut/nut usage prevalence)
CUISINE_RISK_BASE = {
    "Thai": 2.0,
    "Vietnamese": 1.8,
    "Chinese": 1.5,
    "Asian Fusion": 1.5,
    "Indian": 1.3,
    "Japanese": 1.2,
    "Korean": 1.2,
    "Mexican": 1.0,
    "Italian": 0.5,
    "American (Traditional)": 0.5,
    "American (New)": 0.5,
    "Pizza": 0.5,
    "Sandwiches": 0.5,
    "Breakfast & Brunch": 0.6,
    "default": 1.0
}


@dataclass
class G1GroundTruth:
    """All GT primitives for G1.1 - deterministically computed."""
    # Layer 1 Aggregates
    n_mild: int
    n_moderate: int
    n_severe: int
    n_total_incidents: int
    n_positive: int
    n_negative: int
    n_betrayal: int

    # Layer 2 Derived
    incident_score: float
    safety_credit: float
    cuisine_modifier: float
    n_allergy_reviews: int
    review_density: float
    confidence_penalty: float
    most_recent_incident_year: int
    incident_age: int
    recency_decay: float
    total_incident_weight: float
    credibility_factor: float

    # Layer 3 Final
    incident_impact: float
    safety_impact: float
    cuisine_impact: float
    raw_risk: float
    final_risk_score: float
    verdict: str


def get_cuisine_modifier(categories: str) -> float:
    """Extract highest-risk cuisine modifier from category string."""
    cats = [c.strip() for c in categories.split(",")]
    max_risk = CUISINE_RISK_BASE["default"]
    for cat in cats:
        if cat in CUISINE_RISK_BASE:
            max_risk = max(max_risk, CUISINE_RISK_BASE[cat])
    return max_risk


def compute_gt_from_data(data: Dict) -> G1GroundTruth:
    """
    Compute all GT primitives from judgment data for one restaurant.
    This is DETERMINISTIC - no LLM calls, just arithmetic.

    Args:
        data: Restaurant judgment data with 'restaurant_meta' and 'reviews' keys

    Returns:
        G1GroundTruth dataclass with all computed primitives
    """
    reviews = data.get("reviews", [])
    categories = data.get("restaurant_meta", {}).get("categories", "")

    # === LAYER 1: Count from per-review judgments ===
    n_mild = n_moderate = n_severe = 0
    n_positive = n_negative = n_betrayal = 0
    n_allergy_reviews = 0
    incident_reviews = []  # For credibility calculation

    for r in reviews:
        # All judged reviews are allergy-relevant (they matched keywords)
        n_allergy_reviews += 1

        severity = r.get("incident_severity", "none")
        account = r.get("account_type", "none")
        interaction = r.get("safety_interaction", "none")

        # Count incidents (firsthand only)
        if account == "firsthand":
            if severity == "mild":
                n_mild += 1
                incident_reviews.append(r)
            elif severity == "moderate":
                n_moderate += 1
                incident_reviews.append(r)
            elif severity == "severe":
                n_severe += 1
                incident_reviews.append(r)

        # Count safety interactions
        if interaction == "positive":
            n_positive += 1
        elif interaction == "negative":
            n_negative += 1
        elif interaction == "betrayal":
            n_betrayal += 1

    n_total_incidents = n_mild + n_moderate + n_severe

    # === LAYER 2: Compute derived values ===

    # Step 2.3: Incident Score
    incident_score = float((n_mild * 2) + (n_moderate * 5) + (n_severe * 15))

    # Step 2.4: Safety Credit
    safety_credit = (n_positive * 1.0) - (n_negative * 0.5) - (n_betrayal * 5.0)

    # Step 2.5: Cuisine Modifier
    cuisine_modifier = get_cuisine_modifier(categories)

    # Step 2.6: Review Density & Confidence
    review_density = min(1.0, n_allergy_reviews / 10.0)
    confidence_penalty = 1.0 - (0.3 * (1 - review_density))

    # Step 2.7: Recency
    if incident_reviews:
        years = []
        for r in incident_reviews:
            date_str = r.get("date", "2020-01-01")
            try:
                year = int(date_str[:4])
            except (ValueError, TypeError):
                year = 2020
            years.append(year)
        most_recent_incident_year = max(years)
    else:
        most_recent_incident_year = 2020  # Default old

    incident_age = 2025 - most_recent_incident_year
    recency_decay = max(0.3, 1.0 - (incident_age * 0.15))

    # Step 2.8: Credibility Factor
    if incident_reviews:
        total_weight = 0.0
        for r in incident_reviews:
            stars = r.get("stars", 3)
            useful = r.get("useful", 0)
            weight = (5 - stars) + math.log(useful + 1)
            total_weight += weight
        credibility_factor = total_weight / max(n_total_incidents, 1)
    else:
        total_weight = 0.0
        credibility_factor = 1.0

    # === LAYER 3: Final Score ===
    BASE_RISK = 2.5

    incident_impact = incident_score * recency_decay * credibility_factor
    safety_impact = safety_credit * confidence_penalty
    cuisine_impact = cuisine_modifier * 0.5

    raw_risk = (BASE_RISK
                + incident_impact
                - safety_impact
                + cuisine_impact
                - (n_betrayal * 3.0))  # Extra penalty for false assurance

    final_risk_score = max(0.0, min(20.0, raw_risk))

    # Verdict
    if final_risk_score < 4.0:
        verdict = "Low Risk"
    elif final_risk_score < 8.0:
        verdict = "High Risk"
    else:
        verdict = "Critical Risk"

    return G1GroundTruth(
        n_mild=n_mild,
        n_moderate=n_moderate,
        n_severe=n_severe,
        n_total_incidents=n_total_incidents,
        n_positive=n_positive,
        n_negative=n_negative,
        n_betrayal=n_betrayal,
        incident_score=incident_score,
        safety_credit=safety_credit,
        cuisine_modifier=cuisine_modifier,
        n_allergy_reviews=n_allergy_reviews,
        review_density=review_density,
        confidence_penalty=round(confidence_penalty, 3),
        most_recent_incident_year=most_recent_incident_year,
        incident_age=incident_age,
        recency_decay=round(recency_decay, 3),
        total_incident_weight=round(total_weight, 3),
        credibility_factor=round(credibility_factor, 3),
        incident_impact=round(incident_impact, 3),
        safety_impact=round(safety_impact, 3),
        cuisine_impact=round(cuisine_impact, 3),
        raw_risk=round(raw_risk, 3),
        final_risk_score=round(final_risk_score, 2),
        verdict=verdict
    )


def load_judgments() -> Dict[str, Dict]:
    """Load all judgments from file."""
    if not JUDGMENTS_FILE.exists():
        raise FileNotFoundError(f"Judgments file not found: {JUDGMENTS_FILE}")

    with open(JUDGMENTS_FILE, 'r') as f:
        data = json.load(f)

    return data.get("judgments", {})


def compute_gt_from_judgments(restaurant_name: str) -> G1GroundTruth:
    """
    Compute GT for a single restaurant by name.

    Args:
        restaurant_name: Name of the restaurant

    Returns:
        G1GroundTruth dataclass
    """
    all_judgments = load_judgments()

    if restaurant_name not in all_judgments:
        raise ValueError(f"No judgments found for: {restaurant_name}")

    return compute_gt_from_data(all_judgments[restaurant_name])


def compute_all_gt() -> Dict[str, G1GroundTruth]:
    """Compute GT for all restaurants with stored judgments."""
    all_judgments = load_judgments()
    return {name: compute_gt_from_data(data) for name, data in all_judgments.items()}


def save_computed_gt():
    """Compute GT for all restaurants and save to file."""
    all_gt = compute_all_gt()

    output = {
        "task_id": "G1",
        "computed_at": __import__("datetime").datetime.now().isoformat(),
        "formula_version": "v1",
        "restaurants": {}
    }

    # Collect verdict distribution
    verdicts = {"Low Risk": 0, "High Risk": 0, "Critical Risk": 0}

    for name, gt in all_gt.items():
        output["restaurants"][name] = asdict(gt)
        verdicts[gt.verdict] += 1

    output["summary"] = {
        "total_restaurants": len(all_gt),
        "verdict_distribution": verdicts
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    return output


def print_gt(restaurant_name: str = None):
    """Print GT for one or all restaurants."""
    if restaurant_name:
        gt = compute_gt_from_judgments(restaurant_name)
        print(f"\n{'='*60}")
        print(f"G1.1 Ground Truth: {restaurant_name}")
        print(f"{'='*60}")
        for field, value in asdict(gt).items():
            print(f"  {field}: {value}")
    else:
        all_gt = compute_all_gt()
        verdicts = {"Low Risk": 0, "High Risk": 0, "Critical Risk": 0}
        for name, gt in all_gt.items():
            verdicts[gt.verdict] += 1
            print(f"{name}: {gt.verdict} (score={gt.final_risk_score})")
        print(f"\nDistribution: {verdicts}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute G1.1 Ground Truth")
    parser.add_argument("--restaurant", type=str, help="Compute for specific restaurant")
    parser.add_argument("--save", action="store_true", help="Save computed GT to file")
    args = parser.parse_args()

    if args.save:
        output = save_computed_gt()
        print(f"Saved GT for {output['summary']['total_restaurants']} restaurants")
        print(f"Distribution: {output['summary']['verdict_distribution']}")
        print(f"Output: {OUTPUT_FILE}")
    else:
        print_gt(args.restaurant)
