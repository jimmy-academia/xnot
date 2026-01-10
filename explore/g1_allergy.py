#!/usr/bin/env python3
"""
G1.1: Peanut Allergy Safety Task

Semantic reasoning task for assessing restaurant safety for severe peanut allergies.
Uses deterministically computed GT from stored per-review LLM judgments.

GT Pipeline:
1. Keyword filtering identifies allergy-relevant reviews
2. LLM extracts per-review signals (incident_severity, account_type, safety_interaction)
3. Python formulas compute all aggregates deterministically
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class TaskG1GroundTruth:
    """Ground truth for peanut allergy safety assessment.

    Primitives computed via explicit formulas from per-review semantic judgments.
    """
    # Layer 1: Incident Counts (from firsthand accounts only)
    n_total_incidents: int  # n_mild + n_moderate + n_severe

    # Layer 2: Derived Metrics
    incident_score: float  # (mild*2 + moderate*5 + severe*15)
    recency_decay: float  # 0.3-1.0 based on incident age
    credibility_factor: float  # Weight based on stars + useful votes

    # Layer 3: Final Outputs
    final_risk_score: float  # 0-20, clamped
    verdict: str  # "Low Risk", "High Risk", "Critical Risk"


# Global cache for computed GT
G1_COMPUTED_GT_CACHE = None


def load_g1_computed_gt():
    """Load pre-computed ground truth from deterministic computation."""
    global G1_COMPUTED_GT_CACHE
    if G1_COMPUTED_GT_CACHE is not None:
        return G1_COMPUTED_GT_CACHE

    try:
        gt_path = Path(__file__).parent / 'data' / 'semantic_gt' / 'task_G1' / 'computed_gt.json'
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                data = json.load(f)
                G1_COMPUTED_GT_CACHE = data.get('restaurants', {})
        else:
            print(f"WARNING: Computed GT file not found at {gt_path}")
            G1_COMPUTED_GT_CACHE = {}
    except Exception as e:
        print(f"ERROR loading computed GT: {e}")
        G1_COMPUTED_GT_CACHE = {}

    return G1_COMPUTED_GT_CACHE


def compute_task_g1_ground_truth(reviews: List[Any], restaurant: Any) -> TaskG1GroundTruth:
    """
    Load G1.1 ground truth from pre-computed deterministic values.

    The GT is computed from stored per-review LLM judgments via explicit formulas.
    See g1_gt_compute.py for the full computation logic.
    """
    gt_cache = load_g1_computed_gt()
    res_name = restaurant.get('name', 'Unknown')

    if res_name in gt_cache:
        gt = gt_cache[res_name]
        return TaskG1GroundTruth(
            n_total_incidents=int(gt.get('n_total_incidents', 0)),
            incident_score=float(gt.get('incident_score', 0)),
            recency_decay=float(gt.get('recency_decay', 1.0)),
            credibility_factor=float(gt.get('credibility_factor', 1.0)),
            final_risk_score=float(gt.get('final_risk_score', 2.5)),
            verdict=str(gt.get('verdict', 'Low Risk'))
        )

    # Fallback for unknown restaurants
    return TaskG1GroundTruth(
        n_total_incidents=0,
        incident_score=0.0,
        recency_decay=1.0,
        credibility_factor=1.0,
        final_risk_score=2.75,  # BASE_RISK + CUISINE_IMPACT default
        verdict="Low Risk"
    )


TASK_G1_PROMPT = """Analyze the reviews for PEANUT/NUT ALLERGY SAFETY using the exact formulas below.

## STEP 1: Per-Review Semantic Extraction

For each review mentioning allergies, extract:

1. INCIDENT_SEVERITY: "none" | "mild" | "moderate" | "severe"
   - none: No allergic reaction described
   - mild: Minor symptoms (stomach upset, mild discomfort)
   - moderate: Visible symptoms (hives, swelling, needed medication)
   - severe: Life-threatening (anaphylaxis, EpiPen, ER visit, hospitalization)

2. ACCOUNT_TYPE: "none" | "firsthand" | "secondhand" | "hypothetical"
   - none: No incident
   - firsthand: Personal experience ("I had", "my child", "we experienced")
   - secondhand: Reported ("I heard", "friend told me")
   - hypothetical: Concern without incident

3. SAFETY_INTERACTION: "none" | "positive" | "negative" | "betrayal"
   - none: No staff interaction about allergies
   - positive: Staff asked about allergies AND successfully accommodated
   - negative: Staff dismissive or refused to accommodate
   - betrayal: Staff CLAIMED safe BUT customer still had a reaction

## STEP 2: Aggregate Counts (Firsthand Incidents Only)

N_MILD = Count reviews where (INCIDENT_SEVERITY="mild" AND ACCOUNT_TYPE="firsthand")
N_MODERATE = Count reviews where (INCIDENT_SEVERITY="moderate" AND ACCOUNT_TYPE="firsthand")
N_SEVERE = Count reviews where (INCIDENT_SEVERITY="severe" AND ACCOUNT_TYPE="firsthand")
N_TOTAL_INCIDENTS = N_MILD + N_MODERATE + N_SEVERE

N_POSITIVE = Count reviews where SAFETY_INTERACTION="positive"
N_NEGATIVE = Count reviews where SAFETY_INTERACTION="negative"
N_BETRAYAL = Count reviews where SAFETY_INTERACTION="betrayal"

## STEP 3: Calculate Derived Values

INCIDENT_SCORE = (N_MILD * 2) + (N_MODERATE * 5) + (N_SEVERE * 15)

SAFETY_CREDIT = (N_POSITIVE * 1.0) - (N_NEGATIVE * 0.5) - (N_BETRAYAL * 5.0)

CUISINE_MODIFIER = lookup restaurant categories:
  Thai=2.0, Vietnamese=1.8, Chinese=1.5, Asian=1.5,
  Indian=1.3, Japanese=1.2, Korean=1.2, Mexican=1.0,
  Italian=0.5, American=0.5, Pizza=0.5, default=1.0

N_ALLERGY_REVIEWS = Count reviews mentioning allergy-related terms
REVIEW_DENSITY = min(1.0, N_ALLERGY_REVIEWS / 10.0)
CONFIDENCE_PENALTY = 1.0 - (0.3 * (1 - REVIEW_DENSITY))

MOST_RECENT_INCIDENT_YEAR = max(year) from incident reviews (default: 2020)
INCIDENT_AGE = 2025 - MOST_RECENT_INCIDENT_YEAR
RECENCY_DECAY = max(0.3, 1.0 - (INCIDENT_AGE * 0.15))

For each incident review, calculate:
  WEIGHT = (5 - stars) + log(useful + 1)
TOTAL_INCIDENT_WEIGHT = sum(WEIGHT for incident reviews)
CREDIBILITY_FACTOR = TOTAL_INCIDENT_WEIGHT / max(N_TOTAL_INCIDENTS, 1)
  (default: 1.0 if no incidents)

## STEP 4: Final Score Calculation

BASE_RISK = 2.5

INCIDENT_IMPACT = INCIDENT_SCORE * RECENCY_DECAY * CREDIBILITY_FACTOR
SAFETY_IMPACT = SAFETY_CREDIT * CONFIDENCE_PENALTY
CUISINE_IMPACT = CUISINE_MODIFIER * 0.5

RAW_RISK = BASE_RISK + INCIDENT_IMPACT - SAFETY_IMPACT + CUISINE_IMPACT - (N_BETRAYAL * 3.0)

FINAL_RISK_SCORE = max(0.0, min(20.0, RAW_RISK))

VERDICT:
  If FINAL_RISK_SCORE < 4.0: "Low Risk"
  If 4.0 <= FINAL_RISK_SCORE < 8.0: "High Risk"
  If FINAL_RISK_SCORE >= 8.0: "Critical Risk"

## OUTPUT

Report all intermediate values and final answers."""


TASK_G1_TOLERANCES = {
    'n_total_incidents': 0,  # Exact count
    'incident_score': 0,  # Exact calculation
    'recency_decay': 0.1,  # Float tolerance
    'credibility_factor': 0.2,  # Float tolerance
    'final_risk_score': 1.5,  # Score tolerance
}


# Task Registry (G1 only for now)
TASK_REGISTRY = {
    'G1': {
        'name': 'Peanut Allergy Safety',
        'ground_truth_class': TaskG1GroundTruth,
        'compute_ground_truth': compute_task_g1_ground_truth,
        'prompt': TASK_G1_PROMPT,
        'tolerances': TASK_G1_TOLERANCES,
        'scoring_fields': [
            'n_total_incidents',
            'incident_score',
            'recency_decay',
            'credibility_factor',
            'final_risk_score'
        ],
    },
}


def get_task(task_id: str) -> dict:
    """Get task configuration by ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    """List all available task IDs."""
    return list(TASK_REGISTRY.keys())
