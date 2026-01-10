#!/usr/bin/env python3
"""
G1.1: Peanut Allergy Safety Task

Semantic reasoning task for assessing restaurant safety for severe peanut allergies.
Uses LLM-generated ground truth for primitives that cannot be computed via keyword matching.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TaskG1GroundTruth:
    """Ground truth for peanut allergy safety assessment."""
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
    """Load pre-computed semantic ground truth from JSON file."""
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
    G1.1: Peanut Allergy Safety (Semantic Reasoning).

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


# Task Registry (G1 only for now)
TASK_REGISTRY = {
    'G1': {
        'name': 'Peanut Allergy Safety',
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
    """Get task configuration by ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    """List all available task IDs."""
    return list(TASK_REGISTRY.keys())
