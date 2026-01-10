#!/usr/bin/env python3
"""
Cumulative Ordinal AUPRC Scoring for G1.1 (Peanut Allergy Safety)

See doc/ORDINAL_AUPRC.md for methodology and references.
"""

import json
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional


# Ordinal class mapping
CLASS_ORDER = {'Low Risk': 0, 'High Risk': 1, 'Critical Risk': 2}
CLASS_NAMES = ['Low Risk', 'High Risk', 'Critical Risk']


def load_ground_truth(gt_file: str) -> Dict[str, dict]:
    """Load semantic ground truth indexed by restaurant name."""
    with open(gt_file, 'r') as f:
        data = json.load(f)
    return {item['restaurant']: item for item in data}


def calculate_ordinal_auprc(
    y_true_ordinal: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, float]:
    """
    Calculate Cumulative Ordinal AUPRC.

    Args:
        y_true_ordinal: Ground truth ordinal labels (0=Low, 1=High, 2=Critical)
        y_scores: Predicted risk scores (higher = more risky)
        class_names: Names for each ordinal class

    Returns:
        Dictionary with all AUPRC metrics
    """
    n_classes = len(class_names)
    n_samples = len(y_true_ordinal)

    results = {
        'n_samples': n_samples,
        'distribution': {},
    }

    # Count class distribution
    for i, name in enumerate(class_names):
        count = (y_true_ordinal == i).sum()
        results['distribution'][name] = int(count)

    # === CUMULATIVE ORDINAL AUPRC ===
    # For each threshold k, compute AUPRC for P(Y >= k)

    cumulative_auprcs = []
    cumulative_naps = []  # Normalized (chance-corrected)

    for k in range(1, n_classes):  # k=1 (>=High), k=2 (>=Critical)
        # Binary: 1 if class >= k, else 0
        y_binary = (y_true_ordinal >= k).astype(int)
        prevalence = y_binary.mean()

        threshold_name = f">={class_names[k]}"

        if prevalence > 0 and prevalence < 1:
            ap = average_precision_score(y_binary, y_scores)
            # Normalized AP: (AP - random) / (1 - random)
            nap = (ap - prevalence) / (1 - prevalence)
        else:
            ap = 0.0
            nap = 0.0

        results[f'auprc_ge_{k}'] = ap
        results[f'nap_ge_{k}'] = nap
        results[f'prevalence_ge_{k}'] = prevalence
        results[f'threshold_{k}_name'] = threshold_name

        cumulative_auprcs.append(ap)
        cumulative_naps.append(nap)

    # Combined Ordinal AUPRC (mean of cumulative thresholds)
    results['ordinal_auprc'] = np.mean(cumulative_auprcs)
    results['ordinal_nap'] = np.mean(cumulative_naps)

    # === ADDITIONAL METRICS ===

    # Severity ordering: Among risky (High+Critical), is Critical ranked higher?
    mask_risky = y_true_ordinal >= 1
    if mask_risky.sum() > 1:
        y_risky = y_true_ordinal[mask_risky]
        scores_risky = y_scores[mask_risky]
        y_critical_among_risky = (y_risky == 2).astype(int)

        if y_critical_among_risky.sum() > 0 and y_critical_among_risky.sum() < len(y_critical_among_risky):
            results['severity_ordering_ap'] = average_precision_score(y_critical_among_risky, scores_risky)
        else:
            results['severity_ordering_ap'] = np.nan
    else:
        results['severity_ordering_ap'] = np.nan

    return results


def calculate_from_results_file(results_file: str, gt_file: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate ordinal AUPRC from a benchmark results file.

    Args:
        results_file: Path to detailed results JSON
        gt_file: Optional path to ground truth file (uses embedded GT if not provided)

    Returns:
        Dictionary with all metrics
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    y_true_ordinal = []
    y_scores = []
    restaurants = []

    for run in data.get('runs', []):
        if 'results' not in run:
            continue
        if run.get('restaurant') == 'validation_restaurant':
            continue

        # Ground truth verdict
        gt_verdict = run['results']['verdict']['expected']
        if gt_verdict not in CLASS_ORDER:
            continue

        # Predicted score
        pred_score = run['results']['final_risk_score']['predicted']
        if pred_score is None:
            pred_score = 0.0

        y_true_ordinal.append(CLASS_ORDER[gt_verdict])
        y_scores.append(float(pred_score))
        restaurants.append(run.get('restaurant', 'Unknown'))

    y_true_ordinal = np.array(y_true_ordinal)
    y_scores = np.array(y_scores)

    return calculate_ordinal_auprc(y_true_ordinal, y_scores)


def print_report(metrics: Dict[str, float]) -> None:
    """Print formatted AUPRC report."""
    print("=" * 70)
    print("CUMULATIVE ORDINAL AUPRC REPORT")
    print("=" * 70)
    print(f"Samples: {metrics['n_samples']}")
    print(f"Distribution: {metrics['distribution']}")
    print()

    print("CUMULATIVE THRESHOLDS:")
    print("-" * 50)

    # Threshold 1: >= High (High + Critical vs Low)
    print(f"  Threshold 1: {metrics.get('threshold_1_name', '>=High')}")
    print(f"    Prevalence:  {metrics['prevalence_ge_1']:.3f}")
    print(f"    AUPRC:       {metrics['auprc_ge_1']:.3f}")
    print(f"    nAP:         {metrics['nap_ge_1']:.3f}")
    print()

    # Threshold 2: >= Critical (Critical vs Rest)
    print(f"  Threshold 2: {metrics.get('threshold_2_name', '>=Critical')}")
    print(f"    Prevalence:  {metrics['prevalence_ge_2']:.3f}")
    print(f"    AUPRC:       {metrics['auprc_ge_2']:.3f}")
    print(f"    nAP:         {metrics['nap_ge_2']:.3f}")
    print()

    print("=" * 70)
    print("COMBINED ORDINAL METRICS:")
    print("-" * 50)
    print(f"  Ordinal AUPRC:  {metrics['ordinal_auprc']:.3f}  (mean of cumulative AUPRCs)")
    print(f"  Ordinal nAP:    {metrics['ordinal_nap']:.3f}  (chance-corrected)")

    if not np.isnan(metrics.get('severity_ordering_ap', np.nan)):
        print(f"  Severity Order: {metrics['severity_ordering_ap']:.3f}  (Critical > High among risky)")

    print("=" * 70)


def demo_with_synthetic_data():
    """Demo with synthetic predictions to show metric behavior."""
    print("\n" + "=" * 70)
    print("DEMO: Synthetic Data")
    print("=" * 70)

    # Load actual GT distribution
    import os
    gt_path = os.path.join(os.path.dirname(__file__), 'data', 'semantic_gt', 'all_judgments.json')

    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        y_true = np.array([CLASS_ORDER[item['verdict']] for item in gt_data])
        print(f"Loaded GT: {len(y_true)} restaurants")
        print(f"Distribution: Low={sum(y_true==0)}, High={sum(y_true==1)}, Critical={sum(y_true==2)}")

        # Scenario 1: Perfect predictions
        print("\n--- Scenario 1: Perfect Predictions ---")
        y_scores_perfect = y_true.astype(float) * 5 + np.random.normal(0, 0.1, len(y_true))
        metrics = calculate_ordinal_auprc(y_true, y_scores_perfect)
        print_report(metrics)

        # Scenario 2: Random predictions
        print("\n--- Scenario 2: Random Predictions ---")
        y_scores_random = np.random.uniform(0, 10, len(y_true))
        metrics = calculate_ordinal_auprc(y_true, y_scores_random)
        print_report(metrics)

        # Scenario 3: Inverted (worst case)
        print("\n--- Scenario 3: Inverted Predictions (Worst Case) ---")
        y_scores_inverted = (2 - y_true).astype(float) * 5 + np.random.normal(0, 0.1, len(y_true))
        metrics = calculate_ordinal_auprc(y_true, y_scores_inverted)
        print_report(metrics)
    else:
        print(f"GT file not found: {gt_path}")
        print("Run with a results file instead.")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python score_auprc.py <results_file.json>")
        print("       python score_auprc.py --demo")
        print()
        demo_with_synthetic_data()
    elif sys.argv[1] == '--demo':
        demo_with_synthetic_data()
    else:
        metrics = calculate_from_results_file(sys.argv[1])
        print_report(metrics)
