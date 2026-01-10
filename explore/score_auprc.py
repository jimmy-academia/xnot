#!/usr/bin/env python3
"""
AUPRC (Average Precision) Scoring for G1

Treats allergy risk assessment as a ranking problem.
- Positive class: High/Critical Risk (GT)
- Ranking signal: predicted final_risk_score
- Metric: Average Precision (AUPRC)
- Normalized AP: (AP - prevalence) / (1 - prevalence)

Random baseline ≈ prevalence (e.g., 0.20 if 20% positive)
Normalized random baseline ≈ 0.0
Perfect ranking → nAP = 1.0
"""

import json
import numpy as np
from sklearn.metrics import average_precision_score
from typing import List, Tuple


def calculate_auprc(results_file) -> dict:
    """
    Calculate AUPRC from benchmark results.
    
    Uses predicted final_risk_score to rank restaurants,
    treats High/Critical as positive class.
    """
    data = json.load(open(results_file))
    
    # Extract predictions and ground truth
    y_true = []  # Binary: 1 if High/Critical, 0 if Low
    y_scores = []  # Predicted risk scores (for ranking)
    
    for run in data['runs']:
        if 'results' not in run or run.get('restaurant') == 'validation_restaurant':
            continue
        
        # Ground truth
        gt_verdict = run['results']['verdict']['expected']
        y_true.append(1 if gt_verdict in ['High Risk', 'Critical Risk'] else 0)
        
        # Predicted score (for ranking)
        pred_score = run['results']['final_risk_score']['predicted']
        if pred_score is None:
            pred_score = 0.0
        y_scores.append(float(pred_score))
   
    # Calculate metrics
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    prevalence = y_true.mean()
    
    # Average Precision (AUPRC)
    ap = average_precision_score(y_true, y_scores)
    
    # Normalized AP (random = 0, perfect = 1)
    nap = (ap - prevalence) / (1 - prevalence) if prevalence < 1.0 else 0.0
    
    # Also calculate Positive-class F1 for comparison
    # Use median predicted score as threshold
    threshold = np.median(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("="*70)
    print("AUPRC SCORING (Ranking-Based Evaluation)")
    print("="*70)
    print()
    print(f"Dataset Composition:")
    print(f"  Positive (High/Critical): {y_true.sum():3d} ({prevalence*100:5.1f}%)")
    print(f"  Negative (Low):           {(~y_true.astype(bool)).sum():3d} ({(1-prevalence)*100:5.1f}%)")
    print(f"  Total:                    {len(y_true):3d}")
    print()
    print(f"Ranking Quality:")
    print(f"  Average Precision (AP):   {ap:.3f}")
    print(f"  Random Baseline:          {prevalence:.3f} (prevalence)")
    print(f"  Normalized AP (nAP):      {nap:.3f} ← PRIMARY METRIC")
    print(f"    (0 = random ranking, 1 = perfect)")
    print()
    print(f"Binary Classification (at median threshold={threshold:.1f}):")
    print(f"  Positive-class F1:        {f1_pos:.3f}")
    print(f"  Precision:                {precision:.3f}")
    print(f"  Recall:                   {recall:.3f}")
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"• AUPRC measures ranking quality for detecting High/Critical cases")
    print(f"• Normalized AP (nAP) = {nap:.3f}")
    print(f"  - nAP ≈ 0: Random ranking (baseline)")
    print(f"  - nAP ≈ 1: Perfect ranking (all High/Critical at top)")
    print(f"• Target: nAP < 0.15 for difficult task")
    print("="*70)
    
    return {
        'ap': ap,
        'nap': nap,
        'prevalence': prevalence,
        'f1_pos': f1_pos,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python score_auprc.py <detailed_results.json>")
        sys.exit(1)
    
    calculate_auprc(sys.argv[1])
