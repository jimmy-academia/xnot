#!/usr/bin/env python3
"""
AUPRC (Average Precision) Scoring for G1
"""

import json
import numpy as np
from sklearn.metrics import average_precision_score
from typing import List, Tuple


def calculate_auprc(results_file) -> dict:
    """
    Calculate AUPRC from benchmark results with 2-Tier Difficulty.
    
    1. Risk Detection (High+Critical vs Low): Primary difficulty metric (nAP)
    2. Critical Isolation (Critical vs Rest): Detecting the worst offenders
    3. Severity Ordering (Critical vs High): Ranking consistency at the top
    """
    try:
        data = json.load(open(results_file))
    except Exception as e:
        print(f"Error loading results file: {e}")
        return {}
    
    # Extract predictions and ground truth
    gt_verdicts = []
    y_scores = []  # Predicted risk scores (for ranking)
    
    for run in data['runs']:
        if 'results' not in run or run.get('restaurant') == 'validation_restaurant':
            continue
        
        # Ground truth
        gt_verdicts.append(run['results']['verdict']['expected'])
        
        # Predicted score (for ranking)
        pred_score = run['results']['final_risk_score']['predicted']
        if pred_score is None:
            pred_score = 0.0
        y_scores.append(float(pred_score))
   
    # Convert to numpy
    y_scores = np.array(y_scores)
    
    # --- METRIC 1: RISK DETECTION (Risky vs Safe) ---
    # Can the model separate High/Critical (Positive) from Low (Negative)?
    y_true_risky = np.array([1 if gt in ['High Risk', 'Critical Risk'] else 0 for gt in gt_verdicts])
    
    prevalence_risky = y_true_risky.mean()
    if prevalence_risky > 0 and prevalence_risky < 1:
        ap_risky = average_precision_score(y_true_risky, y_scores)
        nap_risky = (ap_risky - prevalence_risky) / (1 - prevalence_risky)
    else:
        ap_risky = 0.0
        nap_risky = 0.0

    # --- METRIC 2: CRITICAL ISOLATION (Critical vs Rest) ---
    # Can the model separate Critical (Positive) from High/Low (Negative)?
    y_true_critical = np.array([1 if gt == 'Critical Risk' else 0 for gt in gt_verdicts])
    
    prevalence_critical = y_true_critical.mean()
    if prevalence_critical > 0 and prevalence_critical < 1:
        ap_critical = average_precision_score(y_true_critical, y_scores)
        nap_critical = (ap_critical - prevalence_critical) / (1 - prevalence_critical)
    else:
        ap_critical = 0.0
        nap_critical = 0.0

    # --- METRIC 3: SEVERITY ORDERING (Critical > High) ---
    # Among ONLY the risky restaurants, does Critical rank higher than High?
    mask_severe = [gt in ['High Risk', 'Critical Risk'] for gt in gt_verdicts]
    mask_severe = np.array(mask_severe)
    
    if mask_severe.sum() > 1:
        y_scores_severe = y_scores[mask_severe]
        y_true_severe = np.array([1 if gt == 'Critical Risk' else 0 for gt in np.array(gt_verdicts)[mask_severe]])
        
        if y_true_severe.sum() > 0 and y_true_severe.sum() < len(y_true_severe):
            ap_severe_ordering = average_precision_score(y_true_severe, y_scores_severe)
        else:
            ap_severe_ordering = 0.0 # Sample too small or uniform class
    else:
        ap_severe_ordering = 0.0

    # Calculate Binary Stats for Risky Class (at median threshold)
    threshold = np.median(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = ((y_true_risky == 1) & (y_pred == 1)).sum()
    fp = ((y_true_risky == 0) & (y_pred == 1)).sum()
    fn = ((y_true_risky == 1) & (y_pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("="*70)
    print("AUPRC SCORING (2-Tier Ranking Difficulty)")
    print("="*70)
    print(f"Total Evaluated: {len(gt_verdicts)}")
    num_critical = y_true_critical.sum()
    num_high = y_true_risky.sum() - num_critical
    num_low = len(gt_verdicts) - y_true_risky.sum()
    print(f"Distribution: {num_critical} Critical, {num_high} High, {num_low} Low")
    print()
    print(f"1. RISK DETECTION (Risky vs Safe):")
    print(f"   AP_Risky:          {ap_risky:.3f} (Random={prevalence_risky:.3f})")
    print(f"   nAP_Risky:         {nap_risky:.3f}  ← PRIMARY DIFFICULTY METRIC")
    print()
    print(f"2. CRITICAL ISOLATION (Critical vs Rest):")
    print(f"   AP_Critical:       {ap_critical:.3f} (Random={prevalence_critical:.3f})")
    print(f"   nAP_Critical:      {nap_critical:.3f}")
    print()
    print(f"3. SEVERITY ORDERING (Critical ranked above High?):")
    print(f"   AP_Severity:       {ap_severe_ordering:.3f}")
    print()
    print("="*70)
    
    return {
        'nap_risky': nap_risky,
        'nap_critical': nap_critical,
        'ap_severe_ordering': ap_severe_ordering,
        'f1_risky': f1_pos
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python score_auprc.py <detailed_results.json>")
        sys.exit(1)
    
    calculate_auprc(sys.argv[1])
