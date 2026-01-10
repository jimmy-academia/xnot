#!/usr/bin/env python3
"""
Re-score existing benchmark results using F2-score metric.

Usage:
    python explore/rescore_f2.py explore/results/run_20260110_022204/detailed_results.json
"""

import json
import sys
from pathlib import Path


def calculate_f2_score(results_file):
    """
    Calculate F2 score from detailed results.
    
    Treats "High Risk" and "Critical Risk" verdicts as positive class.
    """
    data = json.load(open(results_file))
    
    tp = 0  # True Positives: Correctly predicted High/Critical
    fp = 0  # False Positives: Predicted High/Critical, actually Low
    fn = 0  # False Negatives: Predicted Low, actually High/Critical
    tn = 0  # True Negatives: Correctly predicted Low
    
    for run in data['runs']:
        if 'results' not in run or 'verdict' not in run['results']:
            continue
            
        gt_verdict = run['results']['verdict']['expected']
        pred_verdict = run['results']['verdict']['predicted']
        
        # Define positive class as High or Critical Risk
        gt_positive = gt_verdict in ["High Risk", "Critical Risk"]
        pred_positive = pred_verdict in ["High Risk", "Critical Risk"]
        
        if gt_positive and pred_positive:
            tp += 1
        elif not gt_positive and pred_positive:
            fp += 1
        elif gt_positive and not pred_positive:
            fn += 1
        else:
            tn += 1
    
    # Calculate F2 score
    if tp == 0:
        f2_score = 0.0
        precision = 0.0
        recall = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        beta = 2.0
        if (precision + recall) == 0:
            f2_score = 0.0
        else:
            f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    total = tp + fp + fn + tn
    
    print(f"\nF2-Score Analysis:")
    print(f"==================")
    print(f"Total Restaurants: {total}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (High/Critical correct): {tp}")
    print(f"  False Positives (Low should be High):    {fp}")
    print(f"  False Negatives (High should be Low):    {fn}")
    print(f"  True Negatives  (Low correct):           {tn}")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F2-Score:  {f2_score:.3f}")
    print(f"\nInterpretation:")
    print(f"  F2 heavily penalizes false negatives (missing dangerous restaurants)")
    print(f"  Current FN rate: {fn/total*100:.1f}%")
    
    return {
        'f2_score': f2_score,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total': total
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rescore_f2.py <detailed_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    calculate_f2_score(results_file)
