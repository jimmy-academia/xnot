#!/usr/bin/env python3
"""
Re-score benchmark results using FILTERED strict scoring.
Only evaluates on "challenging" cases (High/Critical Risk ground truth).

Usage:
    python explore/rescore_filtered.py explore/results/run_20260110_022204/detailed_results.json
"""

import json
import sys
from pathlib import Path


def calculate_filtered_score(results_file):
    """
    Calculate strict scores only on High/Critical Risk cases.
    """
    data = json.load(open(results_file))
    
    all_scores = []
    high_risk_scores = []
    low_risk_scores = []
    
    for run in data['runs']:
        if 'results' not in run or '_total_score' not in run['results']:
            continue
        
        restaurant = run.get('restaurant', 'Unknown')
        total_score = run['results']['_total_score']
        gt_verdict = run['results'].get('verdict', {}).get('expected', 'Unknown')
        pred_verdict = run['results'].get('verdict', {}).get('predicted', 'Unknown')
        
        all_scores.append({
            'restaurant': restaurant,
            'score': total_score,
            'gt_verdict': gt_verdict,
            'pred_verdict': pred_verdict
        })
        
        if gt_verdict in ["High Risk", "Critical Risk"]:
            high_risk_scores.append(total_score)
        else:
            low_risk_scores.append(total_score)
    
    # Calculate metrics
    global_mean = sum(s['score'] for s in all_scores) / len(all_scores) if all_scores else 0
    high_risk_mean = sum(high_risk_scores) / len(high_risk_scores) if high_risk_scores else 0
    low_risk_mean = sum(low_risk_scores) / len(low_risk_scores) if low_risk_scores else 0
    
    print(f"\nFiltered Scoring Analysis:")
    print(f"==========================")
    print(f"Total Restaurants: {len(all_scores)}")
    print(f"\nBreakdown by Ground Truth Difficulty:")
    print(f"  High/Critical Risk (challenging): {len(high_risk_scores)} restaurants")
    print(f"  Low Risk (easy):                  {len(low_risk_scores)} restaurants")
    print(f"\nStrict Scores (Verdict × Avg(Primitives)):")
    print(f"  All Cases:            {global_mean:.3f}")
    print(f"  High/Critical ONLY:   {high_risk_mean:.3f} ← This is the 'hard' difficulty")
    print(f"  Low Risk ONLY:        {low_risk_mean:.3f} ← This is the 'easy' difficulty")
    print(f"\nInterpretation:")
    print(f"  The {high_risk_mean:.3f} score on challenging cases shows true reasoning difficulty.")
    print(f"  Target: <0.15 for sufficient difficulty.")
    
    # Show some examples of failures on high-risk cases
    print(f"\nHigh/Critical Risk Cases (sorted by score):")
    high_risk_cases = [s for s in all_scores if s['gt_verdict'] in ["High Risk", "Critical Risk"]]
    high_risk_cases.sort(key=lambda x: x['score'])
    
    for i, case in enumerate(high_risk_cases[:10]):  # Show bottom 10
        print(f"  {case['restaurant']}: {case['score']:.3f} (GT: {case['gt_verdict']}, Pred: {case['pred_verdict']})")
    
    return {
        'all_mean': global_mean,
        'high_risk_mean': high_risk_mean,
        'low_risk_mean': low_risk_mean,
        'n_high_risk': len(high_risk_scores),
        'n_low_risk': len(low_risk_scores)
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rescore_filtered.py <detailed_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    calculate_filtered_score(results_file)
