#!/usr/bin/env python3
"""
Calculate Harmonic Mean of Per-Class Strict Scores.
Prevents gaming while allowing normal evaluation.
"""

import json
import sys


def calculate_harmonic_mean_score(results_file):
    """
    Harmonic mean of per-class strict scores.
    
    Formula:
    - low_mean = Avg(strict_score where GT=Low)
    - high_mean = Avg(strict_score where GT=High/Critical)
    - final = 2 * (low_mean * high_mean) / (low_mean + high_mean)
    
    Properties:
    - All-high gaming: low_mean=0 → final=0
    - All-low gaming: high_mean=0 → final=0
    - Balanced: Both classes contribute equally
    """
    data = json.load(open(results_file))
    
    low_scores = []
    high_scores = []
    
    for run in data['runs']:
        if 'results' not in run or run.get('restaurant') == 'validation_restaurant':
            continue
        
        total_score = run['results']['_total_score']
        gt_verdict = run['results']['verdict']['expected']
        
        if gt_verdict in ['High Risk', 'Critical Risk']:
            high_scores.append(total_score)
        else:
            low_scores.append(total_score)
    
    # Calculate means
    low_mean = sum(low_scores) / len(low_scores) if low_scores else 0
    high_mean = sum(high_scores) / len(high_scores) if high_scores else 0
    
    # Harmonic mean
    if low_mean + high_mean == 0:
        harmonic_mean = 0
    else:
        harmonic_mean = 2 * (low_mean * high_mean) / (low_mean + high_mean)
    
    # Also calculate other metrics for comparison
    global_mean = sum(low_scores + high_scores) / (len(low_scores) + len(high_scores))
    macro_mean = (low_mean + high_mean) / 2
    
    print("="*70)
    print("HARMONIC MEAN SCORING RESULTS")
    print("="*70)
    print()
    print(f"Dataset Composition:")
    print(f"  Low Risk cases:          {len(low_scores):2d}")
    print(f"  High/Critical Risk cases: {len(high_scores):2d}")
    print(f"  Total:                    {len(low_scores) + len(high_scores):2d}")
    print()
    print(f"Per-Class Strict Scores:")
    print(f"  Low Risk mean:           {low_mean:.3f}")
    print(f"  High/Critical Risk mean: {high_mean:.3f}")
    print()
    print(f"Aggregate Metrics:")
    print(f"  Global Mean (simple avg): {global_mean:.3f}")
    print(f"  Macro Mean (class-balanced): {macro_mean:.3f}")
    print(f"  HARMONIC MEAN (anti-gaming): {harmonic_mean:.3f} ← PRIMARY METRIC")
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"• Harmonic mean penalizes class imbalance")
    print(f"• Gaming strategies (all-high or all-low) → score = 0")
    print(f"• Current score: {harmonic_mean:.3f}")
    print(f"• Target: < 0.15")
    print("="*70)
    
    return {
        'low_mean': low_mean,
        'high_mean': high_mean,
        'harmonic_mean': harmonic_mean,
        'macro_mean': macro_mean,
        'global_mean': global_mean
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rescore_harmonic.py <detailed_results.json>")
        sys.exit(1)
    
    calculate_harmonic_mean_score(sys.argv[1])
