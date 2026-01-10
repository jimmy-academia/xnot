#!/usr/bin/env python3
"""
Calculate established classification metrics on benchmark results.

Metrics:
- Macro F1 (verdict-level)
- Matthews Correlation Coefficient (MCC)
- Per-Class Strict Scoring (reasoning-level)
"""

import json
import sys
import math


def calculate_metrics(results_file):
    data = json.load(open(results_file))
    
    # For verdict-level metrics
    tp = fp = fn = tn = 0
    
    # For reasoning-level metrics (per-class strict scores)
    positive_class_scores = []  # GT = High/Critical
    negative_class_scores = []  # GT = Low
    
    for run in data['runs']:
        if 'results' not in run:
            continue
            
        total_score = run['results']['_total_score']
        gt_verdict = run['results']['verdict']['expected']
        pred_verdict = run['results']['verdict']['predicted']
        
        # Classify as binary: Positive = High/Critical, Negative = Low
        gt_pos = gt_verdict in ["High Risk", "Critical Risk"]
        pred_pos = pred_verdict in ["High Risk", "Critical Risk"]
        
        # Confusion matrix (verdict-level)
        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
        
        # Per-class strict scores (reasoning-level)
        if gt_pos:
            positive_class_scores.append(total_score)
        else:
            negative_class_scores.append(total_score)
    
    # ===== VERDICT-LEVEL METRICS (Established) =====
    
    # Precision, Recall, F1 for positive class
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
    
    # Precision, Recall, F1 for negative class
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
    
    # Macro F1 (established - treats both classes equally)
    macro_f1 = (f1_pos + f1_neg) / 2
    
    # Matthews Correlation Coefficient (established)
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0
    
    # Balanced Accuracy (established)
    balanced_acc = (recall_pos + recall_neg) / 2
    
    # ===== REASONING-LEVEL METRICS (Established per-class scoring) =====
    
    positive_class_mean = sum(positive_class_scores) / len(positive_class_scores) if positive_class_scores else 0
    negative_class_mean = sum(negative_class_scores) / len(negative_class_scores) if negative_class_scores else 0
    macro_strict = (positive_class_mean + negative_class_mean) / 2
    
    # Print results
    print(f"\n{'='*60}")
    print(f"ESTABLISHED METRICS")
    print(f"{'='*60}")
    
    print(f"\n1. VERDICT-LEVEL (Classification Metrics)")
    print(f"   ----------------------------------------")
    print(f"   Confusion Matrix:")
    print(f"     TP: {tp}  FP: {fp}")
    print(f"     FN: {fn}  TN: {tn}")
    print(f"\n   Macro F1:            {macro_f1:.3f} (established - equal class weight)")
    print(f"   MCC:                 {mcc:.3f} (established - all quadrants)")
    print(f"   Balanced Accuracy:   {balanced_acc:.3f} (established - equal recall)")
    
    print(f"\n2. REASONING-LEVEL (Per-Class Strict Scores)")
    print(f"   ----------------------------------------")
    print(f"   Positive Class (High/Critical Risk GT):")
    print(f"     N: {len(positive_class_scores)}")
    print(f"     Mean Strict Score: {positive_class_mean:.3f} ← Key metric for difficulty")
    print(f"\n   Negative Class (Low Risk GT):")
    print(f"     N: {len(negative_class_scores)}")
    print(f"     Mean Strict Score: {negative_class_mean:.3f}")
    print(f"\n   Macro-Averaged Strict: {macro_strict:.3f} (equal class weight)")
    
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    print(f"• Verdict-Level Metrics: Measure if verdicts are correct")
    print(f"• Reasoning-Level Metrics: Measure if REASONING is correct")
    print(f"• Positive Class Strict Score {positive_class_mean:.3f}: True difficulty")
    print(f"  (This is what you want < 0.15)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rescore_established.py <detailed_results.json>")
        sys.exit(1)
    
    calculate_metrics(sys.argv[1])
