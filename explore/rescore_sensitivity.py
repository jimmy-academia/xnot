#!/usr/bin/env python3
"""
Re-score with different scoring_fields configurations.
"""

import json
import sys


def score_field(predicted, expected, tolerance=0.05):
    """Score a single field (from eval.py)"""
    if isinstance(expected, bool):
        return 1.0 if predicted == expected else 0.0
    elif isinstance(expected, str):
        if predicted is None:
            return 0.0
        pred_stripped = str(predicted).strip().strip('"').strip("'")
        exp_stripped = str(expected).strip()
        return 1.0 if pred_stripped == exp_stripped else 0.0
    elif isinstance(expected, (int, float)):
        if predicted is None:
            return 0.0
        try:
            pred_val = float(predicted)
            diff = abs(pred_val - expected)
            return 1.0 if diff <= tolerance else max(0.0, 1.0 - (diff / tolerance))
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def rescore_with_fields(results_file, scoring_fields):
    """Re-score using specific scoring fields"""
    data = json.load(open(results_file))
    
    tolerances = {
        'review_density': 0.01, 'star_variance': 0.3, 'date_decay': 0.2,
        'threat_index': 1.0, 'safety_buffer': 1.0, 'volatility': 1.0,
        'hygiene_penalty': 1.0, 'recency_factor': 0.1,
        'star_severity_correlation': 0.15, 'evidence_quality_score': 0.1,
        'temporal_trend': 0.2, 'cuisine_modifier': 0.5,
        'baseline_uncertainty': 0.3, 'compound_risk_index': 1.5,
        'safety_margin': 1.0, 'final_risk_score': 1.0
    }
    
    positive_scores = []
    
    for run in data['runs']:
        if 'results' not in run:
            continue
        
        results = run['results']
        gt_verdict = results['verdict']['expected']
        
        # Only evaluate on positive class
        if gt_verdict not in ['High Risk', 'Critical Risk']:
            continue
        
        # Recalculate score with new scoring_fields
        verdict_score = results['verdict']['score']
        
        if verdict_score == 0.0:
            new_score = 0.0
        else:
            # Calculate scores for specified fields only
            field_scores = []
            for field in scoring_fields:
                if field in results and field != 'verdict':
                    expected = results[field]['expected']
                    predicted = results[field]['predicted']
                    tolerance = tolerances.get(field, 0.05)
                    field_score = score_field(predicted, expected, tolerance)
                    field_scores.append(field_score)
            
            if field_scores:
                avg_primitives = sum(field_scores) / len(field_scores)
                new_score = verdict_score * avg_primitives
            else:
                new_score = verdict_score
        
        positive_scores.append(new_score)
    
    mean_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    
    # Distribution
    bins = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
    for s in positive_scores:
        if s < 0.2: bins['0.0-0.2'] += 1
        elif s < 0.4: bins['0.2-0.4'] += 1
        elif s < 0.6: bins['0.4-0.6'] += 1
        elif s < 0.8: bins['0.6-0.8'] += 1
        else: bins['0.8-1.0'] += 1
    
    return mean_score, bins, len(positive_scores)


if __name__ == '__main__':
    results_file = sys.argv[1]
    
    # Test different configurations
    configs = {
        'Current (16 fields)': [
            'review_density', 'star_variance', 'date_decay',
            'threat_index', 'safety_buffer', 'volatility', 'hygiene_penalty',
            'recency_factor', 'star_severity_correlation', 'evidence_quality_score',
            'temporal_trend', 'cuisine_modifier',
            'baseline_uncertainty', 'compound_risk_index', 'safety_margin',
            'final_risk_score'
        ],
        'Top 7 (L4+L5+L6)': [
            'recency_factor', 'star_severity_correlation', 'evidence_quality_score',
            'temporal_trend', 'cuisine_modifier',
            'baseline_uncertainty', 'compound_risk_index', 'safety_margin',
            'final_risk_score'
        ],
        'Top 5 (L5+L6+best)': [
            'baseline_uncertainty', 'compound_risk_index', 'safety_margin',
            'temporal_trend', 'final_risk_score'
        ],
        'Top 4 (L5+L6)': [
            'baseline_uncertainty', 'compound_risk_index', 'safety_margin',
            'final_risk_score'
        ],
    }
    
    print("\n" + "="*70)
    print("SCORING FIELD SENSITIVITY ANALYSIS (Positive Class Only)")
    print("="*70)
    
    for name, fields in configs.items():
        mean, bins, n = rescore_with_fields(results_file, fields)
        print(f"\n{name}:")
        print(f"  Scored Primitives: {len(fields)}")
        print(f"  Mean Score: {mean:.3f}")
        print(f"  Distribution (N={n}):")
        for bucket, count in bins.items():
            pct = count / n * 100 if n > 0 else 0
            bar = '█' * int(pct / 3)
            print(f"    {bucket}: {count:2d} ({pct:5.1f}%) {bar}")
    
    print("\n" + "="*70)
