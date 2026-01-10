#!/usr/bin/env python3
"""
Generate semantic judgments with human-reviewable output.
"""

import json
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent))
from semantic_judge import generate_semantic_judgment, save_for_human_review


def generate_semantic_judgments(dataset_path, sample_size=20, output_dir='data/human_review'):
    """
    Generate semantic ground truth for sample of restaurants.
    Output in human-reviewable format.
    """
    
    with open(dataset_path, 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    # Random sample for human validation
    sampled = random.sample(all_data, min(sample_size, len(all_data)))
    
    print(f"Generating semantic judgments for {len(sampled)} restaurants...")
    print("="*70)
    
    judgments = []
    for i, data in enumerate(sampled):
        # Extract business info from nested structure
        business = data.get('business', {})
        restaurant_name = business.get('name', f'Unknown_{i}')

        print(f"\n[{i+1}/{len(sampled)}] {restaurant_name[:40]}")

        # Pass business dict (not full data) to semantic judge
        judgment = generate_semantic_judgment(business, data.get('reviews', []))
        
        judgments.append({
            'restaurant': restaurant_name,
            **judgment
        })
        
        print(f"  → Risk: {judgment['final_risk_score']:.1f}, Verdict: {judgment['verdict']}")
        print(f"  → Firsthand: {judgment['firsthand_severe_count']}, False Assurance: {judgment['false_assurance_count']}")
    
    # Save for human review
    save_for_human_review(judgments, output_dir)
    
    # Analyze distribution
    print("\n" + "="*70)
    print("DISTRIBUTION")
    print("="*70)
    low = sum(1 for j in judgments if j['final_risk_score'] < 4.0)
    high = sum(1 for j in judgments if 4.0 <= j['final_risk_score'] < 8.0)
    critical = sum(1 for j in judgments if j['final_risk_score'] >= 8.0)
    
    print(f"Low Risk (<4):      {low:2d} ({low/len(judgments)*100:5.1f}%)")
    print(f"High Risk (4-8):    {high:2d} ({high/len(judgments)*100:5.1f}%)")
    print(f"Critical (8+):      {critical:2d} ({critical/len(judgments)*100:5.1f}%)")
    
    return judgments


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/dataset_K100.jsonl')
    parser.add_argument('--sample', type=int, default=20, help='Sample size for human validation')
    parser.add_argument('--output', default='data/human_review')
    args = parser.parse_args()
    
    generate_semantic_judgments(args.dataset, args.sample, args.output)
