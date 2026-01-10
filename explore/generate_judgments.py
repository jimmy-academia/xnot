#!/usr/bin/env python3
"""
Generate LLM judgments for entire N=100 dataset.
Results are cached so this only runs once.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_judge import get_or_generate_judgment


def generate_all_judgments(dataset_path, max_count=100):
    """Generate judgments for all restaurants in dataset"""
    
    with open(dataset_path, 'r') as f:
        lines = list(f)[:max_count]
    
    print(f"Generating judgments for {len(lines)} restaurants...")
    print("="*70)
    
    results = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        restaurant_name = data.get('name', f'Unknown_{i}')
        
        print(f"\n[{i+1}/{len(lines)}] {restaurant_name[:40]}")
        
        judgment = get_or_generate_judgment(data, data.get('reviews', []))
        
        results.append({
            'restaurant': restaurant_name,
            'risk_score': judgment['final_risk_score'],
            'verdict': judgment['verdict'],
            'reasoning': judgment['reasoning']
        })
        
        print(f"  → Score: {judgment['final_risk_score']:.1f}, Verdict: {judgment['verdict']}")
    
    # Analyze distribution
    print("\n" + "="*70)
    print("FINAL DISTRIBUTION")
    print("="*70)
    
    low = sum(1 for r in results if r['risk_score'] < 4.0)
    high = sum(1 for r in results if 4.0 <= r['risk_score'] < 8.0)
    critical = sum(1 for r in results if r['risk_score'] >= 8.0)
    
    print(f"Low Risk (<4):      {low:3d} ({low/len(results)*100:5.1f}%)")
    print(f"High Risk (4-8):    {high:3d} ({high/len(results)*100:5.1f}%)")
    print(f"Critical (8+):      {critical:3d} ({critical/len(results)*100:5.1f}%)")
    print(f"\nMean: {sum(r['risk_score'] for r in results)/len(results):.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/packed/dataset_N100.jsonl')
    parser.add_argument('--max', type=int, default=100)
    args = parser.parse_args()
    
    generate_all_judgments(args.dataset, args.max)
