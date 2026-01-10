#!/usr/bin/env python3
"""
explore/design_loop.py

Executes the SCALE Design Loop.
1. Loads a Task (e.g., G1.1).
2. Loads Data (e.g., dataset_N50.jsonl).
3. Computes Ground Truth (Python Logic).
4. Runs LLM (gpt-5-nano) asynchronously.
5. Compares and Generates Report (Float Scoring).
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path.cwd()))

from explore.tasks.g1 import G1_SevereAllergy
from utils.llm import call_llm_async

DATA_FILE = Path("explore/data/dataset_N50.jsonl")

async def process_context(task, context, sem):
    async with sem:
        # LLM Call
        prompt = task.format_prompt(context)
        try:
            # Enforce gpt-5-nano
            response_text = await call_llm_async(
                prompt,
                model="gpt-5-nano"
            )
            # Try parsing JSON
            try:
                # Clean up potential markdown blocks
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                response = json.loads(clean_text)
            except:
                # Fallback if model doesn't output JSON
                response = {"verdict": response_text, "reasoning": "Could not parse JSON"}
                
        except Exception as e:
            response = {"verdict": "ERROR", "reasoning": str(e)}
            
        # Score it
        score = task.score_response(context, response)
        
        return {
            "business_id": context['business_id'],
            "name": context['business']['name'],
            "gt_verdict": score['gt']['verdict'],
            "gt_reasoning": score['gt']['reasoning'],
            "llm_verdict": response.get('verdict', 'UNKNOWN'),
            "llm_reasoning": response.get('reasoning', ''),
            "score": score['score'],
            "verdict_score": score['verdict_score'],
            "primitive_score": score['primitive_score'],
            "details": score.get('match_details', {})
        }

async def main():
    # 1. Setup Task
    task = G1_SevereAllergy()
    print(f"--- Running Design Loop for Task: {task.name} ---")
    
    # 2. Load Data
    contexts = []
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found!")
        return

    with open(DATA_FILE) as f:
        for line in f:
            contexts.append(json.loads(line))
            
    print(f"Loaded {len(contexts)} contexts from {DATA_FILE}")
    
    # 3. Execution
    sem = asyncio.Semaphore(10) # 10 Concurrent requests
    tasks = [process_context(task, ctx, sem) for ctx in contexts]
    
    results = []
    
    # Run
    for i, f_result in enumerate(asyncio.as_completed(tasks)):
        res = await f_result
        results.append(res)
        print(f"Processed {i+1}/{len(contexts)}...", end='\r')
            
    # 4. Analysis
    # Averages
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_verdict = sum(r['verdict_score'] for r in results) / len(results)
    avg_prim = sum(r['primitive_score'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print(f"RESULTS REPORT: {task.name}")
    print(f"Average SCORE: {avg_score:.3f}")
    print(f"  - Verdict Match Rate: {avg_verdict:.1%}")
    print(f"  - Primitive Match Rate: {avg_prim:.1%}")
    print("="*60)
    
    # Show Failures
    failures = [r for r in results if r['score'] < 1.0]
    print(f"\nExample Failures ({len(failures)} total):")
    for f in failures[:5]:
        print(f"\n[FAIL] {f['name']} (Score: {f['score']})")
        print(f"  GT : {f['gt_verdict']}")
        print(f"  LLM: {f['llm_verdict']}")
        print(f"  Details: {f['details']}")
        
    # Save Report
    report_file = f"explore/report_{task.task_id}_N50_scored.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
