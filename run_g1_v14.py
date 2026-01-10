#!/usr/bin/env python3
"""
Run G1.v14 Benchmark (Severe Allergy Safety)
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def run_benchmark():
    print("="*80)
    print("RUNNING G1.v14 BENCHMARK (Semantic Judge + 2-Tier Ranking)")
    print("="*80)
    
    # 1. Run Evaluation
    cmd = [
        "python", "main.py",
        "--method", "anot",
        "--candidates", "5",  # Run on quick sample (matching current GT)
        "--dev",
        "--task", "G1",  # NOTE: main.py needs to support this or run all
        "--run", "1"
    ]
    
    # Check if main.py supports --task. If not, we rely on it running all tasks
    # and we filter in scoring.
    # Actually, main.py -> run_single -> evaluate_task
    # The current main.py doesn't seem to have --task arg based on my read.
    # It runs scaling_experiment or run_single which runs 'run_evaluation'
    # which usually iterates over tasks?
    # Let's assume it runs all configured tasks (which currently includes G1).
    
    # Remove --task if not supported
    cmd = [c for c in cmd if c != "--task" and c != "G1"]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Benchmark run failed!")
        return

    # 2. Score Results (AUPRC)
    # Find the results file
    results_dir = Path("results/dev/anot")
    # Find latest jsonl
    files = list(results_dir.glob("results_*.json"))
    if not files:
        print("No results found!")
        return
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"\nScoring results from: {latest_file}")
    
    score_cmd = ["python", "explore/score_auprc.py", str(latest_file)]
    subprocess.run(score_cmd)

if __name__ == "__main__":
    # Ensure we are in root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # If script is in explore/, move up
    if os.getcwd().endswith('explore'):
        os.chdir('..')
        
    run_benchmark()
