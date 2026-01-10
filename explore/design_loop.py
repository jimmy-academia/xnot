#!/usr/bin/env python3
"""
explore/design_loop.py
Async Task Design Loop.

Usage:
    python explore/design_loop.py --query "..."
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm import call_llm_async, get_model

async def run_design_loop(task_id, query):
    model = "gpt-5-nano"
    print(f"--- Design Loop: {task_id} ---")
    print(f"Model: {model}")
    print(f"Query: {query}")
    
    # 1. Fetch Context (Mock for now)
    context = "Reference Context: [Review 1: ... Review 2: ...]" 
    
    print("\n[Executing Asynchronously]...")
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
    
    # Call Async
    response = await call_llm_async(prompt, model=model)
    
    print(f"\n[Model Response]:\n{response}\n")
    print("-" * 30)
    print("VERDICT: Check Primitives (P) and Final Answer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", default="Test_Task")
    parser.add_argument("--query", default="Who is the best server?")
    
    args = parser.parse_args()
    
    # Run async
    asyncio.run(run_design_loop(args.task_id, args.query))
