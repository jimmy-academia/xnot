import json
from pathlib import Path

DATA_FILE = Path("explore/data/packed/dataset_N50.jsonl")
OUT_DIR = Path("explore/data/eval_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_FILE) as f:
    lines = [json.loads(line) for line in f]
    
# Take first 5 for now, or random? Let's take first 5 stratified.
# Actually, eval.py expects ONE file per restaurant:
# Line 1: Restaurant Meta
# Lines 2...: Reviews
# My dataset_N50.jsonl has `business` and `reviews` in ONE object.

for i, ctx in enumerate(lines[:50]): # Unpack 50
    # Sanitize filename
    clean_name = "".join(c if c.isalnum() else "_" for c in ctx['business']['name'])
    fname = f"{i:03d}_{clean_name}.jsonl"
    
    with open(OUT_DIR / fname, "w") as out:
        # Line 1: Business Meta (eval.py expects this)
        out.write(json.dumps(ctx['business']) + "\n")
        
        # Lines 2+: Reviews
        for r in ctx['reviews']:
            # eval.py expects each review on a new line
            out.write(json.dumps(r) + "\n")
            
print(f"Unpacked {i+1} contexts to {OUT_DIR}")
