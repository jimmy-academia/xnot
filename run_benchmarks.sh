#!/bin/bash
# Run all baselines on selections 2 and 3

cd /Users/jimmyyeh/Documents/Station/xnot
source .venv/bin/activate

METHODS="cot cotsc react ps decomp l2m selfask prp listwise setwise parade rankgpt finegrained pal pot cot_table weaver"

for sel in 2 3; do
    for method in $METHODS; do
        echo "=============================================="
        echo "Running $method on selection_$sel"
        echo "=============================================="
        python3 main.py --method "$method" --selection "$sel" 2>&1 | tail -20
        echo ""
    done
done

echo "All benchmarks complete!"
