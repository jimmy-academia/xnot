#!/bin/bash
#
# run_benchmarks.sh - Run all champion methods on philly_cafes dataset
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all methods
#   ./scripts/run_benchmarks.sh cot listwise # Run specific methods
#   RUNS=3 ./scripts/run_benchmarks.sh       # Set number of runs
#

set -e

# Configuration
RUNS=${RUNS:-1}  # Number of runs per method (override with RUNS=N)
DATA="philly_cafes"
METHODS=("cot" "ps" "listwise" "weaver" "anot")

# If methods specified as args, use those instead
if [ $# -gt 0 ]; then
    METHODS=("$@")
fi

echo "========================================"
echo "ANoT Benchmark Runner"
echo "========================================"
echo "Data: $DATA"
echo "Methods: ${METHODS[*]}"
echo "Runs per method: $RUNS"
echo "========================================"
echo ""

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "Activated virtual environment: $VIRTUAL_ENV"
    else
        echo "Warning: No virtual environment found"
    fi
fi

# Run each method
for method in "${METHODS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running method: $method"
    echo "========================================"

    if [ "$RUNS" -gt 1 ]; then
        # Multiple runs with auto-aggregation
        python main.py --method "$method" --data "$DATA" --benchmark --auto "$RUNS"
    else
        # Single run
        python main.py --method "$method" --data "$DATA" --benchmark
    fi
done

echo ""
echo "========================================"
echo "All benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved in: results/benchmarks/"
echo ""

# Show summary of all methods
echo "Quick summary:"
for method in "${METHODS[@]}"; do
    summary_file="results/benchmarks/${method}_${DATA}/clean/summary.json"
    if [ -f "$summary_file" ]; then
        hits5=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('mean', {}).get('Hits@5', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $method: Hits@5 = $hits5"
    else
        echo "  $method: (no summary yet)"
    fi
done
