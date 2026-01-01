#!/bin/bash
# Run all baseline methods in parallel (clean runs, no defense)
#
# Usage:
#   ./scripts/run_baselines.sh [selection_number] [max_parallel_jobs]
#
# Examples:
#   ./scripts/run_baselines.sh 1      # Run on selection_1 with 4 parallel jobs
#   ./scripts/run_baselines.sh 1 8    # Run on selection_1 with 8 parallel jobs

set -e

SELECTION="${1:-1}"
MAX_PARALLEL="${2:-4}"

# All baselines (excluding anot methods)
METHODS=(
    "cot" "react" "decomp" "prp" "listwise" "finegrained"
    "cotsc" "l2m" "ps" "selfask" "parade" "rankgpt" "setwise"
    "pal" "pot" "cot_table" "weaver"
)

echo "========================================"
echo "Running ${#METHODS[@]} baseline methods"
echo "Selection: selection_$SELECTION"
echo "Max parallel: $MAX_PARALLEL"
echo "========================================"

run_method() {
    local method=$1
    local selection=$2
    echo "[$(date +%H:%M:%S)] Starting: $method"
    python main.py --method "$method" --selection "$selection" --benchmark 2>&1 | \
        tee "logs/${method}_sel${selection}.log"
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Completed: $method"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $method (exit code $status)"
    fi
    return $status
}

export -f run_method
export SELECTION

# Create logs directory
mkdir -p logs

# Run methods in parallel
if command -v parallel &> /dev/null; then
    # GNU parallel available
    printf '%s\n' "${METHODS[@]}" | parallel -j "$MAX_PARALLEL" run_method {} "$SELECTION"
else
    # Fallback to xargs
    printf '%s\n' "${METHODS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_method "$@"' _ {} "$SELECTION"
fi

echo ""
echo "========================================"
echo "All baselines complete!"
echo "========================================"
echo ""
echo "Aggregating results..."

# Aggregate each method and print summary
for method in "${METHODS[@]}"; do
    python -c "
from utils.aggregate import aggregate_benchmark_runs
try:
    summary = aggregate_benchmark_runs('$method', 'yelp', 'selection_$SELECTION')
    if summary.get('type') == 'ranking':
        hits_at_1 = summary.get('hits_at', {}).get(1, {}).get('mean', 0)
        print(f'$method: {hits_at_1:.4f} (Hits@1)')
    elif summary.get('type') == 'accuracy':
        print(f'$method: {summary.get(\"mean\", 0):.4f} (Accuracy)')
    elif summary.get('error'):
        print(f'$method: {summary.get(\"error\")}')
    else:
        print(f'$method: No results')
except Exception as e:
    print(f'$method: ERROR - {e}')
"
done
