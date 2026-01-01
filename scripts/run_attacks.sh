#!/bin/bash
# Run all baseline methods with all attacks in parallel
#
# Usage:
#   ./scripts/run_attacks.sh [selection] [max_parallel] [attacks...]
#
# Examples:
#   ./scripts/run_attacks.sh 1 4                    # All 12 attacks, 4 parallel
#   ./scripts/run_attacks.sh 1 8 typo_10 typo_20    # Specific attacks only
#   ./scripts/run_attacks.sh 1 4 inject_override    # Single attack

set -e

SELECTION="${1:-1}"
MAX_PARALLEL="${2:-4}"
shift 2 2>/dev/null || true

# Attacks to run (default: all 12)
if [ $# -gt 0 ]; then
    ATTACKS=("$@")
else
    ATTACKS=(
        # Typo attacks
        "typo_10" "typo_20"
        # Injection attacks
        "inject_override" "inject_fake_sys" "inject_hidden" "inject_manipulation"
        # Fake review attacks
        "fake_positive" "fake_negative"
        # Sarcastic attacks
        "sarcastic_wifi" "sarcastic_noise" "sarcastic_outdoor" "sarcastic_all"
    )
fi

# All 17 baseline methods (same as run_baselines.sh)
METHODS=(
    "cot" "react" "decomp" "prp" "listwise" "finegrained"
    "cotsc" "l2m" "ps" "selfask" "parade" "rankgpt" "setwise"
    "pal" "pot" "cot_table" "weaver"
)

run_method_attack() {
    local method=$1
    local attack=$2
    local selection=$3
    echo "[$(date +%H:%M:%S)] Starting: $method + $attack"
    python main.py --method "$method" --attack "$attack" --selection "$selection" --benchmark --seed 42 2>&1 | \
        tee "logs/${method}_${attack}_sel${selection}.log"
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Completed: $method + $attack"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $method + $attack (exit code $status)"
    fi
    return $status
}

export -f run_method_attack

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Attack Benchmark"
echo "Selection: selection_$SELECTION"
echo "Methods: ${#METHODS[@]}"
echo "Attacks: ${#ATTACKS[@]} (${ATTACKS[*]})"
echo "Total runs: $((${#METHODS[@]} * ${#ATTACKS[@]}))"
echo "Max parallel: $MAX_PARALLEL"
echo "========================================"

# Track timing
START_TIME=$(date +%s)

for attack in "${ATTACKS[@]}"; do
    echo ""
    echo "========================================"
    echo ">>> Attack: $attack ($(date +%H:%M:%S))"
    echo "========================================"

    if command -v parallel &> /dev/null; then
        # GNU parallel available
        printf '%s\n' "${METHODS[@]}" | parallel -j "$MAX_PARALLEL" run_method_attack {} "$attack" "$SELECTION"
    else
        # Fallback to xargs
        printf '%s\n' "${METHODS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_method_attack "$@"' _ {} "$attack" "$SELECTION"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "All attacks complete!"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "========================================"
echo ""
echo "Aggregating results..."
echo ""

# Export for Python script
export SELECTION
export ATTACKS_LIST="${ATTACKS[*]}"

# Print summary table
python3 << 'PYEOF'
from utils.aggregate import aggregate_benchmark_runs
import sys
import os

methods = ["cot", "react", "decomp", "prp", "listwise", "finegrained",
           "cotsc", "l2m", "ps", "selfask", "parade", "rankgpt", "setwise",
           "pal", "pot", "cot_table", "weaver"]

# Get attacks from environment or use defaults
attacks_env = os.environ.get('ATTACKS_LIST', '')
if attacks_env:
    attacks = attacks_env.split()
else:
    attacks = [
        "typo_10", "typo_20", "inject_override", "inject_fake_sys",
        "inject_hidden", "inject_manipulation", "fake_positive", "fake_negative",
        "sarcastic_wifi", "sarcastic_noise", "sarcastic_outdoor", "sarcastic_all"
    ]

selection = os.environ.get('SELECTION', '1')

# Truncate attack names for display
def short_name(a, maxlen=10):
    if len(a) <= maxlen:
        return a
    # Remove common prefixes for brevity
    for prefix in ['inject_', 'sarcastic_', 'fake_']:
        if a.startswith(prefix):
            return a[len(prefix):][:maxlen]
    return a[:maxlen]

# Header
print(f"{'Method':<12}", end="")
for a in attacks:
    print(f"{short_name(a):<10}", end="")
print()
print("-" * (12 + 10 * len(attacks)))

# Results
for method in methods:
    print(f"{method:<12}", end="")
    for attack in attacks:
        try:
            s = aggregate_benchmark_runs(method, 'yelp', f'selection_{selection}', attack)
            if s.get('type') == 'ranking':
                val = s.get('hits_at', {}).get(1, {}).get('mean', 0)
            else:
                val = s.get('mean', 0)
            print(f"{val:<10.3f}", end="")
        except Exception:
            print(f"{'---':<10}", end="")
    print()

print()
print("Legend: Values are Hits@1 (ranking) or Accuracy")
PYEOF
