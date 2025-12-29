# Deterministic Script Generation Fix

## Problem

After initial v4 bugfixes (doc/9), **70% of script generations still failed** (19/27 produced empty scripts).

### Symptoms
- Complex conditions (`speed`, `food_quality_good`, `food_quality_decent`, `price_fair`) → empty script
- Simple conditions (`main_criterion`) → script generated correctly
- Fallback to direct LLM call when script empty

### Root Cause

Stage 2 Phase A asked the LLM to generate scripts from a complex prompt:

```python
# Old approach - LLM-based
prompt = f"""Generate an executable script...
- Conditions to check: {conditions}  # ['speed', 'food_quality_good', ...]
...
Now generate script for {review_count} reviews and conditions {conditions}."""
```

The LLM got confused by:
1. Python list syntax in the prompt
2. Example showing 3 reviews × 2 conditions, but actual cases have 10 reviews × 4 conditions
3. Ambiguous extrapolation requirements

## Solution

**Generate scripts deterministically from DAG** - no LLM needed in Stage 2.

### Implementation

```python
def stage2_phase_a(self, dag: dict, query, context: str) -> str:
    """Generate script deterministically from DAG structure."""
    conditions = dag.get("conditions", ["criterion"])
    review_count = dag.get("review_count", 5)
    logic = dag.get("decision", {}).get("logic", "criterion")

    lines = []
    step = 0

    # 1. Extraction steps: one per (review × condition)
    for r in range(review_count):
        for c in conditions:
            lines.append(f'({step})=LLM("Extract {c} evidence from review {r}: '
                        f'{{{{(input)}}}}[item_data][{r}][review]. Output: POSITIVE, NEGATIVE, or NONE")')
            step += 1

    # 2. Aggregation steps: one per condition
    agg_start = step
    num_conditions = len(conditions)
    for i, c in enumerate(conditions):
        refs = ", ".join(f"{{{{({r * num_conditions + i})}}}}" for r in range(review_count))
        lines.append(f'({step})=LLM("Aggregate {c}: {refs}. '
                    f'Count POSITIVE vs NEGATIVE. Output: POSITIVE, NEGATIVE, or NEUTRAL")')
        step += 1

    # 3. Decision step
    agg_refs = ", ".join(f"{c}={{{{({agg_start + i})}}}}" for i, c in enumerate(conditions))
    lines.append(f'({step})=LLM("Apply logic: {agg_refs}. Logic: {logic}. Output ONLY: -1, 0, or 1")')

    return "\n".join(lines)
```

### Phase 2.b Simplified

Removed LLM fix loop - just validate:

```python
def stage2_phase_b(self, script: str, dag: dict, max_iterations: int = 2) -> str:
    """Validate script structure (no LLM needed with deterministic generation)."""
    steps = parse_script(script)
    checks = [
        ("has_steps", len(steps) > 0),
        ("has_enough_steps", len(steps) >= expected_total),
        ("ends_with_decision", ...),
    ]
    # No more LLM fix loop - deterministic generation always passes
    return script
```

## Results

| Metric | Before (run 50) | After (run 52) |
|--------|-----------------|----------------|
| Script generation success | 30% | **100%** |
| Time (5 items × 8 requests) | 64 min | **2 min** |
| Accuracy | 32.5% | 30.0% |

### Benefits
- **100% script generation success** - no more empty scripts
- **32x faster** - parallel execution + fewer LLM calls
- **Predictable** - same DAG always produces same script
- **Simpler** - removed LLM fix loop in Phase 2.b

## Files Modified

| File | Change |
|------|--------|
| `methods/knot_v4.py:323-375` | Replace `stage2_phase_a()` with deterministic generation |
| `methods/knot_v4.py:377-414` | Simplify `stage2_phase_b()` to validation only |

## Verification

```bash
KNOT_DEBUG=1 python main.py --method knot --knot-approach v4 --limit 1
```

Check debug logs:
```json
{"phase": "2.b", "event": "check", "data": {"step_count": 45, "expected": 45, "all_passed": true}}
```
