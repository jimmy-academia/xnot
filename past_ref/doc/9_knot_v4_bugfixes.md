# KNoT v4 Bug Fixes

## Overview

After initial v4 implementation, all predictions were 0 regardless of input. This document describes the debugging process and fixes applied.

## Symptoms

- Run 31: 12/40 correct (30%) - but ALL predictions were 0
- Accuracy only came from cases where gold label happened to be 0
- Debug logs were empty (0 entries)

## Root Causes & Fixes

### Bug 1: Variable Substitution Failing

**Symptom**: Script step showed empty review content:
```
Step (0): From review 0: , extract evidence about speed...
                        ^ empty!
```

**Root Cause**: v4 generates scripts with dict-style access:
```
{(input)}[item_data][0][review]
```
But the pipeline was running with `mode=string`, which passes the query as a formatted string instead of a dict. The `substitute_variables` function returns empty string when trying dict access on a string.

**Fix**: Force v4 to use `mode=dict`:

```python
# methods/knot.py:471-476
elif approach == "v4":
    from .knot_v4 import KnowledgeNetworkOfThoughtV4
    # v4 requires dict mode for variable substitution
    if mode != "dict" and DEBUG:
        print(f"Warning: v4 requires mode=dict, overriding mode={mode}")
    _executor = KnowledgeNetworkOfThoughtV4(mode="dict", run_dir=run_dir)
```

```python
# main.py:284-288
# v4 approach requires dict mode for variable substitution
if args.method == "knot" and approach == "v4":
    eval_mode = "dict"
else:
    eval_mode = args.mode if args.method == "knot" else "string"
```

### Bug 2: parse_final_answer Not Handling Intermediate Values

**Symptom**: Script intermediate steps output "POSITIVE", "NEGATIVE", "NEUTRAL" which all parsed to 0.

**Root Cause**: `parse_final_answer` only handled explicit -1/0/1 values and "recommend" keywords, defaulting to 0 for anything else.

**Fix**: Added handling for POSITIVE/NEGATIVE/NEUTRAL:

```python
# methods/base.py:454-461
lower = output.lower()
# Handle POSITIVE/NEGATIVE/NEUTRAL from script intermediate steps
if "negative" in lower:
    return -1
if "positive" in lower:
    return 1
if "neutral" in lower:
    return 0
```

### Bug 3: Ambiguous Aggregation Prompt

**Symptom**: Aggregation step returned confused response:
```
"Please provide the FAST and SLOW evidence items..."
```

**Root Cause**: The example prompt template was ambiguous:
```
"Aggregate speed evidence from {(0)}, {(2)}, {(4)}. Count FAST vs SLOW. Output: POSITIVE, NEGATIVE, or NEUTRAL"
```
When substituted, it became "Aggregate speed evidence from FAST, SLOW. Count FAST vs SLOW..." which confused the LLM.

**Fix**: Improved example in `knot_v4.py:356-357`:

```python
# Before
(6)=LLM("Aggregate speed evidence from {{(0)}}, {{(2)}}, {{(4)}}. Count FAST vs SLOW. Output: POSITIVE, NEGATIVE, or NEUTRAL")

# After
(6)=LLM("Evidence: R0={{(0)}}, R1={{(2)}}, R2={{(4)}}. If more FAST than SLOW -> POSITIVE. If more SLOW than FAST -> NEGATIVE. Otherwise -> NEUTRAL. Output ONLY: POSITIVE, NEGATIVE, or NEUTRAL")
```

## Verification

Created minimal test case with 2 reviews, 1 request:
- R0: "Fast service, great food!" (gold: recommend)
- R1: "Slow and overpriced."

Execution trace:
```
Step (0): "Fast service, great food!" -> FAST
Step (1): "Slow and overpriced." -> SLOW
Step (2): Evidence R0=FAST, R1=SLOW -> POSITIVE
Step (3): speed=POSITIVE -> 1

Final answer: 1
Overall: 1.0000 (1/1)
```

## Files Modified

| File | Change |
|------|--------|
| `methods/knot.py` | Force `mode=dict` for v4 approach |
| `methods/base.py` | Handle POSITIVE/NEGATIVE/NEUTRAL in `parse_final_answer` |
| `methods/knot_v4.py` | Improved aggregation prompt example, added debug output |
| `main.py` | Set `eval_mode=dict` when approach is v4 |

## Lessons Learned

1. **Mode consistency**: When a method generates scripts assuming dict access, the input must be formatted as dict
2. **Answer parsing**: All possible LLM outputs should be handled, not just the expected final format
3. **Prompt clarity**: Aggregation prompts should be explicit about decision rules, not just say "count X vs Y"
4. **Fast iteration**: Use minimal test cases (2 reviews, 1 request) for debugging - full dataset is too slow
