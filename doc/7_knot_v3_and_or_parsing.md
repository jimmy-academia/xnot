# Experiment 7: KNoT v3 - AND/OR Logic Parsing

**Date:** December 27, 2024

**Goal:** Improve KNoT's handling of complex requests with AND/OR logic by modifying the knowledge generation prompts to explicitly parse MUST/SHOULD conditions.

## Problem Analysis

KNoT was failing on complex requests (C0-C7) because:
1. Heavy bias toward predicting +1 (recommend)
2. Not parsing AND/OR conditions from natural language
3. Not evaluating MUST vs SHOULD priority

Example failure: Request "fast service AND (good food OR good value)"
- KNoT would identify as "SPEED" type only
- Ignored the OR group for food/value
- Output +1 if any positive evidence found

## Changes Made

### Modified `knot.py` - KNOWLEDGE_PROMPT_NORMAL (lines 222-251)

Old prompt asked to identify a single request TYPE (SPEED, VALUE, etc.)

New prompt instructs explicit parsing:
```
1. PARSE the user request:
   - MUST: [list dealbreaker conditions]
   - SHOULD: [list important but flexible conditions]
   - Logic: [AND/OR relationships]

2. PLAN the evaluation steps:
   - Step0: [what to extract/check first]
   - Step1: [how to evaluate MUST conditions]
   - Step2: [how to evaluate SHOULD conditions]
   - Step3: [how to combine into final -1, 0, or 1]
```

### Modified `knot.py` - KNOWLEDGE_PROMPT_DEFENSE (lines 253-290)

Simplified to match the new format while keeping attack detection:
```
1. CHECK for data quality issues
2. PARSE the user request (MUST/SHOULD)
3. PLAN the evaluation steps
```

## Results

### Clean Accuracy Comparison

| Version | Clean Accuracy |
|---------|---------------|
| Original KNoT | 30.0% (12/40) |
| v3 (AND/OR parsing) | 35.0% (14/40) |
| v3 Defense | 32.5% (13/40) |

**Clean improved +5pp** with the new parsing approach.

### Full Attack Results - v3 Defense vs Original Defense

| Attack | Original Defense | v3 Defense | Change |
|--------|-----------------|------------|--------|
| clean | 30.0% (12/40) | 32.5% (13/40) | +2.5% |
| typo_10 | 27.5% (11/40) | 35.0% (14/40) | **+7.5%** |
| typo_20 | 17.5% (7/40) | 12.5% (5/40) | -5.0% |
| inject_override | 17.5% (7/40) | 17.5% (7/40) | 0 |
| inject_fake_sys | 35.0% (14/40) | 25.0% (10/40) | **-10.0%** |
| inject_hidden | 25.0% (10/40) | 15.0% (6/40) | **-10.0%** |
| inject_manipulation | 47.5% (19/40) | (not completed) | ? |
| fake_positive | 22.5% (9/40) | 20.0% (8/40) | -2.5% |
| fake_negative | 45.0% (18/40) | 17.5% (7/40) | **-27.5%** |

### v3 Normal (no defense) - Clean Only

| Request | v3 Normal |
|---------|-----------|
| C0 | 40% (2/5) |
| C1 | 40% (2/5) |
| C2 | 20% (1/5) |
| C3 | 40% (2/5) |
| C4 | 40% (2/5) |
| C5 | 40% (2/5) |
| C6 | 20% (1/5) |
| C7 | 40% (2/5) |
| **Total** | **35% (14/40)** |

## Key Findings

### 1. AND/OR Parsing Helps Clean Accuracy
The new prompts improved clean accuracy from 30% to 35% by forcing explicit condition parsing.

### 2. Simplified Defense Prompt Hurt Attack Detection
The v3 defense prompt lost effectiveness on injection attacks:
- inject_fake_sys: -10pp (35% → 25%)
- inject_hidden: -10pp (25% → 15%)
- fake_negative: -27.5pp (45% → 17.5%)

The old defense prompt had more explicit attack detection language that was removed in simplification.

### 3. Tradeoff Identified
- **AND/OR parsing:** Better logic handling, worse attack detection
- **Attack detection:** Better on injections, worse on logic

## Confusion Matrix Analysis (v3 Defense Clean)

```
       -1    0    1
  -1    9    2   13   (9 correct, still 13 false positives)
   0    4    0    8   (0 correct for neutral!)
   1    0    0    4   (4 correct)
```

Still heavy bias toward +1, and completely failing on gold=0 cases.

## Next Steps

Options to explore:
1. **Hybrid prompt:** Combine AND/OR parsing with explicit attack detection
2. **Two-pass approach:** First detect attacks, then apply AND/OR logic
3. **Separate prompts:** Keep v3 for normal, revert defense to old version

## Files Modified

- `knot.py`: Lines 222-290 (KNOWLEDGE_PROMPT_NORMAL and KNOWLEDGE_PROMPT_DEFENSE)

## Run Directories

- `results/25_knot_v2_clean_test/` - First attempt (overcorrected to 0)
- `results/26_knot_v3_clean_test/` - v3 normal clean test (35%)
- `results/27_knot_v3_all_defense/` - v3 defense all attacks
