# KNoT v5 - Work in Progress

## Status: INCOMPLETE

Implementation started but not finished. The hardcoded request matching approach was rejected.

---

## What Was Done

### 1. Adversarial Dataset Generator

**File:** `scripts/generate_adversarial_data.py`

Created script to generate evaluation dataset with:
- 20-40 reviews per item (vs current 6-12)
- Per-review condition tags (speed, food_quality, value, service, consistency, portion_size, ambiance, cleanliness)
- Pre-applied mixed attacks:
  - ~10% typo attacks
  - ~10% injection attacks (override, fake_system, hidden, manipulation)
  - 1-2 fake reviews per item
- Attack metadata tracking

### 2. Generated Dataset

**File:** `data/processed/adversarial_data.jsonl`

- 10 items, avg 32.8 reviews/item
- Attack distribution: 26 typo, 26 injection, 20 fake, 256 clean
- Label distribution balanced across C0-C7

### 3. Baseline Results

| Method | Accuracy |
|--------|----------|
| CoT | 36.25% (29/80) |
| CoT+Defense | 46.25% (37/80) |

Both baselines fail on adversarial data as expected.

### 4. Partial v5 Implementation

**File:** `methods/knot_v5.py`

4-stage pipeline structure:
- Stage 1: Review classification & filtering (CLEAN/SUSPICIOUS/ATTACK)
- Stage 2: Evidence extraction with confidence weighting
- Stage 3: Cross-review consensus (2+ agreement required)
- Stage 4: Logic-aware decision (MUST/SHOULD/NICE)

**Issue:** Request ID extraction was hardcoded to match context text patterns. This approach is fragile and not generalizable.

---

## What's Needed

### Better Request ID Passing

The method only receives `(query, context)` but v5 needs to know which request structure (C0-C7) to apply. Options:

1. **Pass request_id in context**: Modify `eval.py` to include request ID in context string
2. **Pass request structure directly**: Modify method signature to accept request structure
3. **Parse request dynamically**: Have v5 parse the context text to extract conditions using LLM

### Proper Logic Evaluation

Instead of hardcoding REQUEST_STRUCTURES, should:
1. Parse request structure from context using LLM
2. Or receive structure as parameter

---

## Files Changed

| File | Change |
|------|--------|
| `scripts/generate_adversarial_data.py` | CREATE - adversarial dataset generator |
| `data/processed/adversarial_data.jsonl` | CREATE - generated dataset |
| `methods/knot_v5.py` | CREATE - partial v5 implementation (WIP) |
| `methods/__init__.py` | EDIT - added v5 import and approach |
| `main.py` | EDIT - added v5 to knot-approach choices |
| `results/55_cot_adversarial/` | CoT baseline results |
| `results/56_cot_defense_adversarial/` | CoT+Defense baseline results |

---

## Next Steps

1. Design proper request structure passing mechanism
2. Remove hardcoded request matching
3. Complete v5 evaluation pipeline
4. Compare v5 accuracy vs baselines
