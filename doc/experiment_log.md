# Experiment Log

This document tracks all experiments comparing knot (Knowledge Network of Thought) vs cot (Chain-of-Thought) methods.

## Summary

| Experiment | knot Accuracy | cot Accuracy | Key Finding |
|------------|---------------|--------------|-------------|
| 1. gpt-4o-mini Baseline | 56% | 40% | knot +16pp on clean data |
| 2. Mixed Model + Attacks | 47% (consistent) | 20-53% (varies) | knot more robust |

---

## Experiments

### 1. gpt-4o-mini Baseline
**File:** [1_gpt4omini_baseline.md](1_gpt4omini_baseline.md)

**Summary:** Compared knot vs cot using gpt-4o-mini for all LLM calls. knot (56%) outperforms cot (40%) by 16 percentage points on real Yelp data.

**Key Result:** knot's 2-phase architecture provides better reasoning decomposition.

---

### 2. Mixed Model Robustness Testing
**File:** [2_mixed_model_robustness.md](2_mixed_model_robustness.md)

**Summary:** Compared knot (gpt-4o-mini planner + gpt-5-nano worker) vs cot on clean and adversarial inputs. knot shows consistent 47% accuracy across all attack types while cot degrades from 53% to 20% under attacks.

**Key Results:**
- knot wins 4/8 attack scenarios
- knot provides robustness at cost of ~6pp clean accuracy
- gpt-5-nano requires `max_completion_tokens=4096` due to reasoning tokens

---

## Technical Notes

### Model Configurations Tested
1. **All gpt-4o-mini:** Best clean accuracy for knot
2. **Mixed (planner=4o-mini, worker=5-nano):** Good robustness
3. **All gpt-5-nano:** Works after token limit fix

### Code Locations
- `knot.py` - Knowledge Network of Thought implementation
- `cot.py` - Chain-of-Thought implementation
- `llm.py` - LLM API wrapper with model config
- `main.py` - Evaluation runner
