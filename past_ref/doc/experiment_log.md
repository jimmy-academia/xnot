# Experiment Log

This document tracks all experiments comparing knot (Knowledge Network of Thought) vs cot (Chain-of-Thought) methods.

## Summary

| Experiment | knot Accuracy | cot Accuracy | Key Finding |
|------------|---------------|--------------|-------------|
| 1. gpt-4o-mini Baseline | 56% | 40% | knot +16pp on clean data |
| 2. Mixed Model + Attacks | 47% (consistent) | 20-53% (varies) | knot more robust |
| 3. Complex Task Design | **43.75%** | 22.5% | knot +21pp on complex tasks |
| 4. Parallel + All Attacks | 29% avg (60% best) | - | Parallel exec works; attack detection works |
| 5. CoT Normal vs Defense | - | 53% / 20% | Defense prompts hurt CoT; minimal prompt better |

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

### 3. Complex Task Design
**File:** [3_complex_task_design.md](3_complex_task_design.md)

**Summary:** Designed harder evaluation tasks requiring structured reasoning. Implemented 5-level evidence scoring (-2 to +2), requirement levels (MUST/SHOULD/NICE), and compound logic (AND/OR). Both methods use gpt-5-nano.

**Key Results:**
- knot: 43.75% (7/16) vs cot: 22.5% (9/40)
- Gap: +21.25 percentage points
- Complex tasks require structured decomposition that cot can't handle
- Natural language requirement parsing tests LLM understanding

---

### 4. Parallel Execution with All Attacks
**File:** [4_knot_parallel_all_attacks.md](4_knot_parallel_all_attacks.md)

**Summary:** Implemented DAG-based parallel execution for knot scripts. Tested across 9 attack types with gpt-5-nano. Best performance on inject_hidden (60%) where attack detection worked correctly.

**Key Results:**
- Parallel execution working (2-3 steps per layer)
- Attack detection successful in knowledge phase
- Empty script problem when knowledge contains final answer
- inject_hidden: 60%, typo_20/inject_fake_sys: 40%, others: 20%

---

### 5. CoT All Attacks (Normal vs Defense)
**File:** [5_cot_all_attacks.md](5_cot_all_attacks.md)

**Summary:** Tested CoT method with two prompt versions - normal (minimal) and defense (with data quality checks). Counterintuitively, defense prompts significantly hurt performance.

**Key Results:**
- CoT Normal: 53% average (80% clean, 100% inject_fake_sys)
- CoT Defense: 20% average (uniform poor performance)
- Defense prompts cause over-filtering and distraction
- Minimal prompt allows model to focus on content analysis

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
