# KNoT v4 Planning Quality Analysis

## Executive Summary

**Run 52:** 30% accuracy - ALL 40 predictions default to 0

The 100% script generation success (doc/10) masks deep planning quality failures in Stage 1. The deterministic script generator faithfully replicates broken DAG structures.

---

## Observed Symptoms

| Metric | Run 50 (LLM scripts) | Run 52 (Deterministic) |
|--------|---------------------|------------------------|
| Script success | 30% | 100% |
| Predictions -1 | 6 | **0** |
| Predictions 0 | 21 | **40** |
| Predictions +1 | 13 | **0** |
| Accuracy | 32.5% | 30.0% |

**Key observation:** Deterministic generation eliminated script failures but exposed that the underlying DAG planning is broken - 100% of outputs collapse to 0.

---

## Root Cause Analysis: Stage 1 Planning Quality

### Issue 1: Hierarchical Logic Structure Lost (Phase 1.i)

**Problem:** Condition extraction flattens AND/OR trees into lists.

```
User request (C0):
  "Quick service is non-negotiable. I'd appreciate good food,
   but if it's just decent and prices are fair, that works too."

Ground truth structure:
  speed AND (good_food OR (decent_food AND fair_price))

LLM Phase 1.i output:
  must: ["speed"]
  should: ["good_food", "decent_food", "fair_price"]  # FLAT LIST
  logic: "speed AND (good_food OR (decent_food AND fair_price))"  # String, not tree
```

**Impact:** The hierarchical `(decent_food AND fair_price)` grouping becomes three independent conditions. The logic string exists but is not validated against the condition list structure.

### Issue 2: Malformed Composite Conditions (Phase 1.i → 1.v)

**Problem:** Some conditions contain logic operators as literal strings.

```
User request (C6):
  "Food needs to be either really good or value needs to be great"

Phase 1.i extracts:
  must: ["food_quality OR value"]  # MALFORMED: "OR" embedded in condition name

Phase 1.v DAG creates:
  steps: ["extract_food_quality OR value_R0"]  # Invalid step name
```

**Impact:** Extraction steps ask LLM to evaluate `"food_quality OR value"` as a single atomic condition rather than decomposing into separate evaluations.

### Issue 3: Generic Plans Fail Validation (Phase 1.iii → 1.iv)

**Problem:** Phase 1.iii generates abstract plans; Phase 1.iv uses string matching to validate.

```
Phase 1.iii output:
  "1. For each review, extract evidence for each condition"
  "2. Aggregate evidence per condition"
  "3. Apply logic to get final answer"

Phase 1.iv validation:
  Check: all(c.lower() in plan_text.lower() for c in conditions)
  Conditions: ["good_food", "friendly_service", "reasonable_prices"]
  Result: FAIL - "good_food" not in "for each condition"
```

**Impact:** Validation fails → triggers LLM regeneration → produces bloated 12-step plans that Phase 2 ignores anyway.

### Issue 4: Decision Rules Not Executed (Phase 1.i → Stage 3)

**Problem:** Phase 1.i generates decision_rule, but Stage 3 only uses the logic string.

```
Phase 1.i decision_rule:
  "if polite_staff negative -> -1;
   if polite_staff positive AND food_quality positive -> 1;
   else -> 0"

Stage 3 decision step:
  "Apply logic: staff=POSITIVE, food=NEGATIVE. Logic: staff AND food. Output: -1, 0, or 1"
```

**Impact:** The nuanced decision_rule (which correctly handles partial matches) is never used. The logic string `"staff AND food"` doesn't specify what output to give for partial matches.

### Issue 5: Semantic-Symbolic Gap (Stage 2 → Stage 3)

**Problem:** Aggregation outputs semantic labels, but decision prompt doesn't explain interpretation.

```
Aggregation outputs:    speed=POSITIVE, food=NEGATIVE
Decision prompt:        "Logic: speed AND food. Output: -1, 0, or 1"
Missing information:    What is POSITIVE AND NEGATIVE? True? False?
```

**Impact:** LLM produces ambiguous responses → parse_final_answer() defaults to 0.

---

## Failure Chain

```
Phase 1.i                    Phase 1.v                Stage 2              Stage 3
   │                            │                        │                    │
   ▼                            ▼                        ▼                    ▼
Flatten AND/OR          Malformed conditions      Ignore plan         Ambiguous decision
into lists              in DAG structure          entirely            prompt
   │                            │                        │                    │
   └─────────────────────────────┴────────────────────────┴────────────────────┘
                                        │
                                        ▼
                             All predictions = 0
```

---

## Two Categories of Problems

### Category A: Planning Quality (Stage 1)
- Logic structure not preserved as tree
- Conditions contain embedded operators
- Validation uses brittle string matching
- decision_rule ignored in execution

### Category B: Execution Quality (Stage 3)
- Semantic labels need explicit interpretation
- Logic strings need evaluation rules
- Decision prompt lacks mapping guidance

**Hardcoding prompt fixes only addresses Category B.** Category A problems mean the DAG itself is structurally incorrect, so even perfect execution would produce wrong answers.

---

## Summary Table

| Issue | Phase | Root Cause | Severity |
|-------|-------|------------|----------|
| Flat condition lists | 1.i | AND/OR tree → flat list | CRITICAL |
| Embedded operators | 1.i | `"X OR Y"` as single string | CRITICAL |
| String-match validation | 1.iv | Condition names ≠ plan text | HIGH |
| Unused decision_rule | 1.i→3 | Generated but never executed | CRITICAL |
| Semantic-symbolic gap | 2→3 | POSITIVE → ? in logic | HIGH |
| Default to 0 | 3 | Ambiguous response parsing | MEDIUM |

---

## Architectural Questions

1. **Should logic be preserved as a tree structure?**
   - Current: String like `"speed AND (food OR value)"`
   - Alternative: Nested dict `{"AND": ["speed", {"OR": ["food", "value"]}]}`

2. **Should conditions be atomic only?**
   - Current: Allow `"food_quality OR value"` as single condition
   - Alternative: Force decomposition into `["food_quality", "value"]` with explicit OR aggregation

3. **Should decision_rule be executable code?**
   - Current: Natural language string, ignored
   - Alternative: Parse into executable logic with explicit output mapping

4. **Should aggregation output numeric or semantic?**
   - Current: POSITIVE/NEGATIVE/NEUTRAL (semantic)
   - Alternative: +1/-1/0 (numeric, directly usable in logic)

5. **Is the 3-stage architecture appropriate?**
   - Current: Stage 1 (plan) → Stage 2 (script) → Stage 3 (execute)
   - Alternative: Simpler 2-stage with direct plan-to-execution?

---

## Related Documentation

- `doc/9_knot_v4_bugfixes.md` - Initial v4 bugfixes
- `doc/10_deterministic_script_generation.md` - Script generation fix (100% success)
- `results/52_knot/` - Run with 30% accuracy, all predictions = 0
