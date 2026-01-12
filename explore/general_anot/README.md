# General ANoT: Two-Phase Task Execution

General ANoT separates **task understanding** (Phase 1) from **task execution** (Phase 2) using a structured intermediate representation called the **Formula Seed**.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Task Prompt    │────▶│   Phase 1 (LLM)  │────▶│  Formula Seed   │
│  (Query; en)    │     │  "Understand"    │     │  (JSON schema)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Restaurant    │────▶│   Phase 2        │────▶│  Risk Score +   │
│   + Reviews     │     │  (Interpreter)   │     │  Verdict        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Key insight**: The Formula Seed is a *predetermined schema* that captures task semantics without hardcoding domain logic. Phase 2 is a generic interpreter that executes any Formula Seed.

---

## Phase 1: Generate Formula Seed

**Location**: `phase1.py`

### How It Works

Phase 1 makes a **single LLM call** with detailed instructions. There are **no few-shot examples** - the model learns the schema purely from the instruction prompt.

The prompt (`PHASE1_PROMPT`) asks the LLM to think through:

1. **FILTERING**: What keywords identify relevant reviews?
2. **EXTRACTION**: What semantic signals must be extracted from each review?
3. **AGGREGATION**: How are extracted values counted/combined?
4. **EXTERNAL DATA**: What restaurant-level lookups are needed?
5. **COMPUTATION**: What formulas produce the final output?

### Why No Few-Shot?

The task formula (input) is already highly structured and detailed. The LLM just needs to:
- Parse the formula's requirements
- Organize them into the predetermined schema sections
- Preserve all semantics (variable names, thresholds, formulas)

The schema acts as a "frame" that the LLM fills in. It's not generating novel logic - it's translating structured natural language into structured JSON.

### Formula Seed Schema

The output has these sections:

```json
{
  "version": "1.0",
  "overview": { "purpose": "...", "current_year": 2025 },

  "filtering": {
    "relevance_detectors": [
      { "type": "substring", "patterns": ["allergy", "peanut", ...] }
    ]
  },

  "per_review_extraction_schema": {
    "target_review_fields": {
      "INCIDENT_SEVERITY": {
        "type": "enum",
        "values": ["none", "mild", "moderate", "severe"],
        "default": "none",
        "extraction_rules": {
          "signals_for_severe": ["anaphylaxis", "epipen", ...]
        }
      }
    }
  },

  "aggregation_definitions": {
    "N_SEVERE": {
      "source": "reviews",
      "condition": { "INCIDENT_SEVERITY": "severe", "ACCOUNT_TYPE": "firsthand" }
    },
    "N_TOTAL_INCIDENTS": { "formula": "N_MILD + N_MODERATE + N_SEVERE" }
  },

  "external_data_and_lookup": {
    "CUISINE_MODIFIERS": { "Thai": 2.0, "Italian": 0.5, "default": 1.0 }
  },

  "calculation_steps": {
    "order_of_operations": ["TRUST_RAW", "TRUST_SCORE", ..., "VERDICT"],
    "definitions": {
      "TRUST_RAW": "1.0 + (N_POSITIVE * 0.1) - (N_NEGATIVE * 0.2)",
      "VERDICT": "if FINAL_RISK_SCORE < 4.0 then 'Low Risk' else ..."
    }
  },

  "output_specification": {
    "reported_values": ["FINAL_RISK_SCORE", "VERDICT", ...]
  }
}
```

---

## Phase 2: Execute Formula Seed

**Location**: `phase2.py`

Phase 2 is a **generic interpreter** with no task-specific knowledge. It reads the Formula Seed and executes it step-by-step.

**Key insight**: Only Step 2 uses LLM. Everything else is deterministic Python execution.

### Execution Overview

| Step | Executor | Input | Output |
|------|----------|-------|--------|
| 1. Filter | **Python** | Review text | Relevant reviews |
| 2. Extract | **LLM** | Relevant reviews | Structured JSON per review |
| 3. Aggregate | **Python** | Structured JSON | Counts (N_SEVERE, etc.) |
| 4. Calculate | **Python** | Counts | Final score (Network of Thought) |
| 5. Output | **Python** | All values | Requested fields |

### Execution Steps

#### Step 1: Filter Reviews - Sparse Retrieval (`_filter_reviews`)

**Executor**: Python (keyword matching)
**Input**: All restaurant reviews (raw text)
**Output**: Subset of reviews mentioning relevant keywords

```python
# Sparse retrieval: keyword search on review TEXT
patterns = ["allergy", "peanut", "allergic", "epipen", ...]  # From Formula Seed

for review in all_reviews:
    if any(pattern in review.text for pattern in patterns):
        relevant_reviews.append(review)
```

**Example**: From 200 reviews → 5 reviews that mention allergy-related keywords.

---

#### Step 2: Extract Signals - LLM Classification (`_extract_signals`)

**Executor**: LLM (one call per relevant review)
**Input**: Relevant reviews from Step 1
**Output**: Structured JSON with semantic fields

For each relevant review, LLM extracts fields defined in Formula Seed:

```python
# LLM prompt (built from Formula Seed schema)
prompt = f"""
FIELDS TO EXTRACT:

INCIDENT_SEVERITY:
  Values: ["none", "mild", "moderate", "severe"]
  Look for severe: anaphylaxis, epipen, ER visit...

ACCOUNT_TYPE:
  Values: ["none", "firsthand", "secondhand", "hypothetical"]
  Look for firsthand: "I had", "my child experienced"...

REVIEW TEXT: "My child had hives after eating here. Staff was dismissive."

Return JSON:
"""

# LLM output:
{"INCIDENT_SEVERITY": "moderate", "ACCOUNT_TYPE": "firsthand", "SAFETY_INTERACTION": "negative"}
```

**Output**: List of structured `Extraction` objects:
```python
extractions = [
    Extraction(fields={"INCIDENT_SEVERITY": "moderate", "ACCOUNT_TYPE": "firsthand", ...}, meta={...}),
    Extraction(fields={"INCIDENT_SEVERITY": "severe", "ACCOUNT_TYPE": "firsthand", ...}, meta={...}),
    Extraction(fields={"INCIDENT_SEVERITY": "none", "ACCOUNT_TYPE": "hypothetical", ...}, meta={...}),
]
```

---

#### Step 3: Compute Aggregations - Count Structured Data (`_compute_aggregations`)

**Executor**: Python (NO LLM, NO keyword matching)
**Input**: Structured extractions from Step 2
**Output**: Counts stored in `ctx.values`

Step 3 operates on the **structured JSON from Step 2**, not on review text:

```python
# Formula Seed defines conditions (from Phase 1):
"N_SEVERE": {
    "condition": {"INCIDENT_SEVERITY": "severe", "ACCOUNT_TYPE": "firsthand"}
}

# Python dynamically matches field VALUES (not keywords!):
def _count_matching(self, condition):
    count = 0
    for ext in self.ctx.extractions:
        if self._matches_condition(ext, condition):
            count += 1
    return count

def _matches_condition(self, ext, condition):
    for field, expected in condition.items():  # ← Iterates over WHATEVER Phase 1 defined
        actual = ext.fields.get(field)
        if actual != expected:
            return False
    return True
```

**Not hardcoded!** The field names (`INCIDENT_SEVERITY`, `ACCOUNT_TYPE`) and values (`"severe"`, `"firsthand"`) come from the Formula Seed. The same Python code works for any schema Phase 1 generates.

---

#### Step 4: Execute Calculations - Network of Thought (`_execute_calculations`)

**Executor**: Python (deterministic formula evaluation)
**Input**: Aggregated counts from Step 3
**Output**: All intermediate and final values

This is the **Network of Thought** - a DAG of computations where each intermediate result is stored and referenced by later steps:

```python
# Formula Seed defines dependency order (from Phase 1):
"order_of_operations": [
    "TRUST_RAW",              # Uses: N_POSITIVE, N_NEGATIVE, N_BETRAYAL
    "TRUST_SCORE",            # Uses: TRUST_RAW
    "MILD_WEIGHT",            # Uses: TRUST_SCORE
    "ADJUSTED_INCIDENT_SCORE", # Uses: MILD_WEIGHT, N_MILD, N_MODERATE, N_SEVERE
    ...
    "FINAL_RISK_SCORE",       # Uses: multiple intermediate values
    "VERDICT"                 # Uses: FINAL_RISK_SCORE
]

# Python executes in order, storing each result:
for step_name in order:
    formula = definitions[step_name]
    ctx.values[step_name] = evaluate_formula(formula)  # ← Stored for later steps
```

**Network structure** (intermediate values feed into later calculations):
```
N_POSITIVE ──┐
N_NEGATIVE ──┼──▶ TRUST_RAW ──▶ TRUST_SCORE ──┬──▶ MILD_WEIGHT ────────┐
N_BETRAYAL ──┘                                ├──▶ MODERATE_WEIGHT ─────┼──▶ ADJUSTED_INCIDENT_SCORE
                                              └──▶ TRUST_IMPACT         │              │
N_MILD ─────────────────────────────────────────────────────────────────┘              │
N_MODERATE ────────────────────────────────────────────────────────────────────────────┤
N_SEVERE ──────────────────────────────────────────────────────────────────────────────┘
                                                                                       │
                                                        ... more dependencies ...      │
                                                                                       ▼
                                                                        FINAL_RISK_SCORE ──▶ VERDICT
```

Each node is computed once, stored in `ctx.values`, and available to all downstream nodes.

**Formula types handled**:

| Type | Example |
|------|---------|
| Arithmetic | `"1.0 + (N_POSITIVE * 0.1)"` |
| Conditional | `"X if Y > 0 else Z"` |
| Clamp | `"clamp(X, 0.1, 1.0)"` |
| Lookup | `"derived from CUISINE_MODIFIERS"` |

---

#### Step 5: Return Output (`_get_output`)

**Executor**: Python
**Input**: All computed values in `ctx.values`
**Output**: Dict with requested fields

```python
output_specification = {"reported_values": ["FINAL_RISK_SCORE", "VERDICT"]}

→ {"FINAL_RISK_SCORE": 13.7, "VERDICT": "Critical Risk"}
```

---

## Why This Design?

### Separation of Concerns

| Phase | Responsibility | Flexibility |
|-------|----------------|-------------|
| Phase 1 | Understand task semantics | Can adapt to any task prompt |
| Formula Seed | Structured representation | Predetermined schema, filled dynamically |
| Phase 2 | Execute specification | Generic interpreter, no task knowledge |

### Benefits

1. **Reproducibility**: Formula Seed captures exactly what logic will run
2. **Debuggability**: Can inspect intermediate values at each step
3. **Modularity**: Change Phase 1 prompt without touching Phase 2
4. **Auditability**: Formula Seed is human-readable JSON

### The "Right Amount" of Structure

The schema is:
- **Flexible enough**: LLM fills in task-specific values (keywords, thresholds, formulas)
- **Structured enough**: Phase 2 knows where to find each piece of information
- **Not hardcoded**: No domain logic in the interpreter itself

---

## Method Comparison

Three methods are compared on the G1a allergy safety task:

### 1. Direct LLM (Baseline)

**Location**: `baselines/direct_llm_v2.py`

No special prompting technique. Single LLM call with:
- Restaurant metadata + all reviews
- Full task formula (calculation steps)
- Ask LLM to compute ALL values in one shot

The LLM must filter, classify, count, and compute arithmetic - everything at once.

### 2. Chain of Thought (CoT)

**Location**: `baselines/cot.py`

Explicit step-by-step reasoning instructions:
```
STEP 1: SCAN FOR ALLERGY KEYWORDS
STEP 2: CLASSIFY EACH RELEVANT REVIEW
STEP 3: AGGREGATE COUNTS
STEP 4: COMPUTE DERIVED VALUES
STEP 5: COMPUTE FINAL SCORE
```

Still a single LLM call, but with structured reasoning guidance.

### 3. General ANoT (This Method)

Two-phase separation:
- **Phase 1**: LLM generates Formula Seed (once per task)
- **Phase 2**: Interpreter executes with per-review LLM extraction

LLM only does semantic classification. Interpreter handles filtering, counting, arithmetic deterministically.

### Benchmark Results

| Metric | Direct LLM | CoT | **General ANoT** |
|--------|------------|-----|------------------|
| **Adjusted AUPRC** | 0.426 | 0.607 | **0.765** |
| Ordinal AUPRC | 0.521 | 0.771 | **0.873** |
| Primitive Accuracy | 0.817 | 0.787 | **0.876** |
| AUPRC (>=Critical) | 0.333 | 1.000 | **1.000** |

**Key findings**:
- General ANoT achieves 26% higher Adjusted AUPRC than CoT
- Direct LLM struggles with Critical Risk detection (0.333 vs 1.000)
- Separating execution from understanding improves both ranking AND intermediate accuracy

**Result locations**:
```
explore/results/
├── G1a_cot_k200_*/                    # Chain of Thought
├── G1a-v2_k200_*/                     # Direct LLM baseline
└── general_anot_eval/001_*/           # General ANoT
```

---

## Running Evaluations

```bash
# Generate new formula seed from task prompt
python -m explore.general_anot.phase1

# Run evaluation (100 restaurants)
python -m explore.general_anot.eval
```

**Output structure**:
```
explore/results/general_anot_eval/
└── 001_20260111_235959/
    ├── formula_seed.json   # Copy of seed used (for reproducibility)
    └── results.json        # Predictions + metrics
```

---

## Files

| File | Purpose |
|------|---------|
| `phase1.py` | Generate Formula Seed from task prompt |
| `phase2.py` | `FormulaSeedInterpreter` - execute seed on restaurant data |
| `eval.py` | Full evaluation with AUPRC scoring |
| `utils.py` | JSON formatting, run numbering |
| `FORMULA_SEED_SPEC.md` | Detailed schema specification |
| `DESIGN.md` | Design decisions and rationale |
