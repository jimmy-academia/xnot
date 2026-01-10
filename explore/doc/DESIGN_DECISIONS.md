# SCALE Benchmark: Design Decisions

This document captures key design decisions for the SCALE benchmark, organized for research paper reference.

---

## 1. Ground Truth Architecture

### 1.1 Two-Phase Semantic GT Pipeline

**Decision**: Use keyword filtering + LLM judgment, not full LLM evaluation on all reviews.

**Rationale**:
- 100 restaurants × 200 reviews = 20,000 reviews total
- Only ~16% of reviews are topic-relevant (keyword match)
- **13x reduction** in LLM calls for GT extraction

**Implementation**:
```
Phase 1: Keyword Filter (no LLM)
  - Scan all reviews for task-specific keywords
  - Mark relevance per task (supports multi-task reuse)
  - Result: ~2,000 relevant reviews per task

Phase 2: LLM Judgment (only on matches)
  - Extract semantic primitives from matched reviews
  - Store judgments permanently for reproducibility
```

### 1.2 Deterministic GT Computation

**Decision**: GT is computed from stored judgments via explicit Python formulas.

**Rationale**:
- Judgments are subjective (LLM-dependent), formulas are objective
- Anyone can verify GT by re-running formulas on stored judgments
- Enables formula updates without re-judging reviews
- Supports debugging: trace any aggregate back to specific reviews

**Key Insight**:
```
LLM Judgment (once) → judgments.json → Python formulas → GT values
                      ^^^^^^^^^^^^^^^^
                      ONLY LLM-dependent artifact
                      Everything else is deterministic
```

### 1.3 Dynamic GT per K

**Decision**: Ground truth computed from reviews 0 to K-1 only.

**Rationale**:
- Fair evaluation across different context sizes (K=25, K=50, K=100, K=200)
- Model seeing K reviews should be scored against GT from same K reviews
- Avoids penalizing models for missing information in unseen reviews

**Example** (Vetri Cucina):
```
Incident review at index 39, positive interactions at indices 49, 52, 59, 83, 88...

K=25: incident not visible → 0 incidents → Low Risk
K=50: incident visible, fewer positives → 1 incident → High Risk (6.94)
K=100: incident visible, more positives → 1 incident → Low Risk (3.44)
K=200: most complete picture → 1 incident → Low Risk (1.44)
```

**Empirical GT Distribution by K**:
```
K=25:  Low=99, High=1,  Critical=0  (fewer reviews = fewer incidents visible)
K=50:  Low=96, High=2,  Critical=2
K=100: Low=96, High=1,  Critical=3
K=200: Low=93, High=5,  Critical=2  (complete picture)
```

**Usage**:
```python
from g1_gt_compute import compute_gt_for_k
gt = compute_gt_for_k("Restaurant Name", k=50)  # GT from reviews 0-49
```

---

## 2. Task Naming Convention

### 2.1 Task ID Format: G{group}{letter}

**Decision**: Use `G1a`, `G1b`, `G2a` format (not `G1.1`, `G1.2`).

**Rationale**:
- Avoids confusion with version numbers (G1.1 vs v1.1)
- Compact: 3 characters vs 4
- Supports 10 tasks per group (a-j)
- Consistent with academic task naming conventions

**Structure**:
```
G = Group prefix
{1-10} = Group number (perspective-based)
{a-j} = Task within group

Examples:
  G1a = Group 1, Task a (Peanut Allergy Safety)
  G2c = Group 2, Task c (Large Group Accommodation)
  G10j = Group 10, Task j (Misinformation Flag)
```

---

## 3. Scoring Architecture

### 3.1 Two-Level Evaluation

**Decision**: Per-restaurant primitives + cross-restaurant AUPRC.

**Level 1: Per-Restaurant Primitives**
- Purpose: Explainability and debugging
- What: Score individual calculation steps
- Why: Reveals WHERE reasoning failed

**Level 2: Cross-Restaurant AUPRC**
- Purpose: Benchmark metric for method comparison
- What: Cumulative ordinal AUPRC across all restaurants
- Why: Measures ranking quality, not just point accuracy

### 3.2 Cumulative Ordinal AUPRC

**Decision**: Use cumulative thresholds for ordinal classes.

**Rationale**:
- Risk levels are ordinal: Low < High < Critical
- "High Risk" classification includes both High and Critical
- Standard AUPRC would treat classes as independent

**Calculation**:
```
AUPRC_ge_High: Separate (High+Critical) from Low
AUPRC_ge_Critical: Identify Critical cases

ordinal_auprc = mean(AUPRC_ge_High, AUPRC_ge_Critical)
```

---

## 4. Formula Design (G1a Example)

### 4.1 Layered Computation DAG

**Decision**: 3-layer formula with explicit dependencies.

**Layer 1 - Per-Review Extraction (Semantic)**:
```
incident_severity: none | mild | moderate | severe
account_type: none | firsthand | secondhand | hypothetical
safety_interaction: none | positive | negative | betrayal
```

**Layer 2 - Aggregation (Arithmetic)**:
```
N_TOTAL_INCIDENTS = count(firsthand incidents)
INCIDENT_SCORE = (N_MILD × 2) + (N_MODERATE × 5) + (N_SEVERE × 15)
RECENCY_DECAY = max(0.3, 1.0 - (INCIDENT_AGE × 0.15))
CREDIBILITY_FACTOR = avg((5 - stars) + log(useful + 1))
```

**Layer 3 - Final Score**:
```
RAW_RISK = BASE_RISK + INCIDENT_IMPACT - SAFETY_IMPACT + CUISINE_IMPACT
FINAL_RISK_SCORE = clamp(RAW_RISK, 0, 20)
VERDICT = Low (<4) | High (4-8) | Critical (≥8)
```

### 4.2 Design Rationale

**Why 25+ primitives?**
- Target ~5% zero-shot success rate
- Deep DAG creates error cascade opportunities
- Each semantic extraction can fail independently

**Why explicit formulas in prompt?**
- Tests formula-following, not domain knowledge
- Removes ambiguity in expected behavior
- Enables precise primitive-level scoring

---

## 5. Data Pipeline

### 5.1 Source Data Locking

**Decision**: Lock `dataset_K200.jsonl` as immutable source.

**Rationale**:
- Reproducibility: same data across all experiments
- Hash verification: detect unauthorized changes
- Multi-task reuse: build index once, use for all 100 tasks

### 5.2 Multi-Task Review Index

**Decision**: Build keyword index for ALL tasks in single pass.

**Benefits**:
- O(n) scan instead of O(n × tasks)
- Each review marked for all relevant tasks
- Easy to add new tasks without re-scanning

**Storage**:
```
data/semantic_gt/
├── pipeline_config.json      # All task keywords
├── review_index/             # Per-restaurant keyword matches
│   └── {restaurant}.json     # task_relevance per review
└── task_G1a/                 # Task-specific artifacts
    ├── judgments.json        # LLM judgments
    └── computed_gt.json      # Deterministic GT
```

### 5.3 Session Continuity

**Decision**: Extraction is resumable with checkpoint.

**Implementation**:
- `extraction_log.json` tracks progress
- `--resume` flag continues from last checkpoint
- Safe against interruption

---

## 6. Model Selection

### 6.1 GT Extraction Model

**Decision**: Use gpt-5-nano for semantic judgments.

**Rationale**:
- Fast and cost-effective for high-volume extraction
- Sufficient for binary/categorical judgments
- Formula-based GT reduces sensitivity to model choice

**Validation**: Compared gpt-5-nano GT vs Claude GT:
- 78% agreement on verdict
- Differences mainly on theoretical cuisine risk
- Formula-based GT only counts actual incidents (preferred)

---

## 7. Evaluation Fairness

### 7.1 Review Indexing

**Decision**: Use original review indices (0-indexed from source).

**Rationale**:
- Consistent with dataset ordering
- K=50 means reviews 0-49 (not random 50)
- Reproducible across runs

### 7.2 Keyword-Only Reviews in Judgments

**Decision**: Only store judgments for keyword-matched reviews.

**Rationale**:
- 84% of reviews have no allergy-related content
- Storing "none/none/none" is wasteful
- Absence in judgments = not relevant

---

## Changelog

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-10 | Dynamic GT per K | Fair evaluation across context sizes |
| 2025-01-10 | G1a naming format | Avoid version number confusion |
| 2025-01-10 | Formula-based GT | Deterministic, reproducible |
| 2025-01-10 | Two-phase filtering | 13x efficiency improvement |
