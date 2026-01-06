# Research Plan: ANoT (Adaptive Network of Thought)

## Problem Statement

LLMs fail on structured multi-source data due to:
1. Heterogeneity in user content
2. Adversarial content (fake reviews, injection)
3. Structure confusion (metadata vs text)

## Solution: ANoT

Split-context prompting that:
- Isolates evidence by source
- Generates validated execution scripts
- Adapts to detected heterogeneity/attacks

## Task Framing: Constraint-Satisfying Reranking

**Scenario:** Last-mile RAG

**Fixed Context:** 20 candidates × 20 reviews

**Goal:** Find the ONE candidate satisfying all constraints

---

## Specifications

### Data

| Spec | Value |
|------|-------|
| Domain | Yelp (Philadelphia Cafes) |
| Candidates | 20 restaurants |
| Reviews per candidate | 20 |
| Requests | 50 total |
| Ground truth | Exactly 1 valid per request |

### Request Groups

| Group | Description | Evidence Type | Count |
|-------|-------------|---------------|-------|
| G01 | Simple metadata | `item_meta` | 10 |
| G02 | Review text | `review_text` | 10 |
| G03 | Computed metadata (hours) | `item_meta_hours` | 10 |
| G04 | Social signals (weighted reviews) | `review_text` + `weight_by` | 10 |
| G05 | Nested logic (AND/OR) | Mixed + nested ops | 10 |

### Metrics

| Metric | Role |
|--------|------|
| **Hits@5** | Primary (paper tables) |
| Accuracy | Secondary (diagnosis) |

### Baselines (Champions Only)

| Category | Method | Source |
|----------|--------|--------|
| Reasoning | Plan-and-Solve (ps) | oldsrc/methods/ps.py |
| Ranking | Listwise | oldsrc/methods/listwise.py |
| Structured | Weaver | oldsrc/methods/weaver.py |
| Standard | Zero-shot CoT | oldsrc/methods/cot.py |

### Attacks (3 Categories)

| Category | Attacks |
|----------|---------|
| Noise | typo, verbose, duplicate |
| Injection | inject_override, inject_system |
| Deception | fake_positive, fake_negative, sarcastic |

---

## Progress

### Phase 1: Documentation ✅
- [x] Create directory structure
- [x] Write doc/README.md
- [x] Write doc/research_plan.md
- [x] Write doc/evaluation_spec.md

### Phase 2: Data Pipeline ✅
- [x] Create preprocessing pipeline (preprocessing/curate.py, analyze.py)
- [x] Curate philly_cafes dataset (20 restaurants, 400 reviews)
- [x] Design 50 requests across 5 groups
- [x] Implement validator (data/validate.py)
- [x] Add G03 hours-based evidence (`item_meta_hours`)
- [x] Add G04 weighted review evidence (`weight_by`, `threshold`)
- [x] Validate all 50 requests (50/50 = 100%)
- [x] Document selection process (preprocessing/records/philly_cafes_selection.md)

### Phase 3: Code (Current)
Port methods from oldsrc/ to new structure:

```
methods/
├── __init__.py
├── base.py          # Method interface
├── cot.py           # Chain-of-Thought baseline
├── ps.py            # Plan-and-Solve
├── listwise.py      # Listwise ranking
├── weaver.py        # Structured reasoning
└── anot.py          # Our method
```

**Tasks:**
- [ ] Create methods/ directory with base interface
- [ ] Port cot.py (baseline)
- [ ] Port ps.py (reasoning baseline)
- [ ] Port listwise.py (ranking baseline)
- [ ] Port weaver.py (structured baseline)
- [ ] Create anot.py (our method)
- [ ] Create main.py runner with CLI args
- [ ] Create run.py evaluation loop

### Phase 4: Experiments
- [ ] Run baselines on philly_cafes (clean)
- [ ] Run baselines with attacks
- [ ] Run ANoT variations
- [ ] Collect results and analyze

---

## File Structure

```
anot/
├── main.py                 # Entry point
├── run.py                  # Evaluation loop
├── methods/                # Prompting methods
│   ├── base.py
│   ├── cot.py
│   ├── ps.py
│   ├── listwise.py
│   ├── weaver.py
│   └── anot.py
├── data/
│   ├── validate.py         # Request validator
│   └── philly_cafes/       # Dataset
│       ├── restaurants.jsonl
│       ├── reviews.jsonl
│       ├── requests.jsonl
│       └── groundtruth.jsonl
├── utils/
│   ├── llm.py              # LLM API wrapper
│   └── experiment.py       # Experiment manager
├── preprocessing/
│   ├── curate.py           # Data curation
│   ├── analyze.py          # Feature analysis
│   └── records/            # Selection documentation
├── doc/
│   ├── README.md
│   ├── research_plan.md
│   └── evaluation_spec.md
├── results/                # Experiment outputs
└── oldsrc/                 # Legacy code reference
```

---

## Next Steps

1. **Create methods/base.py** - Define method interface
2. **Port cot.py** - Simplest baseline first
3. **Create main.py** - CLI runner
4. **Test on 5 requests** - Verify pipeline works
5. **Port remaining baselines** - ps, listwise, weaver
6. **Implement anot.py** - Our method
