# SCALE Benchmark Documentation

## Document Index

| Document | Purpose | Paper Section |
|----------|---------|---------------|
| [OVERVIEW.md](OVERVIEW.md) | High-level benchmark description | Introduction |
| [TAXONOMY.md](TAXONOMY.md) | 4 Perspectives × 10 Groups structure | Task Taxonomy |
| [TASKS.md](TASKS.md) | All 100 task definitions | Appendix: Task Catalog |
| [TASK_DESIGN_GUIDELINES.md](TASK_DESIGN_GUIDELINES.md) | How to create new tasks | Methods: Task Design |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | Key architectural choices | Methods: Design Rationale |
| [ORDINAL_AUPRC.md](ORDINAL_AUPRC.md) | Scoring methodology | Methods: Evaluation |
| [DATA_STRATEGY.md](DATA_STRATEGY.md) | Dataset construction | Methods: Data |
| [DIMENSIONS.md](DIMENSIONS.md) | Evaluation dimensions | Results: Analysis |

## Quick Reference

### Task ID Format
```
G{group}{letter}  →  G1a, G2b, G10j
```

### GT Pipeline
```
dataset_K200.jsonl → keyword filter → LLM judgment → deterministic GT
(locked source)      (16% match)      (stored)        (Python formulas)
```

### Dynamic GT per K
```python
# GT computed from reviews 0 to K-1 only
gt = compute_gt_for_k("Restaurant Name", k=50)
```

### Directory Structure
```
doc/
├── README.md              # This index
├── OVERVIEW.md            # Benchmark overview
├── TAXONOMY.md            # Perspective/Group structure
├── TASKS.md               # 100 task definitions
├── TASK_DESIGN_GUIDELINES.md  # Task creation guide
├── DESIGN_DECISIONS.md    # Architectural decisions
├── ORDINAL_AUPRC.md       # Scoring methodology
├── DATA_STRATEGY.md       # Dataset approach
├── DIMENSIONS.md          # Evaluation dimensions
└── archive/               # Historical documents
```
