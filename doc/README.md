# Documentation Index

Comprehensive documentation for the ANoT evaluation framework.

## Documentation by Use Case

### Writing a Research Paper

- [Methodology](paper/methodology.md) - How the evaluation framework works
- [Data Pipeline](paper/data_pipeline.md) - How data flows through the system
- [ANoT Architecture](research/anot_architecture.md) - Three-phase design details
- [Baselines](research/baselines.md) - Implemented methods with citations

### Recreating the philly_cafes Dataset

- [Recreation Guide](guides/recreate_philly_cafes.md) - Step-by-step reproduction
- [Evidence Types](reference/evidence_types.md) - Validation logic reference
- [Request Structure](reference/request_structure.md) - Request JSON schema
- [Prompt Templates](../preprocessing/prompts/) - Claude Code prompts used

### Creating a New Benchmark

- [Benchmark Creation Guide](guides/create_new_benchmark.md) - Template for new datasets
- [Adding Evidence Types](guides/add_evidence_type.md) - Extending validation logic
- [Condition Design](research/condition_design.md) - Bottom-up anchor-first methodology

### Running Experiments

- [Experiments Guide](guides/run_experiments.md) - How to run evaluations
- [Evaluation Spec](research/evaluation_spec.md) - Metrics and protocols

---

## Directory Structure

```
doc/
├── paper/                # Research paper support
│   ├── methodology.md    # Evaluation framework design
│   └── data_pipeline.md  # Data flow documentation
│
├── research/             # Core research documentation
│   ├── anot_architecture.md    # Three-phase ANoT design
│   ├── baselines.md            # Method references
│   ├── evaluation_spec.md      # Evaluation protocol
│   ├── research_plan.md        # Master research plan
│   └── condition_design.md     # Benchmark design methodology
│
├── guides/               # How-to guides
│   ├── recreate_philly_cafes.md  # Dataset reproduction
│   ├── create_new_benchmark.md   # New dataset creation
│   ├── add_evidence_type.md      # Extending validation
│   └── run_experiments.md        # Running evaluations
│
├── reference/            # Technical reference
│   ├── evidence_types.md       # Evidence type specifications
│   ├── request_structure.md    # Request JSON schema
│   └── logging.md              # Output file formats
│
└── internal/             # Development documentation
    ├── TODO.md                 # Current tasks
    ├── attack_plan.md          # Attack implementation
    └── code_quality_audit.md   # Code health audit
```

---

## Quick Reference

### Task

**Constraint-Satisfying Reranking** (Last-Mile RAG)

Given a user request with logical structure and N candidate restaurants, identify the one that satisfies all conditions.

### Benchmark Stats

| Metric | Value |
|--------|-------|
| Restaurants | 50 |
| Requests | 100 |
| Request Groups | 10 (G01-G10) |
| Validation Rate | 100% |

### Request Groups

| Group | Structure | Requests |
|-------|-----------|----------|
| G01 | Simple AND | R01-R10 |
| G02 | Simple OR | R11-R20 |
| G03 | AND-OR Combination | R21-R30 |
| G04 | Credibility Weighting | R31-R40 |
| G05 | Triple OR + Anchor | R41-R50 |
| G06 | Nested OR+AND | R51-R60 |
| G07 | Chained OR | R61-R70 |
| G08 | Unbalanced Structure | R71-R80 |
| G09 | Direct Friends (1-hop) | R81-R90 |
| G10 | Social Circle (2-hop) | R91-R100 |

### Evidence Types

| Type | Description |
|------|-------------|
| `item_meta` | Attribute matching |
| `item_meta_hours` | Operating hours |
| `review_text` | Pattern matching |
| `review_meta` | Reviewer metadata |
| `social_filter` | Social graph filtering |

### Methods

| Method | Description |
|--------|-------------|
| `cot` | Chain-of-Thought |
| `ps` | Plan-and-Solve |
| `plan_act` | Plan-and-Act |
| `listwise` | Listwise Reranking |
| `weaver` | SQL+LLM Hybrid |
| `anot` | **Ours**: Adaptive Network of Thought |

---

## Dataset Documentation

See `data/philly_cafes/`:

| File | Purpose |
|------|---------|
| [README.md](../data/philly_cafes/README.md) | Dataset overview |
| [statistics.md](../data/philly_cafes/statistics.md) | Detailed statistics |
| [requests_reference.md](../data/philly_cafes/requests_reference.md) | Request-answer reference |
| [condition_summary.md](../data/philly_cafes/condition_summary.md) | Condition analysis |

---

## Preprocessing Documentation

See `preprocessing/`:

| File | Purpose |
|------|---------|
| [README.md](../preprocessing/README.md) | Pipeline overview |
| [prompts/](../preprocessing/prompts/) | Claude Code prompt templates |
| [records/](../preprocessing/records/) | Selection documentation |
