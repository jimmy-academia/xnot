# Documentation Index

Comprehensive documentation for the ANoT evaluation framework.

## Documentation by Use Case

### Writing a Research Paper

- [Design Rationale](paper/design_rationale.md) - Benchmark design decisions
- [Methodology](paper/methodology.md) - How the evaluation framework works
- [Data Pipeline](paper/data_pipeline.md) - How data flows through the system
- [ANoT Architecture](paper/anot_architecture.md) - Three-phase design details
- [Baselines](paper/baselines.md) - Implemented methods with citations

### Recreating the philly_cafes Dataset

- [Recreation Guide](guides/recreate_philly_cafes.md) - Step-by-step reproduction
- [Evidence Types](reference/evidence_types.md) - Validation logic reference
- [Request Structure](reference/request_structure.md) - Request JSON schema
- [Prompt Templates](../preprocessing/prompts/) - Claude Code prompts used

### Creating a New Benchmark

- [Benchmark Creation Guide](guides/create_new_benchmark.md) - Template for new datasets
- [Adding Evidence Types](guides/add_evidence_type.md) - Extending validation logic
- [Condition Design](reference/condition_design.md) - Bottom-up anchor-first methodology

### Running Experiments

- [Experiments Guide](guides/run_experiments.md) - How to run evaluations
- [Configuration](reference/configuration.md) - CLI arguments, LLM config
- [Attacks](reference/attacks.md) - Adversarial attack system
- [Defense Mode](reference/defense_mode.md) - Attack-resistant evaluation

### Understanding Methods

- [Methods Architecture](guides/architecture.md) - Method structure conventions
- [Design Rationale](paper/design_rationale.md) - Method choices explained

### Development

- [TODO](internal/TODO.md) - Current development tasks
- [Attack Plan](internal/attack_plan.md) - Attack implementation status
- [Code Quality Audit](internal/code_quality_audit.md) - Code health audit

---

## Directory Structure

```
doc/
├── paper/                # Research paper support
│   ├── design_rationale.md   # Benchmark design decisions
│   ├── methodology.md        # Evaluation framework design
│   ├── data_pipeline.md      # Data flow documentation
│   ├── anot_architecture.md  # Three-phase ANoT design
│   └── baselines.md          # Method references
│
├── guides/               # How-to guides
│   ├── run_experiments.md        # Running evaluations
│   ├── architecture.md           # Methods architecture
│   ├── recreate_philly_cafes.md  # Dataset reproduction
│   ├── create_new_benchmark.md   # New dataset creation
│   └── add_evidence_type.md      # Extending validation
│
├── reference/            # Technical reference
│   ├── configuration.md      # CLI args, LLM config
│   ├── attacks.md            # Attack system
│   ├── defense_mode.md       # Defense mode
│   ├── evidence_types.md     # Evidence type specifications
│   ├── request_structure.md  # Request JSON schema
│   ├── condition_design.md   # Benchmark design methodology
│   └── logging.md            # Output file formats
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

| Group | Structure | Description |
|-------|-----------|-------------|
| G01 | `AND(conds)` | Flat AND, item_meta only |
| G02 | `AND(conds)` | Flat AND, item_meta + review_text |
| G03 | `AND(conds)` | Flat AND, item_meta + item_meta_hours |
| G04 | `AND(conds)` | Flat AND, item_meta + review_text (weighted) |
| G05 | `AND(anchors, OR(a,b,c,d))` | 4-way OR |
| G06 | `AND(anchors, OR(AND(a,b), AND(c,d)))` | OR of two ANDs |
| G07 | `AND(anchors, OR(a,b), OR(c,d))` | Two parallel ORs |
| G08 | `AND(anchors, OR(simple, AND(a,b)))` | Unbalanced OR |
| G09 | `1HOP(['Name'], 'pattern')` | 1-hop social filter (anchor + friends) |
| G10 | `2HOP(['Name'], 'pattern')` | 2-hop social filter (+ friends-of-friends) |

**Note**: Structure uses only `AND`/`OR` operators. Negation is handled at evidence level (`"true": "False"`, `"not_contains"`).

### Evidence Types

| Type | Description |
|------|-------------|
| `item_meta` | Attribute matching |
| `item_meta_hours` | Operating hours |
| `review_text` | Pattern matching in reviews |
| `review_meta` | Reviewer credibility weighting |
| `social_filter` | Social graph filtering |

### Methods

| Method | Description | Defense |
|--------|-------------|---------|
| `cot` | Chain-of-Thought | Yes |
| `ps` | Plan-and-Solve | No |
| `plan_act` | Plan-and-Act | Yes |
| `listwise` | Listwise Reranking | Yes |
| `weaver` | SQL+LLM Hybrid | Yes |
| `anot` | **Ours**: Adaptive Network of Thought | Yes |

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
