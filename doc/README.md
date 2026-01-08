# Documentation Index

## 3-Level Hierarchy

```
Level 1: README.md (Project Entry)
    |
Level 2: doc/README.md (This Index) + Category Directories
    |
Level 3: Individual Documents
```

---

## Level 2: Documentation Categories

### Research Paper (`paper/`)

Documents supporting the research paper.

| Document | Purpose |
|----------|---------|
| [Benchmark Design](paper/benchmark_design.md) | **Experiment section**: Design methodology, principles, procedures |
| [Methodology](paper/methodology.md) | Evaluation framework design |
| [ANoT Architecture](paper/anot_architecture.md) | Three-phase agent design |
| [Baselines](paper/baselines.md) | Implemented methods with citations |
| [Data Pipeline](paper/data_pipeline.md) | Data flow documentation |
| [Design Rationale](paper/design_rationale.md) | High-level design decisions |

### How-To Guides (`guides/`)

Step-by-step instructions for common tasks.

| Document | Purpose |
|----------|---------|
| [Run Experiments](guides/run_experiments.md) | Execute and analyze evaluations |
| [Create New Benchmark](guides/create_new_benchmark.md) | Build a new evaluation dataset |
| [Recreate philly_cafes](guides/recreate_philly_cafes.md) | Reproduce from raw Yelp data |
| [Add Evidence Type](guides/add_evidence_type.md) | Extend validation logic |
| [Methods Architecture](guides/architecture.md) | Method structure conventions |

### Technical Reference (`reference/`)

Specifications and schemas.

| Document | Purpose |
|----------|---------|
| [Evidence Types](reference/evidence_types.md) | All 7 evidence type specifications |
| [Request Structure](reference/request_structure.md) | Request JSON schema |
| [Condition Design](reference/condition_design.md) | Bottom-up anchor-first methodology |
| [Configuration](reference/configuration.md) | CLI arguments, LLM config |
| [Attacks](reference/attacks.md) | Adversarial attack system |
| [Defense Mode](reference/defense_mode.md) | Attack-resistant evaluation |
| [Logging](reference/logging.md) | Output file formats |

### Development (`internal/`)

Internal development documentation.

| Document | Purpose |
|----------|---------|
| [TODO](internal/TODO.md) | Current development tasks |
| [Attack Plan](internal/attack_plan.md) | Attack implementation status |
| [Code Quality Audit](internal/code_quality_audit.md) | Code health audit |

---

## Use-Case Navigation

### Writing a Research Paper

1. Start with [Benchmark Design](paper/benchmark_design.md) for experiment methodology
2. Reference [Methodology](paper/methodology.md) for evaluation framework
3. See [Baselines](paper/baselines.md) for method citations
4. Check [ANoT Architecture](paper/anot_architecture.md) for our method details

### Reproducing the Benchmark

1. [Recreate philly_cafes](guides/recreate_philly_cafes.md) - Step-by-step reproduction
2. [Evidence Types](reference/evidence_types.md) - Validation logic reference
3. [Request Structure](reference/request_structure.md) - Request format

### Creating a New Benchmark

1. [Create New Benchmark](guides/create_new_benchmark.md) - Template and steps
2. [Condition Design](reference/condition_design.md) - Design methodology
3. [Add Evidence Type](guides/add_evidence_type.md) - Extending validation

### Running Experiments

1. [Run Experiments](guides/run_experiments.md) - Execution guide
2. [Configuration](reference/configuration.md) - CLI reference
3. [Attacks](reference/attacks.md) - Adversarial testing

### Understanding the Methods

1. [Methods Architecture](guides/architecture.md) - Structure conventions
2. [ANoT Architecture](paper/anot_architecture.md) - Our method details
3. [Baselines](paper/baselines.md) - Comparison methods

---

## Dataset Documentation

Located in `data/philly_cafes/`:

| File | Purpose |
|------|---------|
| [README.md](../data/philly_cafes/README.md) | Dataset overview |
| [statistics.md](../data/philly_cafes/statistics.md) | Detailed statistics |
| [requests_reference.md](../data/philly_cafes/requests_reference.md) | Request-answer reference |
| [condition_summary.md](../data/philly_cafes/condition_summary.md) | Condition analysis |

---

## Preprocessing Documentation

Located in `preprocessing/`:

| File | Purpose |
|------|---------|
| [README.md](../preprocessing/README.md) | Pipeline overview |
| [prompts/](../preprocessing/prompts/) | Claude Code prompt templates |
| [records/](../preprocessing/records/) | Selection documentation |

---

## Quick Reference

### Benchmark Stats

| Metric | Value |
|--------|-------|
| Restaurants | 20 |
| Requests | 100 |
| Groups | 10 (G01-G10) |
| Validation | 100% |

### Request Groups

| Group | Logic | Evidence |
|-------|-------|----------|
| G01-G04 | Flat AND | Various types |
| G05-G08 | Nested AND/OR | item_meta |
| G09-G10 | Social filter | review_sentiment + social |

### Evidence Types

| Type | Count | Description |
|------|-------|-------------|
| `item_meta` | 341 | Attribute matching |
| `review_sentiment` | 14 | Semantic sentiment |
| `review_text` | 12 | Pattern matching |
| `item_meta_hours` | 10 | Operating hours |
| `review_group_rating` | 7 | Aggregate ratings |
| `social_rating` | 4 | Social network ratings |

---

## Document Map

```
doc/
├── README.md                      # This index
│
├── paper/                         # Research Paper Support
│   ├── benchmark_design.md        # Experiment methodology (NEW)
│   ├── methodology.md             # Evaluation framework
│   ├── anot_architecture.md       # Three-phase design
│   ├── baselines.md               # Method references
│   ├── data_pipeline.md           # Data flow
│   └── design_rationale.md        # High-level rationale
│
├── guides/                        # How-To Guides
│   ├── run_experiments.md         # Running evaluations
│   ├── create_new_benchmark.md    # New dataset creation
│   ├── recreate_philly_cafes.md   # Dataset reproduction
│   ├── add_evidence_type.md       # Extending validation
│   └── architecture.md            # Methods architecture
│
├── reference/                     # Technical Reference
│   ├── evidence_types.md          # Evidence specifications
│   ├── request_structure.md       # Request schema
│   ├── condition_design.md        # Design methodology
│   ├── configuration.md           # CLI/LLM config
│   ├── attacks.md                 # Attack system
│   ├── defense_mode.md            # Defense mode
│   └── logging.md                 # Output formats
│
└── internal/                      # Development
    ├── TODO.md                    # Tasks
    ├── attack_plan.md             # Attack status
    └── code_quality_audit.md      # Code health
```
