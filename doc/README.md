# Documentation Index

This directory contains all research and technical documentation for the ANoT project.

## Benchmark Dataset Documentation

The philly_cafes benchmark is the primary evaluation dataset. Documentation is split between methodology (here) and dataset-specific files (in `data/philly_cafes/`).

### Methodology & Design

| File | Purpose |
|------|---------|
| [condition_design.md](condition_design.md) | **Bottom-up anchor-first design methodology** - Key insights, evidence types, debugging multi-match issues |

### Dataset Reference (in `data/philly_cafes/`)

| File | Purpose |
|------|---------|
| [README.md](../data/philly_cafes/README.md) | Dataset overview, request group designs (G01-G08) |
| [requests_reference.md](../data/philly_cafes/requests_reference.md) | Complete request-answer reference (all 80 requests) |
| [statistics.md](../data/philly_cafes/statistics.md) | Evidence distribution, restaurant coverage, complexity analysis |
| [condition_summary.md](../data/philly_cafes/condition_summary.md) | Unique/rare conditions, restaurant identifiers |

### Supporting Files (in `data/philly_cafes/`)

| File | Purpose |
|------|---------|
| `condition_matrix.json` | Machine-readable condition satisfaction matrix |
| `groundtruth.jsonl` | Validation results for all 80 requests |

---

## Research Documentation

| File | Purpose |
|------|---------|
| [research_plan.md](research_plan.md) | Master research plan with task formulation, baselines, and specifications |
| [evaluation_spec.md](evaluation_spec.md) | Detailed evaluation protocol and method interface |
| [baselines.md](baselines.md) | Baseline methods with paper references |

## Architecture Documentation

| File | Purpose |
|------|---------|
| [anot_architecture.md](anot_architecture.md) | Three-phase ANoT design: Plan → Expand → Execute |
| [logging.md](logging.md) | Logging infrastructure: usage.jsonl and anot_trace.jsonl |

## Development Documentation

| File | Purpose |
|------|---------|
| [code_quality_audit.md](code_quality_audit.md) | Code health audit and refactoring status |
| [attack_plan.md](attack_plan.md) | Adversarial attack implementation plan (WIP) |
| [TODO.md](TODO.md) | Current and future tasks |

---

## Quick Reference

### Task
**Constraint-Satisfying Reranking** (Last-Mile RAG)

Given a user request with logical structure and N candidate restaurants, identify the one that satisfies all conditions.

### Benchmark Stats

| Metric | Value |
|--------|-------|
| Restaurants | 50 |
| Requests | 80 |
| Request Groups | 8 (G01-G08) |
| Validation Rate | 100% |

### Request Groups

| Group | Structure | Complexity | Requests |
|-------|-----------|------------|----------|
| G01 | Simple AND | Low | R00-R09 |
| G02 | Simple OR | Low-Medium | R10-R19 |
| G03 | AND-OR Combination | Medium | R20-R29 |
| G04 | Credibility-Count Weighting | Medium-High | R30-R39 |
| G05 | Triple OR with Anchor | Medium | R40-R49 |
| G06 | Nested OR+AND | High | R50-R59 |
| G07 | Chained OR | High | R60-R69 |
| G08 | Unbalanced Structure | High | R70-R79 |

### Evidence Types

| Type | Usage | Reliability |
|------|-------|-------------|
| `item_meta` | ~49% | High |
| `review_text` | ~43% | Medium |
| `review_meta` | ~5% | Medium |
| `item_meta_hours` | ~3% | High |

### Baselines

CoT, Plan-and-Solve, Plan-and-Act, Listwise, Weaver (see [baselines.md](baselines.md))

### Our Method

**ANoT** (Adaptive Network of Thought) - Three-phase adaptive evaluation (see [anot_architecture.md](anot_architecture.md))
