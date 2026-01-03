# Documentation Index

This directory contains all research documentation for the ANoT project.

## Documents

| File | Purpose |
|------|---------|
| [research_plan.md](research_plan.md) | Master research plan with task formulation, baselines, and specifications |
| [evaluation_spec.md](evaluation_spec.md) | Detailed evaluation protocol and method interface |
| [anot_architecture.md](anot_architecture.md) | Three-phase ANoT design: Explore → Expand → Execute |
| [logging.md](logging.md) | Logging infrastructure: usage.jsonl and anot_trace.jsonl |
| [TODO.md](TODO.md) | Current and future tasks |

## Quick Reference

**Task:** Constraint-Satisfying Reranking (Last-Mile RAG)

**Evidence Budget:** 20 candidates × 20 reviews

**Primary Metric:** Hits@5

**Request Groups:**
- G01: Simple metadata (10)
- G02: Review text (10)
- G03: Computed metadata (10)
- G04: Social signals (10)
- G05: Nested logic (10)

**Baselines:** Plan-and-Solve, Listwise, Weaver, Zero-shot CoT

**Our Method:** ANoT (Adaptive Network of Thought) - Three-phase adaptive evaluation
