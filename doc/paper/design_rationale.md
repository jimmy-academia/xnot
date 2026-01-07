# Design Rationale

This document captures design decisions for the benchmark and evaluation framework.

---

## Request Groups (G01-G10)

100 requests organized into 10 groups of 10 requests each.

### Flat AND Groups (G01-G04)

These groups test flat AND logic with different evidence type combinations.

| Group | Requests | Logic | Evidence Types |
|-------|----------|-------|----------------|
| G01 | R01-R10 | Flat AND | item_meta only |
| G02 | R11-R20 | Flat AND | item_meta + review_text |
| G03 | R21-R30 | Flat AND | item_meta + item_meta_hours |
| G04 | R31-R40 | Flat AND | item_meta + review_text (weighted by reviewer) |

**G04 Detail**: Conditions like `coffee_by_experts` use review_text with `weight_by` field to weight by reviewer experience/popularity.

### Complex Logic Groups (G05-G08)

These groups test nested logical structures.

| Group | Requests | Logic Structure |
|-------|----------|-----------------|
| G05 | R41-R50 | AND with one OR group: `AND(cond, cond, OR(...))` |
| G06 | R51-R60 | AND with OR containing nested ANDs: `AND(cond, OR(AND(...), AND(...)))` |
| G07 | R61-R70 | AND with multiple parallel ORs: `AND(cond, OR(...), OR(...))` |
| G08 | R71-R80 | AND with OR containing AND: `AND(cond, OR(cond, AND(...)))` |

### Social Filter Groups (G09-G10)

These groups test social graph traversal.

| Group | Requests | Logic |
|-------|----------|-------|
| G09 | R81-R90 | 1-hop: "My friend [Name] mentioned [keyword]" |
| G10 | R91-R100 | 2-hop: "[Name] or their friends mentioned [keyword]" |

**Reference**: [data/philly_cafes/requests.jsonl](../../data/philly_cafes/requests.jsonl)

---

## Evidence Types

### Technical Implementation

Three evidence kinds are used in the code:

| Kind | Count | Description |
|------|-------|-------------|
| `item_meta` | 190 | Restaurant attributes (DriveThru, GoodForKids, Ambience, etc.) |
| `item_meta_hours` | 11 | Operating hours checks (monday_early, friday_late, etc.) |
| `review_text` | 207 | Text pattern matching in reviews (with optional weight_by) |

### Logical Classification

For analysis purposes, conditions are classified as:

| Type | Count | Description |
|------|-------|-------------|
| item_meta | 49 | Direct attribute lookup |
| item_meta_hours | 8 | Hours-based conditions |
| review_meta | 24 | Reviewer credibility weighting (technically coded as review_text with weight_by) |
| review_text | 34 | Plain text pattern matching |

**Reference**: [doc/reference/evidence_types.md](../reference/evidence_types.md), [data/philly_cafes/condition_summary.md](../../data/philly_cafes/condition_summary.md)

---

## Scaling Experiment

Tests performance across candidate set sizes.

| Parameter | Values |
|-----------|--------|
| Candidate counts (N) | 10, 20, 30, 40, 50 |
| Gold position | Shuffled per request |
| Metric | Hits@K for K ∈ {1, 3, 5} |

**Reference**: [doc/guides/run_experiments.md](../guides/run_experiments.md), [run/scaling.py](../../run/scaling.py)

---

## Shuffle Strategies

Three strategies for gold item positioning:

| Strategy | Behavior |
|----------|----------|
| `none` | Gold item at original position (first) |
| `middle` | Gold item at position N/2 |
| `random` | Gold item at random position (per-request seed) |

Default: `random`

**Reference**: [run/shuffle.py](../../run/shuffle.py)

---

## Results Directory Structure

### Dev Mode

- Path: `results/dev/{NNN}_{run-name}/`
- Gitignored
- Auto-numbered (001, 002, ...)
- Enabled with `--dev` flag

### Benchmark Mode

- Path: `results/benchmarks/{method}_{data}/{attack}/run_{N}/`
- Git-tracked
- Named by configuration
- Default mode

**Reference**: [doc/reference/configuration.md](../reference/configuration.md), [utils/experiment.py](../../utils/experiment.py)

---

## Methods Architecture

### Package vs Single File

| Structure | Used By | Characteristics |
|-----------|---------|-----------------|
| Package (`methods/anot/`) | anot | Multi-phase, shared state, multiple files |
| Single file (`methods/cot.py`) | cot, ps, listwise, weaver, react | Single-pass, self-contained |

### String Mode vs Dict Mode

| Mode | Methods | Context Handling |
|------|---------|------------------|
| String mode | cot, ps, plan_act, listwise | Pack-to-budget truncation applied |
| Dict mode | anot, weaver, react | Full data dict, selective access |

**Reference**: [doc/paper/anot_architecture.md](anot_architecture.md), [data/loader.py](../../data/loader.py)

---

## Pack-to-Budget Truncation

For string-mode methods only.

### Policy

1. Include all restaurants with metadata (ensures fair evaluation)
2. Add reviews round-robin until token budget exhausted
3. Budget determined by model's input token limit

### Implementation

- Uses tiktoken for token counting
- Coverage stats included in results

**Reference**: [data/loader.py](../../data/loader.py)

---

## Defense Mode

Enabled with `--defense` flag.

### Supported Methods

5 methods support defense mode:
- cot, plan_act, listwise, weaver, anot

### Implementation

Defense preamble added to system prompts.

**Reference**: [doc/reference/defense_mode.md](../reference/defense_mode.md), [methods/shared.py](../../methods/shared.py)

---

## Attack System

12 attack types targeting non-gold items only.

### Attack Categories

| Category | Attacks |
|----------|---------|
| Noise | typo_10, typo_20, heterogeneity |
| Injection | inject_override, inject_fake_sys, inject_promotion |
| Fake Review | fake_positive, fake_negative, sarcastic_wifi, sarcastic_noise, sarcastic_outdoor, sarcastic_all |

### Key Constraint

Gold items are NEVER attacked (fair evaluation preserved).

**Reference**: [doc/reference/attacks.md](../reference/attacks.md), [attack.py](../../attack.py)
