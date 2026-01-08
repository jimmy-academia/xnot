# Benchmark Design Methodology

> For the research paper experiment section: systematic design of structured recommendation benchmark requests.

This document provides a comprehensive methodology for benchmark design, consolidating principles from across the documentation. For implementation details, see the reference docs linked throughout.

---

## 1. Task Definition

**Constraint-Satisfying Reranking**: Given a natural language request with implicit logical constraints and N candidate items with structured attributes and unstructured reviews, identify the unique item satisfying all constraints.

### Benchmark Statistics (philly_cafes)

| Component | Count |
|-----------|-------|
| Requests | 100 (10 groups x 10 requests) |
| Candidate Items | 20 restaurants |
| Reviews | 500 total |
| Social Network | 40 users |

---

## 2. Design Principles

### 2.1 Uniqueness Constraint

Each request must match exactly ONE gold item. The validation system (`data/validate.py`) enforces this by evaluating all candidates against the formal structure.

**Resolution for multi-match**: Add distinguishing `item_meta` conditions (most reliable).

### 2.2 Machine Verifiability

All conditions must be programmatically verifiable without runtime human judgment:
- **Structured attributes**: Direct field lookup
- **Semantic conditions**: Pre-computed judgement cache
- **Social conditions**: Graph traversal algorithms

### 2.3 Semantic Judgement Caching

Semantic understanding (e.g., "mentions espresso positively") uses pre-computed judgements rather than runtime LLM calls or regex patterns.

**Rationale**:
1. Eliminates evaluation-time LLM dependency
2. Ensures reproducibility
3. Avoids brittle pattern matching
4. Enables human verification

**Cache format** (`judgement_cache.json`):
```json
{
  "{review_id}:{aspect}": {
    "review_id": "abc123",
    "business_id": "xyz789",
    "aspect": "espresso",
    "judgement": "positive",
    "note": "Reviewer praises espresso quality"
  }
}
```

### 2.4 Progressive Complexity

Request groups progress from simple to complex reasoning:

| Tier | Groups | Complexity |
|------|--------|------------|
| Basic | G01-G04 | Single evidence type, AND-only |
| Intermediate | G05-G08 | Nested AND/OR combinations |
| Advanced | G09-G10 | Social network integration |

### 2.5 Bottom-Up Anchor-First Design

Instead of top-down request design, we:
1. Build condition satisfaction matrix (which items satisfy which conditions)
2. Identify unique identifiers (minimal condition sets per item)
3. Design requests around anchors (add OR complexity for evaluation challenge)

> See [Condition Design Reference](../reference/condition_design.md) for detailed methodology.

---

## 3. Evidence Types

Seven evidence types with distinct evaluation semantics:

| Type | Purpose | Validation |
|------|---------|------------|
| `item_meta` | Structured attributes | Field lookup |
| `item_meta_hours` | Operating hours | Time range containment |
| `review_text` | Pattern in reviews | Judgement cache lookup |
| `review_sentiment` | Semantic sentiment | Judgement cache + sentiment |
| `review_meta` | Reviewer metadata | Aggregation (any/all/count) |
| `review_group_rating` | Aggregate ratings | Filtered subset statistics |
| `social_rating` | Social network ratings | Graph traversal + rating |

**Distribution in philly_cafes**:
- `item_meta`: 341 conditions (88%)
- `review_sentiment`: 14 (4%)
- `review_text`: 12 (3%)
- `item_meta_hours`: 10 (3%)
- `review_group_rating`: 7 (2%)
- `social_rating`: 4 (1%)

> See [Evidence Types Reference](../reference/evidence_types.md) for schemas and examples.

---

## 4. Request Groups

### G01-G04: Flat AND Logic

| Group | Evidence Types | Example Shorthand |
|-------|---------------|-------------------|
| G01 | `item_meta` only | `AND(drive_thru, good_for_kids, no_tv)` |
| G02 | `item_meta` + `review_text` | `AND(gelato_reviews, mid_price, takeout)` |
| G03 | `item_meta` + `item_meta_hours` | `AND(monday_early, quiet, budget)` |
| G04 | `item_meta` + `review_group_rating` | `AND(outdoor_no, post_2020_avg_gte_4.0)` |

### G05-G08: Nested Boolean Logic

| Group | Structure | Example |
|-------|-----------|---------|
| G05 | `AND(anchors, OR(a,b,c))` | 4-way OR |
| G06 | `AND(anchors, OR(AND(a,b), AND(c,d)))` | OR of ANDs |
| G07 | `AND(anchors, OR(a,b), OR(c,d))` | Parallel ORs |
| G08 | `AND(anchors, OR(simple, AND(a,b)))` | Unbalanced OR |

### G09-G10: Social Network Queries

| Group | Hops | Semantics |
|-------|------|-----------|
| G09 | 1 | Anchor + direct friends |
| G10 | 2 | Anchor + friends + friends-of-friends |

**Text format**:
```
(User: NAME; friends: [FRIEND1, FRIEND2, ...]) <query>
```

---

## 5. Request Structure Format

### JSON Schema

```json
{
  "id": "R01",
  "group": "G01",
  "scenario": "Family Outing",
  "text": "Looking for a family-friendly cafe with TVs and drive-thru",
  "shorthand": "AND(good_for_kids, has_tv, drive_thru)",
  "structure": {
    "op": "AND",
    "args": [
      {"aspect": "good_for_kids", "evidence": {"kind": "item_meta", ...}},
      {"aspect": "has_tv", "evidence": {"kind": "item_meta", ...}},
      {"aspect": "drive_thru", "evidence": {"kind": "item_meta", ...}}
    ]
  },
  "gold_restaurant": "business_id_here"
}
```

> See [Request Structure Reference](../reference/request_structure.md) for complete schema.

---

## 6. Design Procedure

### Phase 1: Data Preparation

1. Curate item set (15-25 items with diverse attributes)
2. Curate review set (20-30 reviews per item)
3. Build social graph (for G09-G10)
4. Generate judgement cache (all semantic patterns)

### Phase 2: Condition Matrix Generation

```bash
python data/scripts/condition_matrix.py
```

Outputs:
- Unique conditions (count=1): Direct anchors
- Rare conditions (count=2-3): Need one disambiguator
- Common conditions (count>10): Cannot anchor alone

### Phase 3: Request Design

For each request:
1. Select gold item with distinctive features
2. Identify minimal unique condition set (anchor)
3. Add OR complexity for evaluation challenge
4. Write natural language text reflecting structure
5. Run validation

### Phase 4: Validation & Refinement

```bash
python -m data.validate <dataset>
```

**Target**: 100% pass rate (all `ok` status)

**Error resolution**:
| Status | Fix |
|--------|-----|
| `multi_match` | Add distinguishing condition |
| `no_match` | Relax conditions or fix evidence |
| `wrong_gold` | Fix anchor targeting |

> See [Create New Benchmark Guide](../guides/create_new_benchmark.md) for step-by-step instructions.

---

## 7. LLM-Assisted Generation

### Automatable Steps

| Step | LLM Role |
|------|----------|
| Attribute analysis | Suggest distinguishing combinations |
| Natural language generation | Convert structures to text |
| Judgement cache population | Determine sentiment from reviews |

### Human-Required Steps

| Step | Why Human |
|------|-----------|
| Gold item strategy | Benchmark balance understanding |
| Validation interpretation | Multi-match resolution decisions |
| Quality review | Text naturalness verification |

### Recommended Workflow

1. Human: Define groups and complexity targets
2. LLM: Generate candidate condition sets
3. Human: Select and refine
4. LLM: Generate natural language
5. Human: Validate, identify failures
6. LLM: Suggest multi-match fixes
7. Human: Apply fixes, re-validate
8. LLM: Populate judgement cache
9. Human: Final verification

---

## 8. Extending to New Domains

### Required Files

```
data/<domain>/
├── restaurants.jsonl       # Items
├── reviews.jsonl           # Reviews
├── requests.jsonl          # Requests
├── user_mapping.json       # Social graph (optional)
├── judgement_cache.json    # Semantic judgements
└── groundtruth.jsonl       # Generated by validation
```

### Adaptation Checklist

- [ ] Map domain attributes to evidence paths
- [ ] Define domain-specific semantic topics
- [ ] Build social graph (if using G09-G10)
- [ ] Populate judgement cache
- [ ] Validate 100% pass rate

> See [Create New Benchmark Guide](../guides/create_new_benchmark.md) for detailed instructions.

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [Evidence Types](../reference/evidence_types.md) | Complete evidence type specifications |
| [Request Structure](../reference/request_structure.md) | JSON schema reference |
| [Condition Design](../reference/condition_design.md) | Bottom-up anchor methodology |
| [Create New Benchmark](../guides/create_new_benchmark.md) | Step-by-step guide |
| [Recreate philly_cafes](../guides/recreate_philly_cafes.md) | Dataset reproduction |

---

*This document synthesizes design principles for research paper documentation. Implementation details are in the referenced documents.*
