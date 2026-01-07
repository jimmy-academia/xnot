# Methodology

How the evaluation framework works, for research paper documentation.

## Task Formulation

### Constraint-Satisfying Reranking

Given:
- **N candidates**: Items with structured metadata and unstructured text (reviews)
- **Request**: Natural language query with implicit logical constraints
- **Ground truth**: Exactly 1 candidate satisfies all constraints

Task: Rank candidates such that the satisfying candidate appears at rank 1.

### Formal Definition

```
Input:
  - C = {c₁, c₂, ..., cₙ}  (candidates)
  - q (natural language request)
  - For each cᵢ:
    - M(cᵢ): structured metadata (attributes, hours)
    - R(cᵢ): unstructured text (reviews)

Output:
  - π: C → {1, ..., N}  (ranking permutation)

Objective:
  - π(c*) = 1, where c* is the unique satisfying candidate
```

### Why This Task?

1. **Multi-source reasoning**: Requires combining structured and unstructured data
2. **Constraint verification**: Must verify logical conditions, not just similarity
3. **Single correct answer**: Enables precise evaluation (Hits@1)
4. **Realistic scenario**: Mirrors last-mile RAG reranking

---

## Evaluation Protocol

### Method Interface

Methods inherit from `BaseMethod` and implement `evaluate_ranking()`:

```python
class MyMethod(BaseMethod):
    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """
        Rank candidates based on constraint satisfaction.

        Args:
            query: User request with constraints
            context: Formatted candidate data (N items × reviews each)
            k: Number of top predictions to return

        Returns:
            String with top-k indices (e.g., "5" or "5, 2, 1")
        """
```

See [doc/guides/architecture.md](../guides/architecture.md) for full interface details.

### Metrics

**Primary: Hits@K**
```
Hits@K = (# requests where gold in top K) / (# total requests)
```

**Secondary: Mean Reciprocal Rank (MRR)**
```
MRR = (1/N) × Σ (1 / rank(gold))
```

### Candidate Scaling

Methods are evaluated across candidate pool sizes:
- N ∈ {10, 20, 30, 40, 50}

This tests scalability and robustness to distractor candidates.

---

## Method Categories

### Baseline Methods

| Method | Approach | Key Characteristic |
|--------|----------|-------------------|
| CoT | Chain-of-Thought | Sequential reasoning |
| Plan-and-Solve | Decomposition | Explicit planning |
| Plan-and-Act | ReAct-style | Tool-augmented |
| Listwise | Direct ranking | Pairwise comparison |
| Weaver | SQL+LLM | Structured queries |

### Our Method: ANoT

Adaptive Network of Thought - three-phase evaluation:

1. **Planning**: Extract conditions, prune candidates
2. **Expansion**: ReAct loop to expand evaluation plan
3. **Execution**: Parallel DAG execution

See [ANoT Architecture](../research/anot_architecture.md) for implementation details.

---

## Method Implementation

### Code Structure

```
methods/
├── cot.py          # Chain-of-Thought
├── ps.py           # Plan-and-Solve
├── plan_act.py     # Plan-and-Act
├── listwise.py     # Listwise Reranking
├── weaver.py       # SQL+LLM Hybrid
└── anot/           # ANoT package
    ├── core.py     # Main class
    ├── helpers.py  # Utilities
    ├── prompts.py  # Prompt templates
    └── tools.py    # Phase 2 tools
```

### Adding a New Method

1. Create `methods/new_method.py`
2. Implement `create_method(args)` factory
3. Register in `utils/arguments.py` METHOD_CHOICES
4. Method must return callable matching interface

---

## Evaluation Pipeline

### Execution Flow

```
main.py
   │
   ├── parse_args()           # CLI argument parsing
   ├── config_llm()           # LLM configuration
   ├── create_experiment()    # Results directory setup
   │
   └── run_single() or run_scaling_experiment()
          │
          ├── load_dataset()         # Load restaurants, reviews, requests
          ├── filter_by_candidates() # Sample N candidates per request
          ├── shuffle_candidates()   # Position bias mitigation
          │
          └── run_evaluation_loop()  # Per-request evaluation
                 │
                 ├── format_query()   # Format candidates for LLM
                 ├── method()         # Call evaluation method
                 └── evaluate_ranking() # Compute metrics
```

### Code Locations

| Component | File | Function |
|-----------|------|----------|
| Entry point | `main.py` | `main()` |
| Argument parsing | `utils/arguments.py` | `parse_args()` |
| Dataset loading | `data/loader.py` | `load_dataset()` |
| Orchestration | `run/orchestrate.py` | `run_single()`, `run_evaluation_loop()` |
| Metrics | `run/evaluate.py` | `evaluate_ranking()`, `compute_multi_k_stats()` |
| Scaling | `run/scaling.py` | `run_scaling_experiment()` |
| Shuffling | `run/shuffle.py` | `shuffle_candidates()` |

---

## Benchmark Design

### Request Groups

10 groups testing different reasoning capabilities:

| Group | Structure | Capability Tested |
|-------|-----------|-------------------|
| G01 | `AND(a, b, c)` | Basic conjunction |
| G02 | `AND(anchor, OR(a, b))` | Disjunctive reasoning |
| G03 | `AND(a, OR(b, c))` | Nested logic |
| G04 | `AND(a, review_meta_*)` | Credibility weighting |
| G05 | `AND(a, OR(b, c, d))` | Multi-way disjunction |
| G06 | `AND(a, OR(AND(b,c), AND(d,e)))` | Nested structures |
| G07 | `AND(a, OR(b,c), OR(d,e))` | Chained disjunctions |
| G08 | `AND(a, OR(b, AND(c,d)))` | Unbalanced structures |
| G09 | `1HOP([friends], pattern)` | Direct social filter |
| G10 | `2HOP([friends], pattern)` | Extended social filter |

### Evidence Types

5 evidence types for condition evaluation:

| Type | Source | Example |
|------|--------|---------|
| `item_meta` | Attributes | WiFi=free |
| `item_meta_hours` | Hours | Open Monday 9-17 |
| `review_text` | Reviews | Pattern "cozy" found |
| `review_meta` | Reviewer info | Elite status |
| `social_filter` | Social graph | Friend mentions |

See [Evidence Types Reference](../reference/evidence_types.md) for specifications.

### Validation

Ground truth is validated programmatically:
- Each request matches exactly 1 restaurant
- `data/validate.py` implements three-value logic evaluation
- 100% validation rate required

---

## Attack Framework

### Attack Categories

| Category | Attacks | Target |
|----------|---------|--------|
| Noise | typo_10, typo_20 | Input quality |
| Injection | inject_override, inject_fake_sys, inject_hidden | Instruction following |
| Deception | fake_positive, fake_negative, sarcastic_* | Content trust |

### Attack Implementation

```python
# attack.py
def apply_attack(data, attack_type, config):
    """
    Apply attack to candidate data.

    Args:
        data: Original candidate data
        attack_type: Attack identifier
        config: Attack parameters

    Returns:
        Modified data with attack applied
    """
```

### Robustness Evaluation

1. Run clean baseline (no attack)
2. Run with each attack type
3. Compare Hits@1 degradation
4. Measure with/without defense prompts

---

## Results Storage

### Output Structure

```
results/
├── dev/                    # Development runs (gitignored)
│   └── {NNN}_{run-name}/
│       ├── config.json     # Run configuration
│       ├── results_1.jsonl # Per-request results
│       ├── usage.jsonl     # Token usage
│       └── debug.log       # Full traces
│
└── benchmarks/             # Benchmark runs (tracked)
    └── {run-name}/
        └── ...
```

### Result Schema

**results_N.jsonl**:
```json
{
  "request_id": "R01",
  "gold_idx": 5,
  "prediction": 5,
  "rank": 1,
  "hit_at_1": true,
  "hit_at_5": true,
  "latency_ms": 2340,
  "tokens_in": 1500,
  "tokens_out": 50
}
```

**usage.jsonl**:
```json
{
  "run": 1,
  "total_tokens_in": 150000,
  "total_tokens_out": 5000,
  "total_latency_ms": 234000,
  "cost_usd": 0.25
}
```

---

## Reproducibility

### Configuration Capture

Each run saves:
- Command line arguments
- LLM configuration (model, temperature, max_tokens)
- Random seeds
- Dataset version
- Timestamp

### Determinism

For reproducible results:
- `--seed 42` for attack randomization
- `--temperature 0.0` for LLM determinism
- `--shuffle none` to disable position shuffling

---

## Related Documentation

- [ANoT Architecture](../research/anot_architecture.md) - Three-phase design
- [Baselines](../research/baselines.md) - Method references
- [Evidence Types](../reference/evidence_types.md) - Condition evaluation
- [Request Structure](../reference/request_structure.md) - Request JSON schema
- [Running Experiments](../guides/run_experiments.md) - Practical guide
