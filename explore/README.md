# General ANoT Exploration

Task-agnostic Adaptive Network of Thought for semantic reasoning tasks.

## Architecture

```
explore/
├── general_anot/           # Main framework
│   ├── phase1.py           # Formula -> Formula Seed (LLM compilation)
│   ├── phase2.py           # Formula Seed interpreter (execution)
│   ├── eval.py             # Full evaluation with AUPRC scoring
│   ├── FORMULA_SEED_SPEC.md  # Specification documentation
│   └── DESIGN.md           # Design principles
│
├── baselines/              # Baseline methods for comparison
│   ├── direct_llm_v1.py    # Direct LLM (V1 formula)
│   ├── direct_llm_v2.py    # Direct LLM (V2 formula)
│   └── cot.py              # Chain of Thought
│
├── scoring/                # Evaluation utilities
│   ├── auprc.py            # Ordinal AUPRC scoring
│   └── ground_truth.py     # Deterministic GT computation
│
├── tasks/                  # Task definitions
│   └── g1_allergy.py       # Peanut allergy safety (G1a, G1a-v2)
│
├── data/                   # Datasets
│   ├── dataset_K200.jsonl  # 100 restaurants, 13K+ reviews
│   └── semantic_gt/        # Stored per-review LLM judgments
│
├── results/                # Evaluation results
│   ├── general_anot_eval/  # Current eval results
│   └── phase1_v2/          # Current Formula Seed
│
├── tools/                  # Data preparation tools
│
├── doc/                    # Additional documentation
│
└── archive/                # Archived/outdated code
```

## Benchmark Results (G1a-v2 Task)

### Method Comparison

| Method | Adjusted AUPRC | Range | Consistency |
|--------|----------------|-------|-------------|
| **General ANoT** | **0.82 avg** | 0.74-0.87 | Stable |
| Direct LLM (zero-shot) | 0.53 avg | 0.43-0.67 | Variable |
| Chain of Thought | 0.49 avg | 0.26-0.61 | Highly variable |

### Detailed Metrics (General ANoT)

| Metric | Value |
|--------|-------|
| Ordinal AUPRC | 0.95 avg |
| Primitive Accuracy | 0.87 avg |
| **Adjusted AUPRC** | **0.82 avg** |
| Verdict Accuracy | 98% |
| Time (100 restaurants) | ~70s |

### Why General ANoT Outperforms Baselines

| Factor | General ANoT | Baselines |
|--------|-------------|-----------|
| **Extraction** | Structured per-review with explicit criteria | LLM interprets entire context at once |
| **Computation** | Deterministic formulas (Python math) | LLM computes everything including arithmetic |
| **Decomposition** | Filtering → Extraction → Aggregation → Computation | Monolithic single-pass |
| **Error isolation** | Errors in one review don't cascade | Single error can corrupt entire output |

### Consistency Analysis

General ANoT's structured approach reduces variance because:
1. Each review is extracted independently with clear criteria
2. Aggregation and computation are deterministic (not LLM-dependent)
3. The Formula Seed provides explicit rules, reducing interpretation variance

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run General ANoT evaluation (100 restaurants, parallel)
python -m explore.general_anot.eval

# Run baselines for comparison
python -m explore.baselines.direct_llm_v2
python -m explore.baselines.cot
```

## Key Concepts

### Formula Seed

The Formula Seed is the output of Phase 1 - a complete, executable specification that Phase 2 interprets. It contains:

1. **Filtering**: Keywords to identify relevant reviews
2. **Extraction**: Semantic signals to extract from each review (with full definitions)
3. **Aggregation**: How to combine extractions (count, sum, max, min)
4. **Computation**: Formulas to compute final results

### Adjusted AUPRC

The evaluation metric that penalizes correct conclusions from wrong reasoning:

```
Adjusted AUPRC = Ordinal AUPRC × Primitive Accuracy
```

- **Ordinal AUPRC**: How well the risk score orders restaurants by true risk class
- **Primitive Accuracy**: How accurately intermediate values match ground truth

This metric ensures the model gets the right answer *for the right reasons*.

### Two-Phase Architecture

```
Phase 1: Compile                    Phase 2: Execute
┌─────────────────────┐            ┌─────────────────────┐
│ Task Formula (NL)   │            │ Restaurant Data     │
│ "Compute risk..."   │            │ - reviews[]         │
└─────────┬───────────┘            │ - metadata          │
          │                        └─────────┬───────────┘
          ▼                                  │
┌─────────────────────┐                      │
│ LLM: Understand     │                      │
│ - What to extract   │                      │
│ - How to aggregate  │                      │
│ - What to compute   │                      │
└─────────┬───────────┘                      │
          │                                  │
          ▼                                  ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Formula Seed (JSON) │───────────▶│ Interpreter         │
│ - filter keywords   │            │ 1. Filter reviews   │
│ - extraction schema │            │ 2. Extract signals  │
│ - aggregation defs  │            │ 3. Aggregate counts │
│ - computation steps │            │ 4. Compute formulas │
└─────────────────────┘            └─────────┬───────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │ Result              │
                                   │ - FINAL_RISK_SCORE  │
                                   │ - VERDICT           │
                                   │ - intermediates...  │
                                   └─────────────────────┘
```

## Usage

```python
from explore.general_anot import generate_formula_seed, FormulaSeedInterpreter

# Phase 1: Compile task formula to executable seed
seed = await generate_formula_seed(task_prompt, "task_name")

# Phase 2: Execute seed on restaurant data
interpreter = FormulaSeedInterpreter(seed)
result = await interpreter.execute(reviews, restaurant_context)

# Result contains FINAL_RISK_SCORE, VERDICT, and all intermediate values
print(result["VERDICT"])  # "Low Risk", "High Risk", or "Critical Risk"
print(result["FINAL_RISK_SCORE"])  # 0.0 - 20.0
```

## Run Commands

```bash
# General ANoT (main framework)
python -m explore.general_anot.eval

# Direct LLM baseline (zero-shot)
python -m explore.baselines.direct_llm_v2

# Chain of Thought baseline
python -m explore.baselines.cot

# With limit for quick testing
python -m explore.baselines.direct_llm_v2 --limit 10
```
