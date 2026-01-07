# ANoT: Adaptive Network of Thought

> Evaluating LLM reasoning on structured constraint-satisfaction tasks

## Overview

ANoT is a research framework for evaluating how well LLMs can solve constraint-satisfying reranking tasks over multi-source data. Given a natural language request with implicit logical constraints and N candidate items with both structured attributes and unstructured text (reviews), the task is to identify the unique item that satisfies all constraints.

**Key Features:**
- 100 benchmark requests across 10 complexity groups (G01-G10)
- 6 baseline methods with paper references
- Adversarial robustness testing framework
- Reproducible data pipeline from raw Yelp data

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Set API key
export OPENAI_API_KEY="sk-..."

# Run evaluation
python main.py --method anot --candidates 20
```

## Documentation

### Research
- [Design Rationale](doc/paper/design_rationale.md) - Benchmark design decisions
- [Methodology](doc/paper/methodology.md) - Evaluation framework design
- [ANoT Architecture](doc/paper/anot_architecture.md) - Three-phase design
- [Baselines](doc/paper/baselines.md) - Implemented methods with citations

### Guides
- [Run Experiments](doc/guides/run_experiments.md) - Execute and analyze evaluations
- [Methods Architecture](doc/guides/architecture.md) - Method structure conventions
- [Recreate Dataset](doc/guides/recreate_philly_cafes.md) - Reproduce philly_cafes from raw Yelp data
- [Create Benchmark](doc/guides/create_new_benchmark.md) - Build a new evaluation benchmark

### Reference
- [Configuration](doc/reference/configuration.md) - CLI arguments, LLM config
- [Attacks](doc/reference/attacks.md) - Adversarial attack system
- [Defense Mode](doc/reference/defense_mode.md) - Attack-resistant evaluation
- [Evidence Types](doc/reference/evidence_types.md) - Validation logic reference
- [Request Structure](doc/reference/request_structure.md) - Request JSON schema

### Internal
- [TODO](doc/internal/TODO.md) - Development tasks
- [Attack Plan](doc/internal/attack_plan.md) - Attack testing plan

See [doc/README.md](doc/README.md) for complete documentation index.

## Benchmark: Philadelphia Cafes

| Metric | Value |
|--------|-------|
| Restaurants | 50 |
| Requests | 100 |
| Groups | 10 (G01-G10) |
| Validation | 100% |

Request groups test progressively complex reasoning:
- **G01-G04**: Flat AND logic with different evidence types
- **G05-G08**: Nested logical structures (AND/OR combinations)
- **G09-G10**: Social graph filtering (1-hop, 2-hop)

## Methods

| Method | Description |
|--------|-------------|
| `cot` | Chain-of-Thought |
| `ps` | Plan-and-Solve |
| `plan_act` | Plan-and-Act |
| `listwise` | Listwise Reranking |
| `weaver` | SQL+LLM Hybrid |
| `anot` | Adaptive Network of Thought (ours) |

```bash
# Compare methods
python main.py --method cot --candidates 20
python main.py --method anot --candidates 20
```

## Project Structure

```
anot/
├── methods/           # Evaluation methods
│   ├── cot.py        # Chain-of-Thought
│   ├── anot/         # ANoT (three-phase)
│   └── ...
├── data/             # Benchmark datasets
│   └── philly_cafes/ # Primary benchmark
├── preprocessing/    # Data pipeline
│   ├── curate.py    # Stage 1: Curation
│   └── prompts/     # Claude Code templates
├── run/             # Evaluation orchestration
├── utils/           # LLM, arguments, experiment utilities
├── doc/             # Documentation
│   ├── paper/       # Research paper support
│   ├── guides/      # How-to guides
│   ├── reference/   # Technical specs
│   └── internal/    # Development docs
└── results/         # Experiment outputs
```

## Running Experiments

```bash
# Basic usage
python main.py --method METHOD --candidates N

# Scaling experiment (varies N)
python main.py --method anot

# Filter requests
python main.py --method anot --limit 1-10  # First group only

# Attack robustness
python main.py --method anot --attack typo_10 --candidates 20
```

See [Running Experiments Guide](doc/guides/run_experiments.md) for details.

## Citation

```bibtex
@article{anot2024,
  title={ANoT: Adaptive Network of Thought for Constraint-Satisfying Reranking},
  author={...},
  year={2024}
}
```
