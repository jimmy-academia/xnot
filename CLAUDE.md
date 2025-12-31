# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based LLM evaluation framework comparing prompting methodologies on a restaurant recommendation task. The system evaluates LLMs against a dataset with ground-truth labels for three user personas.

## Commands

### Run Evaluation
```bash
# Development mode (default): creates results/dev/{NNN}_{run-name}/
python main.py --method cot --run-name baseline
python main.py --method knot --run-name experiment1
python main.py --method knot --mode dict --run-name dict_test

# Benchmark mode: set BENCHMARK_MODE=True in utils/arguments.py
# Creates results/benchmarks/{run-name}/ (tracked in git)

# Custom data paths
python main.py --method knot --data data/processed/complex_data.jsonl --run-name complex

# With pre-generated attacks
python main.py --method knot --attack typo_10 --run-name robustness

# Test with dummy method
python main.py --method dummy --limit 5 -v
```

### Generate Attacked Data
```bash
# Pre-generate attacked datasets (one-time)
python data/scripts/generate_attacks.py data/processed/real_data.jsonl
# Creates: data/attacked/typo_10.jsonl, data/attacked/inject_override.jsonl, etc.
```

### Environment Setup
```bash
# IMPORTANT: Always activate the virtual environment first
source .venv/bin/activate

# NOTE: Do not run pip install commands. If a package is missing, notify the user
# and let them decide whether to install it.

# API key: llm.py auto-loads from ../.openaiapi (no manual export needed)
# Or set manually:
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional LLM configuration
export LLM_PROVIDER="openai"  # or "anthropic" or "local"
export LLM_MODEL="gpt-4o-mini"

# Debug output
export NOT_DEBUG=1   # for rnot.py
export KNOT_DEBUG=1  # for knot.py
```

## Architecture

### Core Files

- **main.py** - Entry point: parses args, sets up experiment, delegates to run.py.

- **run.py** - Evaluation orchestration: `run_evaluation_loop()`, `evaluate()`, `evaluate_parallel()`.

- **utils/experiment.py** - `ExperimentManager` class for dev/benchmark mode directory handling.

- **utils/llm.py** - Unified LLM API wrapper. Supports OpenAI, Anthropic, and local endpoints.

- **methods/cot.py** - Chain-of-Thought using few-shot prompting.

- **methods/knot.py** - Knowledge Network of Thought with dynamic 2-phase script generation.

- **attack.py** - Attack functions (typo, injection, fake_review) and configs.

### Directory Structure

```
data/
├── raw/           # Raw Yelp data
├── processed/     # Generated datasets (real_data.jsonl, complex_data.jsonl)
├── attacked/      # Pre-generated attacked datasets
├── requests/      # User persona definitions
└── scripts/       # Data generation scripts

results/
├── dev/           # Development runs (gitignored)
│   ├── 001_baseline/
│   └── 002_experiment/
└── benchmarks/    # Benchmark runs (tracked in git)
    └── final_run/

utils/             # Utility modules
methods/           # Evaluation methods (cot, knot, etc.)
doc/               # Experiment documentation
```

### Method Interface

All methods implement:
```python
def method(query, context: str) -> int
    # returns: -1 (not recommend), 0 (neutral), 1 (recommend)
```

### Input Modes (knot.py)

**String mode** (default): Input is formatted text, LLM extracts info
```
(0)=LLM("Extract reviews from: {(input)}")
(1)=LLM("Analyze {(0)}[0]")
```

**Dict mode**: Input is dict, direct key/index access
```
(0)=LLM("Name is {(input)}[item_name]")
(1)=LLM("Review: {(input)}[item_data][0][review]")
```

### Key Design Patterns

1. **Leakage Prevention**: `final_answers` and `condition_satisfy` never passed to LLM.

2. **Variable Substitution**: `{(var)}[key][index]` for nested access in scripts.

3. **Dynamic Script Generation** (knot.py): LLM generates execution plan at runtime.

### User Request Personas (R0, R1, R2)

- **R0**: Quiet dining, comfortable seating, budget-conscious
- **R1**: Allergy-conscious, needs clear ingredient labeling
- **R2**: Chicago tourist seeking authentic local experience
