# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based LLM evaluation framework comparing prompting methodologies on a restaurant recommendation task. The system evaluates LLMs against a dataset with ground-truth labels for three user personas.

## Commands

**IMPORTANT: Always use `.venv/bin/python` or activate the virtual environment first!**

### Run Evaluation
```bash
# ALWAYS use the virtual environment
source .venv/bin/activate
# OR prefix commands with .venv/bin/python

# Development mode (default): creates results/dev/{NNN}_{run-name}/
.venv/bin/python main.py --method cot --run-name baseline
python main.py --method anot --run-name experiment1

# Benchmark mode: set BENCHMARK_MODE=True in utils/arguments.py
# Creates results/benchmarks/{run-name}/ (tracked in git)

# Custom data paths
python main.py --method anot --data data/processed/complex_data.jsonl --run-name complex

# With pre-generated attacks
python main.py --method anot --attack typo_10 --run-name robustness

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

# Verbose terminal output (default: on, use --no-verbose to disable)
python main.py --method anot --no-verbose

# Structured logs written to {run_dir}/:
#   results_{n}.jsonl   - predictions + per-request usage
#   usage.jsonl         - consolidated usage across runs
#   anot_trace.jsonl    - ANoT phase-level structured trace
#   debug.log           - ANoT debug (always-on, file-only, overwrites each run)
#   config.json         - run configuration
#
# ANoT debug.log: Full LLM prompts/responses, phase traces, timestamps.
# No env var needed - always written when run_dir exists.
# See doc/logging.md for full schema details
```

## Architecture

### Core Files

- **main.py** - Entry point: parses args, sets up experiment, delegates to run/.

- **run/** - Evaluation orchestration package:
  - `orchestrate.py` - `run_single()`, `run_evaluation_loop()`
  - `evaluate.py` - `evaluate_ranking()`, `compute_multi_k_stats()`
  - `scaling.py` - `run_scaling_experiment()`
  - `shuffle.py` - Shuffle utilities for position bias mitigation
  - `io.py` - Result loading/saving

- **methods/anot/** - Adaptive Network of Thought package:
  - `core.py` - Main `AdaptiveNetworkOfThought` class with 3 phases
  - `helpers.py` - DAG building, formatting utilities
  - `tools.py` - Phase 2 LWT manipulation tools
  - `prompts.py` - LLM prompt constants

- **utils/experiment.py** - `ExperimentManager` class for dev/benchmark mode directory handling.

- **utils/llm.py** - Unified LLM API wrapper. Supports OpenAI, Anthropic, and local endpoints.

- **methods/cot.py** - Chain-of-Thought using few-shot prompting.

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

run/               # Evaluation orchestration package
├── orchestrate.py # run_single, run_evaluation_loop
├── evaluate.py    # evaluate_ranking, stats computation
├── scaling.py     # Scaling experiment
├── shuffle.py     # Shuffle utilities
└── io.py          # Result I/O

methods/           # Evaluation methods
├── anot/          # ANoT package (core.py, helpers.py, tools.py, prompts.py)
├── cot.py         # Chain-of-Thought
├── listwise.py    # Listwise reranking
├── weaver.py      # SQL+LLM hybrid
└── ...

utils/             # Utility modules
doc/               # Implementation documentation (not for main paper)
```

**Note**: Documentation in `doc/` describes implementation details for code maintenance. These are not intended for the main research paper.

### Method Interface

All methods implement:
```python
def method(query, context: str) -> int
    # returns: -1 (not recommend), 0 (neutral), 1 (recommend)
```

### Input Modes (methods/anot/)

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

3. **Dynamic Script Generation** (methods/anot/): LLM generates execution plan at runtime.

### User Request Personas (R0, R1, R2)

- **R0**: Quiet dining, comfortable seating, budget-conscious
- **R1**: Allergy-conscious, needs clear ingredient labeling
- **R2**: Chicago tourist seeking authentic local experience

## WIP: Attack Implementation

**Status**: Planned, not yet integrated. See `doc/attack_plan.md` for details.

**Goal**: Test robustness - attacks should cause CoT to fail while ANoT resists.

**Key files**:
- `oldsrc/attack.py` - Existing attack implementations (typo, injection, fake_review, sarcastic)
- `utils/arguments.py` - Already has `--attack`, `--seed`, `--defense` flags (parsed but unused)

**Next steps**:
1. Copy `oldsrc/attack.py` → `attack.py`
2. Wire into `run/scaling.py` and `run/orchestrate.py` (after `filter_by_candidates`)
3. For injection attacks, target = OPPOSITE of ground truth (gold item gets -1, others get 1)
4. Test: CoT fails with attacks, ANoT resists, defense on CoT doesn't help
