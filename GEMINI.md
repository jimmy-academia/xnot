# GEMINI.md

LLM evaluation framework comparing prompting methods on restaurant recommendations.

## Important Rules for Gemini

- **NEVER use `rm -rf` on benchmark results** - Always prompt the user to delete files manually.
- **Use `--dev` flag for development testing** - Results go to `results/dev/` (gitignored).
- **Only commit benchmark results when explicitly requested**.

## Quick Reference

### Environment Setup
All commands should be run from the project root.
Ensure the virtual environment is activated:

```bash
source .venv/bin/activate
```

### Running Evaluations

Standard evaluation (ANoT method with 50 candidates):
```bash
python main.py --method anot --candidates 50
```

Chain-of-Thought with typo attack:
```bash
python main.py --method cot --candidates 10 --attack typo_10
```

Scaling experiment (default, no --candidates):
```bash
python main.py --method anot
```

### Development Mode
Use `--dev` to save results to `results/dev/` which is ignored by git.
```bash
python main.py --method anot --candidates 50 --dev
```

### Attack Sweep
Run all attacks on a small candidate set:
```bash
python main.py --method cot --attack all --candidates 10
```

## Key Locations

- **Add method**: `methods/newmethod.py` + `methods/__init__.py` (METHOD_REGISTRY)
- **Add attack**: `attack.py` (ATTACK_CONFIGS)
- **Change models**: `utils/llm.py` (MODEL_CONFIG)
- **Modify evaluation**: `run/evaluate.py`

## Method System

### BaseMethod Class
All methods inherit from `BaseMethod` (`methods/base.py`). 
Implement `evaluate` (single item) and `evaluate_ranking` (ranking task).

### Method Registry
Methods are registered in `methods/__init__.py`.
Example: `METHOD_REGISTRY["cot"] = (ChainOfThought, True)` where `True` indicates defense support.

## Data Modes

| Mode | Methods | Context Type | Truncation |
|------|---------|--------------|------------|
| **String** | cot, ps, listwise, etc. | Pre-formatted text | Yes (pack-to-budget) |
| **Dict** | anot, weaver, react | Raw dict | No (selective access) |

Mode is automatically determined by the method type in `run/orchestrate.py`.

## Attack System

Attacks target non-gold items only.
Types include:
- **Noise**: `typo_10`, `typo_20`
- **Injection**: `inject_override`, `inject_fake_sys`, etc.
- **Fake reviews**: `fake_positive`, `fake_negative`
- **Sarcastic**: `sarcastic_wifi`, etc.

Use `--attack all` to run all attacks.

## Directory Structure

- `data/`: Datasets and scripts.
- `results/`: Output results. `dev/` for testing, `benchmarks/` for tracked runs.
- `methods/`: Implementation of different prompting methods.
- `run/`: Core execution logic (`orchestrate.py`, `evaluate.py`).
- `utils/`: Helpers for arguments, LLM, logging, etc.

## Common CLI Arguments

- `--method`: `cot`, `anot`, `plan_act`, etc.
- `--candidates`: Number of items to rank.
- `--attack`: Attack type or `all`.
- `--dev`: Run in dev mode (results in `results/dev/`).
- `--sequential`: Disable parallel execution.
- `--model`: Override default model (e.g. `gpt-4o`).
