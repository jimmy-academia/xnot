# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based LLM evaluation framework comparing prompting methodologies (Chain-of-Thought vs Network-of-Thought) on a restaurant recommendation task. The system evaluates LLMs against a dataset with ground-truth labels for three user personas.

## Commands

### Run Evaluation
```bash
python main.py --method cot --data data.jsonl --out results.jsonl
python main.py --method not --data data.jsonl --out results.jsonl
python main.py --method dummy --limit 5  # Test with dummy method
```

### Environment Setup
```bash
# API key is stored in ../.openaiapi (parent directory)
export OPENAI_API_KEY=$(cat ../.openaiapi)

# Or for Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional LLM configuration (llm.py)
export LLM_PROVIDER="openai"  # or "anthropic" or "local"
export LLM_MODEL="gpt-4o-mini"
export NOT_DEBUG=1  # Enable debug output for rnot.py
```

## Architecture

### Core Files

- **main.py** - Evaluation harness with `--method` flag to select cot/not/dummy.

- **llm.py** - Unified LLM API wrapper. Supports OpenAI, Anthropic, and local endpoints. Auto-detects provider from API keys.

- **cot.py** - Chain-of-Thought using few-shot prompting. Exports `method(query, context) -> int`.

- **rnot.py** - Network-of-Thought with fixed 5-step pipeline. Exports `method(query, context) -> int`.

- **data.jsonl** - Dataset with 9 Chicago restaurants and ground-truth labels for R0/R1/R2.

### Method Interface

All methods implement:
```python
def method(query: str, context: str) -> int
    # returns: -1 (not recommend), 0 (neutral), 1 (recommend)
```

### Key Design Patterns

1. **Leakage Prevention**: `final_answers` and `condition_satisfy` fields are never passed to the LLM.

2. **Provider Abstraction**: Both methods support Anthropic and OpenAI with lazy-loaded dependencies.

3. **Variable Substitution** (rnot.py): Script steps reference previous outputs using `{(index)}` notation.

### User Request Personas (R0, R1, R2)

- **R0**: Quiet dining, comfortable seating, budget-conscious
- **R1**: Allergy-conscious, needs clear ingredient labeling
- **R2**: Chicago tourist seeking authentic local experience
