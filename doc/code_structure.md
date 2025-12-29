# XNoT Code Structure & Developer Guide

This document outlines the organization of the codebase.

## Directory Overview

```text
xnot/
├── main.py                 # Entry point: parse args, setup, delegate to run.py
├── run.py                  # Evaluation orchestration (evaluate, run_evaluation_loop)
├── attack.py               # Attack functions and ATTACK_CONFIGS
│
├── utils/
│   ├── arguments.py        # CLI args + BENCHMARK_MODE/PARALLEL_MODE settings
│   ├── experiment.py       # ExperimentManager (dev/benchmark directory handling)
│   ├── logger.py           # Colored SimpleLogger + DebugLogger
│   ├── llm.py              # Unified LLM API wrapper (OpenAI/Anthropic/Local)
│   └── helper.py           # Legacy helpers (deprecated)
│
├── methods/
│   ├── __init__.py         # get_method() factory
│   ├── cot.py              # Chain-of-Thought
│   ├── knot.py             # Knowledge Network of Thought
│   └── dummy.py            # Dummy method for testing
│
├── data/
│   ├── loader.py           # load_data, load_requests, load_attacked_data
│   ├── raw/                # Raw Yelp data
│   ├── processed/          # Generated datasets (real_data.jsonl)
│   ├── attacked/           # Pre-generated attacked datasets
│   ├── requests/           # User persona definitions
│   └── scripts/            # Data generation scripts
│
├── results/
│   ├── dev/                # Development runs (gitignored)
│   └── benchmarks/         # Benchmark runs (tracked in git)
│
└── doc/                    # Documentation
```

## Execution Flow

```
main.py
  ├── parse_args()                    # utils/arguments.py
  ├── setup_logger_level(verbose)     # utils/logger.py
  ├── config_llm(args)                # utils/llm.py
  ├── create_experiment(args)         # utils/experiment.py
  │     └── ExperimentManager.setup() # creates results/dev/ or results/benchmarks/
  ├── load_data(), load_requests()    # data/loader.py
  ├── get_method_instance()           # methods/__init__.py
  └── run_evaluation_loop()           # run.py
        ├── load_attacked_data()      # data/loader.py
        ├── evaluate() / evaluate_parallel()
        └── experiment.save_results()
```

## Key Settings

**utils/arguments.py:**
- `BENCHMARK_MODE = False` - Set True for benchmark runs (tracked in git)
- `PARALLEL_MODE = True` - Enable parallel evaluation

## Two Run Modes

| Mode | Directory | Git | Overwrites |
|------|-----------|-----|------------|
| Development | `results/dev/{NNN}_{name}/` | Ignored | Auto-increment |
| Benchmark | `results/benchmarks/{name}/` | Tracked | Fails if exists |
