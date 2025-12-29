# XNoT Code Structure & Developer Guide

This document outlines the organization of the codebase, detailing the responsibility of each script and module.

## 📂 Directory Overview

```text
xnot/
├── main.py                 # Primary entry point for experiments
├── llm.py                  # Unified LLM API wrapper (OpenAI/Anthropic/Local)
├── methods/                # Reasoning algorithms (CoT, KNoT, etc.)
├── scripts/                # Data generation and preprocessing tools
├── utils/                  # Helper utilities (logging, path management)
├── data/                   # Dataset storage (inputs and outputs)
├── results/                # Experiment artifacts (logs, predictions)
└── past_ref/               # Past artifacts (to delete after refactor complete)
```

## Detailed script instruction
- main.py
    - parse_args() ...(utils/arguments.py)
    - setup_logger_level(args.verbose) ....(utils/logger.py)
    - setup_llm ...(utils/llm.py)
    - result_dir management
    - load_data
    - prepare_method
    - run_method
- 
