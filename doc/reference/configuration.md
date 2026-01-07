# Configuration Reference

This document describes CLI arguments, LLM configuration, and execution modes.

---

## CLI Arguments

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `philly_cafes` | Dataset name or path to data directory |
| `--run-name` | None | Name for this run (creates results/{N}_{run-name}/) |
| `--limit` | None | Filter requests: N (first N), N-M (range), or N,M,O (specific indices) |
| `--run` | 1 | Target run number |
| `--force` | False | Force overwrite existing results |
| `--review-limit` | None | Limit reviews per restaurant (for testing) |
| `--candidates` | None | Run single evaluation with N candidates |

### Method Arguments

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--method` | `anot` | cot, ps, plan_act, listwise, weaver, anot, react, dummy | Evaluation method |

### Attack Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--attack` | `none` | Attack type (none, all, both, or specific attack name) |
| `--seed` | None | Random seed for reproducible attacks |
| `--defense` | False | Enable defense prompts |
| `--attack-restaurants` | None | Number of non-gold restaurants to attack (default: all) |
| `--attack-reviews` | 1 | Number of reviews per restaurant to modify |
| `--attack-target-len` | None | Target character length (heterogeneity attack only) |

### LLM Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider` | `openai` | LLM provider: openai, anthropic, local |
| `--model` | None | Override model (default: role-based selection) |
| `--temperature` | 0.0 | LLM temperature |
| `--max-tokens` | 1024 | Max tokens for response |
| `--max-tokens-reasoning` | 4096 | Max tokens for reasoning models |
| `--base-url` | None | Base URL for local provider |

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 5 | Number of predictions for Hits@K evaluation |
| `--shuffle` | `random` | Shuffle strategy: none, middle, random |

### Execution Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-concurrent` | 200 | Max concurrent API calls |
| `--sequential` | False | Disable parallel execution |
| `--auto` | 1 | Target number of runs for benchmark |
| `--dev` | False | Use dev mode instead of benchmark mode |
| `--verbose` / `--no-verbose` | True | Enable/disable verbose output |
| `--full` | False | Show full per-request results |

**Reference**: [utils/arguments.py](../../utils/arguments.py)

---

## Results Directory Structure

### Dev Mode

- **Path**: `results/dev/{NNN}_{run-name}/`
- **Behavior**: Gitignored, auto-numbered (001, 002, ...)
- **Enable**: `--dev` flag

### Benchmark Mode

- **Path**: `results/benchmarks/{method}_{data}/{attack}/run_{N}/`
- **Behavior**: Git-tracked, named by configuration
- **Enable**: Default (or `BENCHMARK_MODE=True` in utils/arguments.py)

### Output Files

| File | Description |
|------|-------------|
| `config.json` | Run configuration and final stats |
| `results.jsonl` | Per-request predictions and usage |
| `usage.jsonl` | Consolidated usage across runs |
| `debug.log` | Full LLM prompts/responses (ANoT) |
| `anot_trace.jsonl` | Phase-level structured trace (ANoT) |

**Reference**: [utils/experiment.py](../../utils/experiment.py), [doc/reference/logging.md](logging.md)

---

## LLM Configuration

### Model Selection

Role-based model selection defined in `MODEL_CONFIG`:

```python
MODEL_CONFIG = {
    "planner": "gpt-5-nano",
    "worker": "gpt-5-nano",
    "default": "gpt-5-nano",
}
```

Override with `--model` flag.

### Token Limits

**Fixed input limits** (`MODEL_INPUT_LIMITS`):
```python
{
    "gpt-5-nano": 270000,  # 400k context, 270k input / 32k output split
}
```

**Context windows** (`MODEL_CONTEXT_LIMITS`):
```python
{
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o1": 200000,
    "o3-mini": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    # ... etc
}
```

**Budget calculation**: `get_token_budget(model)` returns input token budget.
- If model has fixed input limit, use that
- Otherwise: `context_limit - output_reserve - safety_margin`

### Rate Limiting

Global semaphore limits concurrent API calls:
- Default: 200 concurrent calls
- Override: `--max-concurrent` flag or `init_rate_limiter(N)`

### Retry Logic

Automatic retry with exponential backoff for:
- Network errors
- Rate limit errors (429)
- Server errors (5xx)

Configuration:
- `max_retries`: 6 (default)
- `request_timeout`: 90.0 seconds

**Reference**: [utils/llm.py](../../utils/llm.py)

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `LLM_PROVIDER` | Override provider (openai, anthropic, local) |
| `LLM_MODEL` | Override model |
| `ANOT_DEBUG` | ANoT debug level (0-3) |

### API Key Loading

Keys are loaded in order:
1. Environment variable
2. File at `../../.openaiapi` (one level above project root)

---

## Mode Flags

Defined in `utils/arguments.py`:

```python
PARALLEL_MODE = True   # Enable parallel execution
BENCHMARK_MODE = True  # Enable benchmark mode (vs dev mode)
```

### Derived Arguments

```python
args.parallel = PARALLEL_MODE and not args.sequential
args.benchmark = BENCHMARK_MODE and not args.dev
```

---

## Method Choices

Available methods defined in `METHOD_CHOICES`:

| Method | Description |
|--------|-------------|
| `cot` | Chain-of-Thought |
| `ps` | Plan-and-Solve |
| `plan_act` | Plan then Act |
| `listwise` | Listwise reranking |
| `weaver` | SQL+LLM hybrid |
| `anot` | Adaptive Network of Thought |
| `react` | ReAct (Reason + Act) |
| `dummy` | Test method (dev mode only) |

**Reference**: [doc/paper/baselines.md](../paper/baselines.md)
