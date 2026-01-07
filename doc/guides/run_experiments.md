# Running Experiments

Guide to running evaluations with the ANoT framework, including methods, attacks, and result analysis.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run single evaluation
python main.py --method cot --candidates 20

# Run with specific requests
python main.py --method anot --candidates 20 --limit 10

# Run all baselines
for method in cot ps plan_act listwise weaver anot; do
    python main.py --method $method --candidates 20
done
```

---

## Basic Usage

### Command Structure

```bash
python main.py \
    --method METHOD \
    --data DATASET \
    --candidates N \
    [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--method` | Evaluation method | `cot`, `anot`, `listwise` |

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Dataset name | `philly_cafes` |
| `--candidates` | Number of candidates | Scaling experiment |
| `--limit` | Filter requests | All requests |
| `--run-name` | Name for results directory | Auto-generated |

---

## Available Methods

### Champion Methods

| Method | Description | Paper Reference |
|--------|-------------|-----------------|
| `cot` | Chain-of-Thought | Wei et al., 2022 |
| `ps` | Plan-and-Solve | Wang et al., 2023 |
| `plan_act` | Plan-and-Act | - |
| `listwise` | Listwise Reranking | Sun et al., 2023 |
| `weaver` | SQL+LLM Hybrid | Arora et al., 2023 |
| `anot` | Adaptive Network of Thought | Ours |

### Testing Method

| Method | Description |
|--------|-------------|
| `dummy` | Returns random prediction (for testing pipeline) |

### Example: Compare Methods

```bash
# Run all champion methods
for method in cot ps plan_act listwise weaver anot; do
    python main.py --method $method --candidates 20 --run-name baseline
done
```

---

## Candidate Scaling

### Fixed Candidates

Run with specific number of candidates:

```bash
python main.py --method anot --candidates 10
python main.py --method anot --candidates 20
python main.py --method anot --candidates 50
```

### Scaling Experiment

Without `--candidates`, runs scaling experiment across multiple N values:

```bash
python main.py --method anot
# Runs: N=10, 20, 30, 40, 50
```

---

## Request Filtering

### First N Requests

```bash
python main.py --method anot --limit 10
# Runs on R01-R10
```

### Request Range

```bash
python main.py --method anot --limit 21-30
# Runs on R21-R30 (G03 group)
```

### Specific Requests

```bash
python main.py --method anot --limit 1,11,21,31,41
# Runs on R01, R11, R21, R31, R41 (one from each G01-G05)
```

### By Group

```bash
# G01 (Simple AND)
python main.py --method anot --limit 1-10

# G04 (Credibility Weighting)
python main.py --method anot --limit 31-40

# G09-G10 (Social Filter)
python main.py --method anot --limit 81-100
```

---

## Attack Experiments

### Available Attacks

**Noise Attacks**

| Attack | Description |
|--------|-------------|
| `typo_10` | 10% character typos |
| `typo_20` | 20% character typos |
| `heterogeneity` | Variable review lengths (requires `--attack-target-len`) |

**Injection Attacks**

| Attack | Description |
|--------|-------------|
| `inject_override` | System prompt override injection |
| `inject_fake_sys` | Fake system message injection |
| `inject_promotion` | Self-promotion injection (code: `inject_hidden`) |

**Fake Review Attacks**

| Attack | Description |
|--------|-------------|
| `fake_positive` | Fake positive reviews |
| `fake_negative` | Fake negative reviews |
| `sarcastic_wifi` | Sarcastic WiFi mentions |
| `sarcastic_noise` | Sarcastic noise mentions |
| `sarcastic_outdoor` | Sarcastic outdoor mentions |
| `sarcastic_all` | All sarcastic patterns |

See [Attacks Reference](../reference/attacks.md) for detailed descriptions.

### Run Single Attack

```bash
python main.py --method anot --candidates 20 --attack typo_10
```

### Run All Attacks

```bash
python main.py --method anot --candidates 20 --attack all
```

### Clean Baseline + All Attacks

```bash
python main.py --method anot --candidates 20 --attack both
```

### Attack Parameters

```bash
# Control which restaurants get attacked
python main.py --method anot --attack inject_override --attack-restaurants 5

# Control reviews per restaurant
python main.py --method anot --attack typo_10 --attack-reviews 3

# Set random seed for reproducibility
python main.py --method anot --attack typo_10 --seed 42

# Enable defense prompts
python main.py --method anot --attack inject_override --defense
```

---

## LLM Configuration

### Provider Selection

```bash
# OpenAI (default)
python main.py --method anot --provider openai

# Anthropic
python main.py --method anot --provider anthropic

# Local endpoint
python main.py --method anot --provider local --base-url http://localhost:8000
```

### Model Override

```bash
# Use specific model
python main.py --method anot --model gpt-4o

# Use with Anthropic
python main.py --method anot --provider anthropic --model claude-3-5-sonnet-20241022
```

### Temperature and Tokens

```bash
# Adjust temperature (default: 0.0)
python main.py --method anot --temperature 0.7

# Adjust max tokens
python main.py --method anot --max-tokens 2048
```

---

## Execution Options

### Parallel Execution (Default)

```bash
# Default: parallel with 200 concurrent calls
python main.py --method anot

# Adjust concurrency
python main.py --method anot --max-concurrent 50
```

### Sequential Execution

```bash
python main.py --method anot --sequential
```

### Multiple Runs

```bash
# Run 3 times for variance measurement
python main.py --method anot --candidates 20 --auto 3
```

---

## Results Directory

### Dev Mode (Default)

```
results/dev/{NNN}_{run-name}/
├── config.json          # Run configuration
├── results_1.jsonl      # Predictions and usage
├── usage.jsonl          # Consolidated usage
├── anot_trace.jsonl     # ANoT phase traces (if method=anot)
└── debug.log            # Debug output
```

### Benchmark Mode

```bash
python main.py --method anot  # BENCHMARK_MODE=True in arguments.py
```

```
results/benchmarks/{run-name}/
├── config.json
├── results_1.jsonl
└── ...
```

### Output Files

| File | Content |
|------|---------|
| `config.json` | Run parameters, timestamps |
| `results_{n}.jsonl` | Per-request predictions and metrics |
| `usage.jsonl` | Token usage across runs |
| `anot_trace.jsonl` | ANoT phase-level trace |
| `debug.log` | Full prompts and responses |

---

## Result Analysis

### View Results

Results are printed after each run:

```
=== RESULTS ===
Method: anot
Candidates: 20
Requests: 100

Hits@1: 85.0%
Hits@5: 95.0%
MRR: 0.892

Per-group breakdown:
  G01: 90.0% (9/10)
  G02: 85.0% (8.5/10)
  ...
```

### Aggregate Multiple Runs

```python
from utils.aggregate import aggregate_benchmark_runs, print_summary

# Get aggregated stats
summary = aggregate_benchmark_runs("anot", "philly_cafes")
print_summary(summary)
```

### Compare Methods

```bash
# Run comparison
for method in cot anot; do
    python main.py --method $method --candidates 20
done

# Results in results/benchmarks/
```

---

## Example Workflows

### 1. Quick Sanity Check

```bash
# Test pipeline with dummy method
python main.py --method dummy --limit 5 --candidates 10 -v
```

### 2. Method Comparison

```bash
# Compare all methods on full dataset
for method in cot ps plan_act listwise weaver anot; do
    python main.py --method $method --candidates 20 --run-name comparison
done
```

### 3. Scaling Analysis

```bash
# Run scaling experiment for anot
python main.py --method anot --run-name scaling
# Runs N=5,10,15,...,50 automatically
```

### 4. Attack Robustness

```bash
# Clean baseline
python main.py --method anot --candidates 20 --attack none

# All attacks
python main.py --method anot --candidates 20 --attack all

# With defense
python main.py --method anot --candidates 20 --attack all --defense
```

### 5. Group-Specific Analysis

```bash
# Test on complex groups only (G06-G08)
python main.py --method anot --candidates 20 --limit 51-80

# Test social filter groups (G09-G10)
python main.py --method anot --candidates 20 --limit 81-100
```

---

## Troubleshooting

### API Key Issues

```bash
# Set API key
export OPENAI_API_KEY="sk-..."
# Or use file: ../.openaiapi

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Rate Limits

```bash
# Reduce concurrency
python main.py --method anot --max-concurrent 10

# Or run sequentially
python main.py --method anot --sequential
```

### Memory Issues

```bash
# Limit requests
python main.py --method anot --limit 20

# Limit reviews per restaurant
python main.py --method anot --review-limit 10
```

### Debug Output

```bash
# Verbose mode (default)
python main.py --method anot -v

# Check debug.log for full prompts/responses
cat results/dev/001_run/debug.log
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `LLM_PROVIDER` | Default provider |
| `LLM_MODEL` | Default model |

---

## Related Documentation

- [Configuration Reference](../reference/configuration.md) - Full CLI arguments
- [Attacks Reference](../reference/attacks.md) - Attack system details
- [Defense Mode](../reference/defense_mode.md) - Attack-resistant evaluation
- [Baselines](../paper/baselines.md) - Method details and citations
- [ANoT Architecture](../paper/anot_architecture.md) - Our method design
- [Logging](../reference/logging.md) - Output file formats
