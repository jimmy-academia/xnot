# anot

Research framework for ANoT (Adaptive Network of Thought) - evaluating LLM prompting methods on structured multi-source data.

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Set API key (or place in ../.openaiapi)
export OPENAI_API_KEY="sk-..."
```

## Data Pipeline

Two-stage process from raw Yelp data to evaluation dataset:

### Stage 1: Curation

```bash
python -m preprocessing.curate          # Interactive city/category selection
python -m preprocessing.analyze {name}  # Generate attribute distributions
```

Output: `preprocessing/output/{name}/` (all filtered restaurants/reviews)

### Stage 2: Ground Truth Selection

Using Claude Code to select 20 restaurants, 400 reviews, 50 requests.

Output: `data/{name}/` (final evaluation dataset)

See `preprocessing/README.md` for full documentation and kickstart prompt.

## Running Evaluations

### Basic Usage

```bash
# Scaling experiment (default) - runs all scale points: 10, 15, 20, 25, 30, 40, 50
python main.py --method cot --data philly_cafes

# Single scale - run with exactly N candidates
python main.py --method cot --candidates 10

# Development mode (saves to results/dev/)
python main.py --method cot --candidates 10 --dev
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | anot | Method: cot, ps, plan_act, listwise, weaver, anot, dummy |
| `--data` | philly_cafes | Dataset name in data/ directory |
| `--candidates` | None | Run single scale with N candidates (default: scaling experiment) |
| `--k` | 5 | Number of predictions for Hits@K evaluation |
| `--limit` | None | Filter requests: N (first N), N-M (range), N,M,O (indices) |
| `--dev` | False | Development mode (results/dev/) |
| `--force` | False | Overwrite existing results |
| `--provider` | openai | LLM provider: openai, anthropic, local |
| `--model` | None | Override model (default: role-based selection) |
| `--auto` | 1 | Target number of runs for benchmark |

### Examples

```bash
# Run CoT on philly_cafes with 10 candidates
python main.py --method cot --candidates 10

# Run all scale points with Plan-and-Solve
python main.py --method ps

# Run first 5 requests only (for testing)
python main.py --method cot --candidates 10 --limit 5

# Run specific requests by index
python main.py --method cot --candidates 10 --limit 0,5,10
```

## Project Structure

```
anot/
├── methods/            # Prompting methods
│   ├── base.py         # Method interface
│   ├── cot.py          # Chain-of-Thought baseline
│   ├── ps.py           # Plan-and-Solve
│   ├── plan_act.py     # Plan-and-Act agent baseline
│   └── anot.py         # Our method (Adaptive Network of Thought)
├── prompts/            # Standard prompt templates
│   └── task_descriptions.py  # Reusable task descriptions
├── utils/              # Shared utilities
│   ├── llm.py          # LLM API wrapper with context tracking
│   └── usage.py        # Token usage and cost tracking
├── preprocessing/      # Stage 1: Data curation pipeline
│   ├── curate.py       # Interactive city/category selection + LLM scoring
│   ├── analyze.py      # Generate attribute distributions
│   ├── raw/            # Raw Yelp data (gitignored)
│   └── output/         # Curated selections + ground_truth_selection.md
├── data/               # Stage 2: Final evaluation datasets (20 restaurants, 50 requests)
├── doc/                # Research documentation
├── results/            # Experiment outputs
└── oldsrc/             # Legacy reference code
```

## Documentation

See [`doc/README.md`](doc/README.md) for full documentation index.

### Benchmark Dataset
- [data/philly_cafes/README.md](data/philly_cafes/README.md) - Dataset overview and request group designs (G01-G08)
- [data/philly_cafes/requests_reference.md](data/philly_cafes/requests_reference.md) - Complete request-answer reference
- [data/philly_cafes/statistics.md](data/philly_cafes/statistics.md) - Evidence distribution and coverage analysis
- [doc/condition_design.md](doc/condition_design.md) - Bottom-up anchor-first design methodology

### Research & Architecture
- [doc/baselines.md](doc/baselines.md) - Implemented baseline methods with paper references
- [doc/anot_architecture.md](doc/anot_architecture.md) - ANoT three-phase design
- [doc/research_plan.md](doc/research_plan.md) - Research plan and specifications
- [doc/logging.md](doc/logging.md) - Logging infrastructure
