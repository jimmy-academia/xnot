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

## Project Structure

```
anot/
├── preprocessing/      # Stage 1: Data curation pipeline
│   ├── curate.py       # Interactive city/category selection + LLM scoring
│   ├── analyze.py      # Generate attribute distributions
│   ├── raw/            # Raw Yelp data (gitignored)
│   └── output/         # Curated selections + ground_truth_selection.md
├── data/               # Stage 2: Final evaluation datasets (20 restaurants, 50 requests)
├── utils/              # Shared utilities (llm.py)
├── doc/                # Research documentation
├── results/            # Experiment outputs
└── oldsrc/             # Legacy reference code
```

## Documentation

See `doc/` for research plan and evaluation protocol.
