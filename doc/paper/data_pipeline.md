# Data Pipeline

How data flows through the system, from raw sources to evaluation results.

## Pipeline Overview

```
Raw Yelp Data
      │
      ▼ [Stage 1: Curation]
preprocessing/output/{name}/
      │
      ▼ [Stage 2: Selection]
data/{name}/
      │
      ▼ [Stage 3: Validation]
data/{name}/groundtruth.jsonl
      │
      ▼ [Stage 4: Evaluation]
results/{mode}/{run-name}/
```

---

## Stage 1: Curation

**Purpose**: Filter raw Yelp data to domain-specific subset.

### Input

```
preprocessing/raw/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_user.json
```

### Process

```python
# preprocessing/curate.py
def curate(city, categories, threshold):
    """
    1. Filter businesses by city and categories
    2. Filter to businesses with min_reviews threshold
    3. Optionally score with LLM for quality
    4. Join reviews with user metadata
    5. Output curated dataset
    """
```

### Output

```
preprocessing/output/{name}/
├── restaurants.jsonl   # All filtered restaurants
├── reviews.jsonl       # All reviews with user metadata
├── analysis.json       # Attribute distributions
└── meta.json           # Selection parameters
```

### Code Location

| File | Purpose |
|------|---------|
| `preprocessing/curate.py` | Interactive curation tool |
| `preprocessing/analyze.py` | Generate analysis.json |

---

## Stage 2: Selection

**Purpose**: Select subset for evaluation and design requests.

### Input

```
preprocessing/output/{name}/
├── restaurants.jsonl
├── reviews.jsonl
└── analysis.json
```

### Process

This stage uses Claude Code with prompts to:
1. Analyze feature distributions
2. Select N items with diverse features
3. Design M requests with logical structures
4. Ensure unique satisfaction (1 item per request)

### Output

```
data/{name}/
├── restaurants.jsonl      # Selected items
├── reviews.jsonl          # Selected reviews
├── requests.jsonl         # Designed requests
├── user_mapping.json      # Social data (optional)
└── condition_matrix.json  # Feature satisfaction matrix
```

### Code Location

| File | Purpose |
|------|---------|
| `preprocessing/records/{name}_selection.md` | Selection documentation |
| `doc/guides/recreate_philly_cafes.md` | Reproduction guide |

---

## Stage 3: Validation

**Purpose**: Verify each request matches exactly 1 item.

### Input

```
data/{name}/
├── restaurants.jsonl
├── reviews.jsonl
└── requests.jsonl
```

### Process

```python
# data/validate.py
def validate_dataset(name):
    """
    For each request:
    1. Parse logical structure
    2. Evaluate against all items
    3. Count matches
    4. Report status (ok, no_match, multi_match)
    5. Generate groundtruth.jsonl
    """
```

### Output

```
data/{name}/
└── groundtruth.jsonl   # Request → gold mapping
```

### Code Location

| File | Purpose |
|------|---------|
| `data/validate.py` | Validation implementation |
| `doc/reference/evidence_types.md` | Evidence type specs |

---

## Stage 4: Evaluation

**Purpose**: Run methods on benchmark and measure performance.

### Input

```
data/{name}/
├── restaurants.jsonl
├── reviews.jsonl
├── requests.jsonl
└── groundtruth.jsonl
```

### Process

```python
# run/orchestrate.py
def run_evaluation_loop(requests, restaurants, reviews, groundtruth, method):
    """
    For each request:
    1. Sample N candidates (include gold)
    2. Shuffle candidates (position bias mitigation)
    3. Format query for LLM
    4. Call method
    5. Compute metrics
    6. Log results
    """
```

### Output

```
results/{mode}/{run-name}/
├── config.json         # Run configuration
├── results_N.jsonl     # Per-request results
├── usage.jsonl         # Token usage
└── debug.log           # Full traces
```

### Code Location

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `run/orchestrate.py` | Evaluation loop |
| `run/evaluate.py` | Metrics computation |
| `run/scaling.py` | Scaling experiments |

---

## Data Schemas

### restaurants.jsonl

```json
{
  "business_id": "unique_id",
  "name": "Restaurant Name",
  "city": "Philadelphia",
  "state": "PA",
  "stars": 4.5,
  "review_count": 150,
  "attributes": {
    "WiFi": "u'free'",
    "DriveThru": "True",
    "Ambience": "{'casual': True, 'romantic': False}",
    "RestaurantsPriceRange2": "2"
  },
  "hours": {
    "Monday": "9:0-22:0",
    "Tuesday": "9:0-22:0"
  }
}
```

### reviews.jsonl

```json
{
  "business_id": "unique_id",
  "review_id": "review_unique_id",
  "text": "Great coffee and cozy atmosphere...",
  "stars": 5,
  "date": "2024-01-15",
  "useful": 10,
  "funny": 2,
  "cool": 5,
  "user": {
    "user_id": "user_unique_id",
    "name": "John D.",
    "review_count": 50,
    "fans": 10,
    "elite": ["2023", "2024"]
  }
}
```

### requests.jsonl

```json
{
  "id": "R01",
  "group": "G01",
  "scenario": "Busy Parent",
  "text": "Looking for a cafe with a drive-thru...",
  "shorthand": "AND(drive_thru, good_for_kids, no_tv)",
  "structure": {
    "op": "AND",
    "args": [
      {"aspect": "drive_thru", "evidence": {"kind": "item_meta", "path": ["attributes", "DriveThru"], "true": "True"}}
    ]
  },
  "gold_restaurant": "business_id"
}
```

### groundtruth.jsonl

```json
{
  "request_id": "R01",
  "gold_restaurant": "business_id",
  "gold_idx": 5,
  "status": "ok"
}
```

### user_mapping.json (for G09-G10)

```json
{
  "user_names": {
    "user_001": "Alice",
    "user_002": "Bob"
  },
  "friend_graph": {
    "user_001": ["user_002", "user_003"],
    "user_002": ["user_001"]
  },
  "restaurant_reviews": {
    "0": [["user_001", "cozy"], ["user_002", "recommend"]],
    "1": [["user_003", "love"]]
  }
}
```

---

## Results Curation

Results are stored in structured format for analysis.

### Current Storage

```
results/
├── dev/                    # Development runs (gitignored)
│   └── 001_test/
│       ├── config.json
│       └── results_1.jsonl
│
└── benchmarks/             # Tracked results
    └── final_experiment/
        └── ...
```

### Result Files

**config.json**:
```json
{
  "method": "anot",
  "data": "philly_cafes",
  "candidates": 20,
  "attack": "none",
  "model": "gpt-4o-mini",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**results_N.jsonl** (per-request):
```json
{
  "request_id": "R01",
  "group": "G01",
  "gold_idx": 5,
  "prediction": 5,
  "rank": 1,
  "hit_at_1": true,
  "hit_at_5": true,
  "latency_ms": 2340,
  "tokens_in": 1500,
  "tokens_out": 50
}
```

### Aggregation

```python
# utils/aggregate.py
def aggregate_benchmark_runs(method, data, attack=None):
    """
    Aggregate results across runs for summary statistics.

    Returns:
        - Mean/std for Hits@1, Hits@5, MRR
        - Per-group breakdown
        - Token usage summary
    """
```

### Future: Results Curation Code

*Note: Dedicated results curation/visualization code is planned but not yet implemented. Current analysis is done via aggregate functions.*

---

## Directory Structure Summary

```
anot/
├── preprocessing/           # Stage 1 & 2 inputs
│   ├── curate.py           # Curation tool
│   ├── analyze.py          # Analysis generator
│   ├── raw/                # Raw Yelp data (gitignored)
│   ├── output/{name}/      # Curated outputs
│   └── records/            # Selection documentation
│
├── data/                   # Stage 2 & 3 outputs
│   ├── validate.py         # Validation tool
│   ├── loader.py           # Data loading utilities
│   └── {name}/             # Benchmark datasets
│       ├── restaurants.jsonl
│       ├── reviews.jsonl
│       ├── requests.jsonl
│       └── groundtruth.jsonl
│
├── methods/                # Stage 4 methods
│   ├── cot.py
│   ├── anot/
│   └── ...
│
├── run/                    # Stage 4 orchestration
│   ├── orchestrate.py
│   ├── evaluate.py
│   └── scaling.py
│
└── results/                # Stage 4 outputs
    ├── dev/                # Development (gitignored)
    └── benchmarks/         # Tracked results
```

---

## Related Documentation

- [Recreating philly_cafes](../guides/recreate_philly_cafes.md) - Step-by-step reproduction
- [Creating New Benchmarks](../guides/create_new_benchmark.md) - New dataset guide
- [Evidence Types](../reference/evidence_types.md) - Validation logic
- [Request Structure](../reference/request_structure.md) - Request JSON schema
- [Running Experiments](../guides/run_experiments.md) - Evaluation guide
