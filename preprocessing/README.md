# Data Preprocessing Pipeline

Two-stage process from raw Yelp data to evaluation dataset.

## Stage 1: Curation

Filter restaurants from raw Yelp data with interactive selection and LLM-assisted scoring.

### Commands

```bash
# Interactive mode (recommended)
python -m preprocessing.curate

# Non-interactive with CLI args
python -m preprocessing.curate \
    --name {name} \
    --city {city} \
    --category "Category1" "Category2"

# Generate attribute distributions
python -m preprocessing.analyze {name}
```

### Output

```
preprocessing/output/{name}/
├── restaurants.jsonl   # All filtered restaurants with LLM scores
├── reviews.jsonl       # All reviews with user metadata
├── analysis.json       # Attribute distributions for GT design
└── meta.json           # Creation parameters (city, categories, threshold)
```

### Interactive Mode

1. **City selection** - Paginated list with search
2. **Category selection** - Multi-select (e.g., `1,3,5`)
3. **Mode selection**:
   - **Auto**: LLM batch scoring, keeps restaurants above threshold
   - **Manual**: Review each restaurant one by one

---

## Stage 2: Ground Truth Selection

Using Claude Code to select a subset for evaluation:
- 20 restaurants (diverse features for single-answer requests)
- 20 reviews per restaurant (400 total)
- 50 requests in 5 groups (G01-G05)

### Output

```
data/{name}/
├── restaurants.jsonl   # 20 selected restaurants
├── reviews.jsonl       # 400 selected reviews
├── requests.jsonl      # 50 requests with structure and gold_restaurant
└── meta.json           # Dataset metadata
```

### Process Record

All selection decisions documented in: `preprocessing/output/{name}/ground_truth_selection.md`

### Kickstart Prompt

```
Create a ground truth evaluation dataset from the curated source data.

**Source files:**
- preprocessing/output/{name}/restaurants.jsonl - all curated restaurants
- preprocessing/output/{name}/reviews.jsonl - all reviews
- preprocessing/output/{name}/analysis.json - attribute distributions

**Reference:**
- preprocessing/output/philly_cafes/ground_truth_selection.md - example process record
- doc/evaluation_spec.md - request structure format and group definitions

**Target output:**
- data/{name}/restaurants.jsonl - 20 restaurants with diverse features
- data/{name}/reviews.jsonl - 20 reviews per restaurant (400 total)
- data/{name}/requests.jsonl - 50 requests in 5 groups (G01-G05), each with exactly 1 correct answer
- data/{name}/meta.json - dataset metadata

**Process:**
1. Read analysis.json to identify rare/unique features for single-answer requests
2. Select 20 restaurants ensuring: min 20 reviews, diverse features, unique combinations
3. For each restaurant, select 20 reviews (mix recent + sampled older)
4. Design 50 requests across 5 groups per doc/evaluation_spec.md
5. Document all decisions in preprocessing/output/{name}/ground_truth_selection.md
```

---

## Repeating the Process

1. Run Stage 1 curation for new city/categories
2. Use Claude Code with kickstart prompt to perform Stage 2 selection
3. Verify all 50 requests have exactly 1 correct answer
4. Document decisions in `ground_truth_selection.md`

---

## Directory Structure

```
preprocessing/
├── curate.py              # Interactive curation tool (Stage 1)
├── analyze.py             # Analysis for GT constraint design
├── raw/                   # Raw Yelp academic dataset (gitignored)
├── output/                # Curated selections
│   └── {name}/
│       ├── restaurants.jsonl
│       ├── reviews.jsonl
│       ├── analysis.json
│       ├── meta.json
│       └── ground_truth_selection.md
└── README.md
```
