# Creating a New Benchmark Dataset

Guide to creating a new evaluation benchmark with a different dataset (e.g., different city, category, or data source).

## Overview

The ANoT framework supports creating new benchmarks from any structured data source. This guide covers:
1. Choosing a domain and data source
2. Running the curation pipeline
3. Analyzing features for unique identification
4. Designing request groups
5. Implementing custom evidence types (if needed)
6. Validating and documenting

---

## Step 1: Choose Domain and Data Source

### Requirements

Your data source should have:
- **Items**: Entities to be ranked (restaurants, products, venues, etc.)
- **Attributes**: Structured metadata for each item
- **Reviews/Descriptions**: Text content for pattern matching
- **Sufficient volume**: At least 100+ items, 20+ items with 20+ reviews each

### Supported Sources

**Yelp Academic Dataset** (built-in):
- Run `python -m preprocessing.curate` with city/category filters

**Custom Data**:
- Convert to JSONL format matching expected schema (see below)

### Data Schema

**restaurants.jsonl** (items):
```json
{
  "business_id": "unique_id",
  "name": "Item Name",
  "attributes": {
    "Category": "value",
    "Feature1": "True",
    "Feature2": "u'option'"
  },
  "hours": {
    "Monday": "9:0-17:0",
    "Tuesday": "9:0-17:0"
  }
}
```

**reviews.jsonl** (text content):
```json
{
  "business_id": "unique_id",
  "text": "Review text content...",
  "stars": 4,
  "user": {
    "user_id": "user_123",
    "review_count": 50,
    "elite": ["2023", "2024"],
    "fans": 10
  },
  "useful": 5,
  "date": "2024-01-15"
}
```

---

## Step 2: Run Curation Pipeline

### For Yelp Data

```bash
# Interactive mode
python -m preprocessing.curate

# Non-interactive
python -m preprocessing.curate \
    --name my_benchmark \
    --city "City Name" \
    --category "Category1" "Category2"
```

### For Custom Data

Place files in `preprocessing/output/{name}/`:
```
preprocessing/output/my_benchmark/
├── restaurants.jsonl   # All items
├── reviews.jsonl       # All reviews/text
└── meta.json           # {"city": "...", "categories": [...]}
```

### Generate Analysis

```bash
python -m preprocessing.analyze my_benchmark
```

Output: `preprocessing/output/my_benchmark/analysis.json`

---

## Step 3: Analyze Features

### Feature Distribution Analysis

Read `analysis.json` to identify:

**Unique features** (1 item has this):
```
DriveThru=True: 1 item → strong anchor
```

**Rare features** (2-5 items):
```
CoatCheck=True: 2 items → usable with another condition
```

**Common features** (many items):
```
WiFi=free: 80+ items → not useful alone
```

### Building a Condition Matrix

Use Claude Code to build a condition satisfaction matrix:

```
Analyze the curated data and build a condition matrix.

For each potential condition:
1. Count how many items satisfy it
2. Identify which specific items satisfy it
3. Flag unique conditions (1 item)
4. Flag rare conditions (2-5 items)

Output format:
| Condition | Count | Items |
|-----------|-------|-------|
| DriveThru=True | 1 | [Milkcrate Cafe] |
| CoatCheck=True | 2 | [Front Street, Tria] |
```

### Bottom-Up Anchor-First Methodology

Design requests starting from unique/rare features:

1. **Identify anchors**: Features that narrow to 1-5 items
2. **Add complexity**: OR conditions for evaluation challenge
3. **Verify uniqueness**: Ensure exactly 1 item matches
4. **Write natural text**: Request should read naturally

---

## Step 4: Design Request Groups

### Standard Groups

| Group | Structure | Purpose |
|-------|-----------|---------|
| G01 | `AND(a, b, c)` | Basic conjunction |
| G02 | `AND(anchor, OR(a, b))` | Simple disjunction |
| G03 | `AND(a, OR(b, c))` | AND-OR combination |
| G04 | `AND(a, review_meta_*)` | Credibility weighting |
| G05 | `AND(a, OR(b, c, d))` | Triple disjunction |
| G06 | `AND(a, OR(AND(b,c), AND(d,e)))` | Nested structures |
| G07 | `AND(a, OR(b,c), OR(d,e))` | Chained disjunctions |
| G08 | `AND(a, OR(b, AND(c,d)))` | Unbalanced structures |
| G09 | `1HOP([friends], pattern)` | Direct social filter |
| G10 | `2HOP([friends], pattern)` | Extended social filter |

### Request Design Prompt

For each group:

```
Design 10 requests for group G0X with structure [pattern].

Domain: {your_domain}
Items: {your_item_count} {your_item_type}

Requirements:
- Each request matches exactly 1 item
- Use bottom-up anchor-first methodology
- Vary complexity within the group
- Ensure diverse item coverage

For each request provide:
- id, group, scenario
- text (natural language)
- shorthand (compact notation)
- structure (JSON)
- gold_{item_type} (unique_id)
```

### Social Groups (G09-G10)

If using social filtering, create synthetic data:

```
Create synthetic social data for G09-G10.

Generate user_mapping.json:
1. user_names: {user_id: "Name"} for 20-30 users
2. friend_graph: {user_id: [friend_ids]} for social connections
3. {item}_reviews: {item_idx: [[user_id, pattern], ...]}

Ensure:
- Some direct friends (1-hop)
- Some friends-of-friends (2-hop)
- Each G09/G10 request has exactly 1 qualifying item
```

---

## Step 5: Implement Custom Evidence Types

### When Needed

Implement custom evidence types when:
- Your domain has unique data structures
- Standard evidence types don't cover your needs
- You need specialized evaluation logic

### Implementation Steps

1. **Define the schema** in your request structures
2. **Add evaluation function** in `data/validate.py`
3. **Register in dispatcher** (`evaluate_condition()`)
4. **Document** in `doc/reference/evidence_types.md`

See [Adding Evidence Types](add_evidence_type.md) for detailed instructions.

### Example: Custom Scoring

```python
# In data/validate.py
def evaluate_custom_score(item: dict, evidence_spec: dict) -> int:
    """Evaluate custom scoring condition."""
    path = evidence_spec.get("path", [])
    min_score = evidence_spec.get("min_score", 0)

    value = get_nested_value(item, path)
    if value is None:
        return evidence_spec.get("missing", 0)

    return 1 if float(value) >= min_score else -1

# Register in evaluate_condition()
elif kind == "custom_score":
    return evaluate_custom_score(item, evidence_spec)
```

---

## Step 6: Validate and Document

### Run Validation

```bash
python -m data.validate my_benchmark
```

All requests should pass (100%).

### Generate Statistics

Create `data/my_benchmark/statistics.md` with:
- Total items, reviews, requests
- Request group distribution
- Evidence type usage
- Item coverage

### Document Selection Process

Create `preprocessing/records/my_benchmark_selection.md` documenting:
- Selection criteria
- Feature analysis
- Unique combinations identified
- Request design rationale

---

## Checklist

### Data Preparation
- [ ] Source data in correct format
- [ ] Sufficient volume (100+ items, 20+ reviews each)
- [ ] Analysis.json generated
- [ ] Feature distribution analyzed

### Request Design
- [ ] 10 groups defined (or subset)
- [ ] 10 requests per group
- [ ] Each request has unique answer
- [ ] Natural language texts written
- [ ] Shorthand notation included

### Validation
- [ ] All requests pass (100%)
- [ ] groundtruth.jsonl generated
- [ ] No multi-match issues
- [ ] No gold_not_match issues

### Documentation
- [ ] statistics.md created
- [ ] Selection process documented
- [ ] README.md updated
- [ ] Evidence types documented

---

## Directory Structure

```
data/my_benchmark/
├── restaurants.jsonl      # Selected items
├── reviews.jsonl          # Selected reviews
├── requests.jsonl         # Designed requests
├── groundtruth.jsonl      # Validation results
├── user_mapping.json      # Social data (if using G09-G10)
├── condition_matrix.json  # Feature satisfaction matrix
├── README.md              # Dataset overview
└── statistics.md          # Dataset statistics

preprocessing/
├── output/my_benchmark/   # Curated source data
│   ├── restaurants.jsonl
│   ├── reviews.jsonl
│   ├── analysis.json
│   └── meta.json
└── records/
    └── my_benchmark_selection.md
```

---

## Example: NYC Hotels Benchmark

Hypothetical example for a hotel recommendation benchmark:

### Step 1: Data Source
- Yelp Hotels category in New York City
- 200 hotels, 5000 reviews

### Step 2: Feature Analysis
```
Unique:
- Rooftop pool: 1 hotel
- Pet spa: 2 hotels

Rare:
- Michelin restaurant: 3 hotels
- Helicopter pad: 2 hotels

Common:
- Free WiFi: 180+ hotels
- Gym: 150+ hotels
```

### Step 3: Request Groups
- G01: `AND(rooftop_pool, parking)` → 1 hotel
- G02: `AND(michelin, OR(spa, gym))` → requires combination
- G04: `AND(luxury, elite_reviewers_mention_service)` → credibility
- G09: `1HOP(['Alice'], 'recommend')` → friend filter

### Step 4: Validation
```bash
python -m data.validate nyc_hotels
# Validation: 100/100 = 100%
```

---

## Related Documentation

- [Evidence Types Reference](../reference/evidence_types.md)
- [Request Structure Reference](../reference/request_structure.md)
- [Adding Evidence Types](add_evidence_type.md)
- [Recreating philly_cafes](recreate_philly_cafes.md)
