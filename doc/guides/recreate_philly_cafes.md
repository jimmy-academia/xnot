# Recreating the philly_cafes Dataset

Step-by-step guide to reproduce the philly_cafes benchmark dataset from raw Yelp data.

## Prerequisites

### 1. Yelp Academic Dataset

Download the Yelp Open Dataset from https://www.yelp.com/dataset

Required files:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

Place in `preprocessing/raw/`:
```
preprocessing/raw/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_review.json
└── yelp_academic_dataset_user.json
```

### 2. Environment Setup

```bash
source .venv/bin/activate
# Ensure rich is installed for interactive UI
```

### 3. API Keys

For LLM-assisted curation scoring:
```bash
export OPENAI_API_KEY="sk-..."
# Or configure in ../.openaiapi
```

---

## Stage 1: Curation

Filter Philadelphia cafes from raw Yelp data.

### Command

```bash
python -m preprocessing.curate
```

### Interactive Selection

1. **City**: Search/select "Philadelphia"
2. **Categories**: Select "Coffee & Tea" and "Cafes"
3. **Mode**: Choose "Auto" for LLM-assisted filtering

### Output

```
preprocessing/output/philly_cafes/
├── restaurants.jsonl   # 112 filtered restaurants
├── reviews.jsonl       # 4302 reviews with user metadata
├── analysis.json       # Attribute distributions
└── meta.json           # Selection parameters
```

### Verification

```bash
# Check restaurant count
wc -l preprocessing/output/philly_cafes/restaurants.jsonl
# Expected: 112

# Check review count
wc -l preprocessing/output/philly_cafes/reviews.jsonl
# Expected: ~4302
```

---

## Stage 2: Ground Truth Selection

Select 50 restaurants and design 100 requests using Claude Code.

### Overview

This stage uses Claude Code to:
1. Analyze attribute distributions
2. Select 50 diverse restaurants
3. Design 100 requests across 10 complexity groups
4. Ensure each request matches exactly 1 restaurant

### Kickstart Prompt

```
Create a ground truth evaluation dataset from the curated source data.

**Source files:**
- preprocessing/output/philly_cafes/restaurants.jsonl - 112 curated restaurants
- preprocessing/output/philly_cafes/reviews.jsonl - 4302 reviews
- preprocessing/output/philly_cafes/analysis.json - attribute distributions

**Reference:**
- preprocessing/records/philly_cafes_selection.md - process documentation
- doc/reference/evidence_types.md - evidence type specifications
- doc/reference/request_structure.md - request JSON schema

**Target output:**
- data/philly_cafes/restaurants.jsonl - 50 restaurants with diverse features
- data/philly_cafes/reviews.jsonl - 20 reviews per restaurant (1000 total)
- data/philly_cafes/requests.jsonl - 100 requests in 10 groups (G01-G10)
- data/philly_cafes/user_mapping.json - synthetic social data for G09-G10

**Process:**
1. Analyze analysis.json to identify rare/unique features
2. Select 50 restaurants with diverse, unique feature combinations
3. For each restaurant, select 20 reviews (mix recent + sampled older)
4. Design 100 requests across 10 groups per doc/reference/request_structure.md
5. For G09-G10, create synthetic user_mapping.json with friend graph
6. Validate all requests match exactly 1 restaurant
```

### Feature Analysis Prompt

Use this to identify unique features for single-answer requests:

```
Analyze the condition_matrix.json and identify:
1. Which restaurants have unique identifying conditions (only 1 restaurant has this feature)
2. Which restaurants need combination anchors (2-3 features together are unique)
3. Recommend 50 restaurants that maximize condition coverage diversity

Output a table with:
- Restaurant name
- Key unique/rare features
- Suggested anchor combinations for single-answer requests
```

### Request Design Prompt

For each group, use:

```
Design 10 requests for group G0X with structure [pattern from request_structure.md].

Requirements:
- Each request must match exactly 1 restaurant from the 50 selected
- Use bottom-up anchor-first methodology (start with unique features)
- Vary complexity within the group
- Ensure diverse restaurant coverage (don't reuse same gold answer)
- Write natural language text that signals all conditions

For each request, provide:
- id, group, scenario (persona name)
- text (natural language)
- shorthand (compact notation)
- structure (full JSON)
- gold_restaurant (business_id)
```

### Social Data Generation Prompt (G09-G10)

```
Create synthetic social data for G09 (1-hop) and G10 (2-hop) requests.

Generate user_mapping.json with:
1. user_names: Map user_id -> human name (e.g., "Alice", "Bob")
2. friend_graph: Map user_id -> list of friend user_ids
3. restaurant_reviews: Map restaurant_idx -> list of [user_id, pattern_mentioned]

Requirements:
- Create 20-30 synthetic users with distinct names
- Build friend graph ensuring some users are:
  - Direct friends of named users (for 1-hop queries)
  - Friends-of-friends (for 2-hop queries)
- For each G09/G10 request, ensure exactly 1 restaurant has a qualifying review
```

---

## Stage 3: Validation

Validate all requests match exactly 1 restaurant.

### Command

```bash
python -m data.validate philly_cafes
```

### Expected Output

```
Dataset: philly_cafes
  Restaurants: 50
  Reviews: 1000
  Requests: 100

Validation: 100/100 = 100%
All 100 requests validated successfully!

Saved: data/philly_cafes/groundtruth.jsonl
```

### Troubleshooting

**no_match** - No restaurant satisfies conditions:
- Check evidence paths are correct
- Verify attribute values match Yelp format (e.g., `"True"` not `true`)

**multi_match** - Multiple restaurants match:
- Add more anchor conditions to make unique
- Use rarer features

**gold_not_match** - Gold restaurant doesn't satisfy:
- Check business_id is correct
- Verify all conditions in structure are satisfied

---

## Stage 4: Request Text Rewriting

Ensure request texts explicitly signal all conditions.

### Analyze

```bash
python -m preprocessing.rewrite_requests philly_cafes --analyze
```

This shows which conditions are missing from request texts.

### Preview

```bash
python -m preprocessing.rewrite_requests philly_cafes --dry-run
```

### Rewrite

```bash
python -m preprocessing.rewrite_requests philly_cafes
```

This overwrites `data/philly_cafes/requests.jsonl` with improved texts.

### Example

```
Original: "I need a cafe with a drive-thru option"
Missing: GoodForKids, HasTV conditions not signaled

Rewritten: "Looking for a cafe with a drive-thru, that's kid-friendly, and without TVs"
```

---

## Final Dataset Structure

```
data/philly_cafes/
├── restaurants.jsonl      # 50 restaurants
├── reviews.jsonl          # 1000 reviews (20 per restaurant)
├── requests.jsonl         # 100 requests (10 per group)
├── groundtruth.jsonl      # Validation results
├── user_mapping.json      # Synthetic social data
├── condition_matrix.json  # Full condition satisfaction matrix
├── README.md              # Dataset overview
├── statistics.md          # Detailed statistics
├── requests_reference.md  # Request-answer reference
└── condition_summary.md   # Condition analysis
```

---

## Request Group Checklist

Verify each group has 10 requests:

| Group | Structure | IDs | Verified |
|-------|-----------|-----|----------|
| G01 | Simple AND | R01-R10 | [ ] |
| G02 | Simple OR | R11-R20 | [ ] |
| G03 | AND-OR Combination | R21-R30 | [ ] |
| G04 | Review Metadata Weighting | R31-R40 | [ ] |
| G05 | Triple OR with Anchor | R41-R50 | [ ] |
| G06 | Nested OR+AND | R51-R60 | [ ] |
| G07 | Chained OR | R61-R70 | [ ] |
| G08 | Unbalanced Structure | R71-R80 | [ ] |
| G09 | Direct Friends (1-hop) | R81-R90 | [ ] |
| G10 | Social Circle (2-hop) | R91-R100 | [ ] |

---

## Validation Checklist

After recreation:

- [ ] 50 restaurants in restaurants.jsonl
- [ ] 1000 reviews in reviews.jsonl (20 per restaurant)
- [ ] 100 requests in requests.jsonl
- [ ] All requests pass validation (100%)
- [ ] Each group has 10 requests
- [ ] G09-G10 have user_mapping.json
- [ ] Request texts signal all conditions

---

## Common Issues

### Yelp Attribute Format

Yelp stores attributes as Python repr strings:
```python
"True"           # Boolean true
"False"          # Boolean false
"u'free'"        # String "free"
"'quiet'"        # String "quiet"
"{'romantic': False, 'casual': True}"  # Dict
```

Use `parse_attr_value()` in validate.py to handle these.

### Missing Attributes

Many restaurants have `None` for optional attributes. Handle with:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "WiFi"],
    "true": "u'free'",
    "missing": 0
  }
}
```

### Hours Format

Hours use `"H:M-H:M"` format:
- `"8:0-22:0"` = 8 AM to 10 PM
- `"0:0-0:0"` = Closed

### Social Filter Data

For G09-G10, ensure `user_mapping.json` exists with:
- All user names referenced in requests
- Friend graph supporting both 1-hop and 2-hop queries
- Restaurant reviews with matching patterns

---

## Related Documentation

- [Evidence Types Reference](../reference/evidence_types.md)
- [Request Structure Reference](../reference/request_structure.md)
- [Creating New Benchmarks](create_new_benchmark.md)
- [Original Selection Record](../../preprocessing/records/philly_cafes_selection.md)
