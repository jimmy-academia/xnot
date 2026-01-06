# Philadelphia Cafes Benchmark Dataset

A structured benchmark dataset for evaluating LLM reasoning on restaurant recommendation tasks with formal logical structures.

## Overview

- **50 restaurants** with attributes and reviews
- **80 requests** across 8 structural complexity groups (G01-G08)
- **100% validation** - each request matches exactly one restaurant

## Dataset Files

| File | Description |
|------|-------------|
| `restaurants.jsonl` | 50 Philadelphia cafe profiles with attributes |
| `reviews.jsonl` | Customer reviews for each restaurant |
| `requests.jsonl` | 80 benchmark requests with logical structures |
| `groundtruth.jsonl` | Gold answers and validation status |
| `condition_matrix.json` | Full condition satisfaction matrix |
| `condition_summary.md` | Human-readable condition analysis |

## Request Groups

### G01: Simple AND (R00-R09)
**Structure**: `AND(condition1, condition2, condition3)`

Basic conjunction of 2-4 conditions. Tests fundamental attribute matching.

**Example R00**:
```
"Looking for a quiet cafe with free WiFi that's good for studying"
AND(noise_quiet, wifi_free, study_reviews)
Gold: [0] Milkcrate Cafe
```

**Complexity**: Low - straightforward attribute checking

---

### G02: Simple OR (R10-R19)
**Structure**: `OR(condition1, condition2)` or `AND(anchor, OR(...))`

Tests disjunctive reasoning - any one condition suffices.

**Example R12**:
```
"Looking for a cafe that's either good for brunch or has outdoor seating"
OR(meal_brunch, outdoor_yes)
Gold: [1] Tria Cafe Rittenhouse
```

**Complexity**: Low-Medium - requires checking multiple paths

---

### G03: AND-OR Combination (R20-R29)
**Structure**: `AND(condition1, OR(condition2, condition3))`

Nested structure with anchor + disjunction.

**Example R21**:
```
"Looking for a mid-priced cafe with either 'cozy' or 'comfortable' mentioned in reviews"
AND(price_mid, OR(cozy, comfortable_reviews))
Gold: [1] Tria Cafe Rittenhouse
```

**Complexity**: Medium - requires understanding nested logic

---

### G04: Review Metadata Weighting (R30-R39)
**Structure**: `AND(conditions, credibility_count_condition)`

Uses credibility-count evaluation: "At least N credible reviewers (above percentile) mention pattern"

**Evaluation Logic**:
- Credibility threshold: 50th percentile of non-zero metadata values
- Minimum credible matches: 2 reviewers must agree
- Weight fields: `review_count`, `fans`, `elite` years, `useful` votes

**Example R32**:
```
"Looking for a cafe with a full bar, where elite reviewers mention 'love', without coat check, offers delivery, and good for dinner"
AND(full_bar, no_coat_check, elite_love, delivery, dinner)
Gold: [9] Gran Caffe L'Aquila
```

**Complexity**: Medium-High - requires credibility-aware evaluation

---

### G05: Triple OR with Anchor (R40-R49)
**Structure**: `AND(anchor1, anchor2, OR(opt1, opt2, opt3))`

Multiple anchoring conditions with three-way disjunction.

**Example R40**:
```
"Looking for a cafe with a drive-thru that has either 'coffee', 'breakfast', or 'friendly' mentioned"
AND(drive_thru, OR(coffee_reviews, breakfast_reviews, friendly_reviews))
Gold: [0] Milkcrate Cafe
```

**Complexity**: Medium - unique anchor simplifies, OR adds options

---

### G06: Nested OR+AND (R50-R59)
**Structure**: `AND(anchor, OR(AND(a,b), AND(c,d)))`

Disjunction of conjunctions - either (A AND B) or (C AND D).

**Example R54**:
```
"Looking for a cafe with free corkage that's either (organic AND music) or (work AND wifi)"
AND(byob_corkage_free, OR(AND(organic, music), AND(work, wifi)))
Gold: [42] Mugshots Coffeehouse
```

**Complexity**: High - requires evaluating nested AND blocks within OR

---

### G07: Chained OR (R60-R69)
**Structure**: `AND(anchor, OR(a,b), OR(c,d))`

Multiple independent OR blocks that all must have at least one match.

**Example R65**:
```
"Looking for a hipster cafe good for lunch, open Monday afternoon,
 that either has 'sandwich' or 'work' mentioned,
 and either 'meeting' or popular reviewers mention 'work'"
AND(hipster, lunch, hours_monday_afternoon, OR(sandwich,work), OR(meeting,popular_work))
Gold: [47] Rocket Cat Cafe
```

**Complexity**: High - multiple disjunctions must all be satisfied

---

### G08: Unbalanced Structure (R70-R79)
**Structure**: `AND(anchor, simple, OR(opt, AND(nested1, nested2)))`

Asymmetric structure with one simple OR option and one complex AND option.

**Example R70**:
```
"Looking for a quiet cafe with beer and wine that's either
 where reviews mention 'slow', or both elite reviewers mention 'work' and 'best'"
AND(alcohol_beer_wine, noise_quiet, OR(slow_reviews, AND(elite_work, best)))
Gold: [7] Swiss Haus Cafe & Pastry Bar
```

**Complexity**: High - unbalanced complexity in OR branches

---

## Evidence Types Used

| Type | Count | Description |
|------|-------|-------------|
| `item_meta` | ~200 | Restaurant attributes (price, noise, WiFi, etc.) |
| `item_meta_hours` | ~30 | Operating hour conditions |
| `review_text` | ~100 | Pattern matching in reviews |
| `review_meta` | ~20 | Credibility-count patterns (G04) |

## Restaurant Coverage

All 50 restaurants are used as gold answers, with distribution across groups ensuring each restaurant appears 1-3 times.

| Restaurant Index | Usage Count | Example Requests |
|------------------|-------------|------------------|
| [0] Milkcrate Cafe | 3 | R00, R30, R40 |
| [1] Tria Cafe | 3 | R01, R12, R41 |
| [2] Front Street Cafe | 2 | R02, R17 |
| ... | ... | ... |

## Validation

```bash
# Validate all requests
.venv/bin/python -m data.validate philly_cafes

# Expected output:
# Validation: 80/80 = 100%
# All 80 requests validated successfully!
```

## Usage in Evaluation

```python
from data.validate import load_jsonl

# Load data
restaurants = load_jsonl('data/philly_cafes/restaurants.jsonl')
requests = load_jsonl('data/philly_cafes/requests.jsonl')
groundtruth = load_jsonl('data/philly_cafes/groundtruth.jsonl')

# Each request has:
# - id: "R00" to "R79"
# - group: "G01" to "G08"
# - text: Natural language request
# - structure: Formal logical structure
# - gold_restaurant: Business ID of correct answer

# Groundtruth has:
# - request_id: Matches request.id
# - gold_restaurant: Business ID
# - gold_idx: Index in restaurants list (0-49)
# - status: "ok" for all 80 requests
```

## Design Methodology

See `doc/condition_design.md` for the complete bottom-up anchor-first design methodology used to create this benchmark.

Key principles:
1. Build condition satisfaction matrix first
2. Identify unique identifiers for each restaurant
3. Design requests around anchors to ensure uniqueness
4. Add OR complexity for evaluation challenge
5. Validate 100% unique matches
