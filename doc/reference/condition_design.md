# Condition Design for Request Generation

This document explains the systematic **bottom-up anchor-first** approach to designing benchmark requests that ensure unique restaurant matches.

## Overview

The benchmark requires each request to match exactly **one** restaurant (no multi-match, no no-match). Achieving this requires careful coordination between request conditions and the underlying data.

### The Problem

Naive request generation fails because:
1. **Common conditions match many restaurants** - "good for brunch" might match 27/50 restaurants
2. **OR conditions expand matches** - `OR(coffee, tea, breakfast)` could match nearly all restaurants
3. **Rare conditions may match zero** - specific combinations might not exist in the dataset

### The Solution: Bottom-Up Anchor-First Design

Instead of designing requests top-down (what sounds good?) and hoping they match one restaurant, we:

1. **Build a condition satisfaction matrix** - enumerate all conditions and which restaurants satisfy each
2. **Identify unique identifiers** - find condition combinations that uniquely identify each restaurant
3. **Design requests around anchors** - start with the unique identifier, add OR complexity for evaluation challenge

## Key Insights from Development

### Insight 1: Condition Matrix is Essential

Before writing any requests, generate the full condition matrix:

```bash
.venv/bin/python data/scripts/condition_matrix.py
```

This reveals:
- **Unique conditions** (count=1): Can directly identify one restaurant
- **Rare conditions** (count=2-3): Need one disambiguator
- **Common conditions** (count>10): Cannot anchor a request alone

### Insight 2: Unique Identifiers are Minimal Condition Sets

Each restaurant needs a **minimal set of conditions** that uniquely identifies it:

| Restaurant | Unique Identifier | Why It Works |
|------------|-------------------|--------------|
| [0] Milkcrate Cafe | `drive_thru` | Only cafe with drive-thru |
| [1] Tria Cafe | `price_upscale` | Only price=3 restaurant |
| [29] Saxbys Rittenhouse | `cozy + has_tv + outdoor + friday_evening + budget` | Combination narrows to 1 |
| [49] Brown Street | `hidden_gem + brunch + outdoor + budget` | Rare review pattern + attributes |

### Insight 3: OR Blocks Add Evaluation Challenge Without Breaking Uniqueness

Once you have a unique identifier anchor, you can add OR blocks for evaluation complexity:

```
AND(
  anchor_conditions,        -- ensures uniqueness
  OR(condition_a, condition_b, condition_c)  -- adds reasoning challenge
)
```

The gold restaurant must satisfy at least one OR branch, but the anchor ensures only one restaurant can match overall.

### Insight 4: Credibility-Count for Weighted Reviews (G04)

Weighted review conditions use **credibility-count evaluation**:

**Semantics**: "At least N credible reviewers (above percentile) mention pattern"

**Parameters** (with defaults):
- `credibility_percentile`: 50 (above median)
- `min_credible_matches`: 2 (at least 2 agree)

**How it works**:
1. Extract metadata values (review_count, fans, elite years, useful votes)
2. Compute credibility threshold as the percentile value of non-zero reviews
3. Count credible reviewers (above threshold) who mention the pattern
4. Return 1 if count >= min_credible_matches, else -1

**Spec format**:
```json
{
  "kind": "review_text",
  "pattern": "coffee",
  "weight_by": {"field": ["user", "review_count"]}
}
```

This replaced the old threshold-based approach which was sensitive to distribution variance and required manual tuning per request.

### Insight 5: Regex Patterns in Validator Must Match Matrix Generator

The condition matrix generator uses regex patterns (`study|studying`), but the validator must also use regex matching. A substring-only validator will fail on patterns with `|` operators.

**Fix applied**: `data/validate.py:evaluate_review_text_pattern()` now uses `re.compile()` for pattern matching.

## Evidence Types

| Type | Description | Reliability | Example |
|------|-------------|-------------|---------|
| `item_meta` | Restaurant attributes | High | `{"kind": "item_meta", "path": ["attributes", "NoiseLevel"], "contains": "quiet"}` |
| `item_meta_hours` | Operating hours | High | `{"kind": "item_meta_hours", "path": ["hours", "Friday"], "true": "18:0-21:0"}` |
| `review_text` | Review pattern (regex) | Medium | `{"kind": "review_text", "pattern": "cozy|comfortable"}` |
| `review_meta` | Credibility-weighted patterns | Medium | `{"kind": "review_text", "pattern": "work", "weight_by": {"field": ["user", "review_count"]}}` |

### Evidence Type Distribution

For comprehensive benchmark coverage:
- **50-60%** `item_meta` conditions (attributes are most reliable)
- **10-15%** `item_meta_hours` conditions (time-based filtering)
- **25-30%** `review_text` conditions (natural language reasoning)
- **5-10%** `review_meta` conditions (credibility-weighted patterns for G04)

## Condition Categories

### Unique Anchors (Count=1)

These conditions uniquely identify a single restaurant:

| Condition | Restaurant | Natural Phrase |
|-----------|------------|----------------|
| `drive_thru` | [0] Milkcrate Cafe | "with a drive-thru" |
| `price_upscale` | [1] Tria Cafe Rittenhouse | "upscale" |
| `wifi_paid` | [25] Frieda | "with paid WiFi" |
| `takeout_no` | [32] Saxbys | "dine-in only" |
| `ambience_intimate` | [21] K'Far Cafe | "intimate" |

### Rare Anchors (Count=2-3)

These need one disambiguator:

| Condition | Count | Restaurants | Disambiguators |
|-----------|-------|-------------|----------------|
| `noise_loud` | 2 | [3] MilkBoy, [4] Kung Fu Tea | `kids_no` vs `kids_yes` |
| `coat_check` | 2 | [1] Tria, [2] Front Street | `price_upscale` vs `price_mid` |
| `byob_corkage_free` | 2 | [28] One Shot, [42] Mugshots | `cozy` vs `hours_friday_evening` |
| `meal_dinner` | 2 | [9] Gran Caffe, [24] Manakeesh | `hours_friday_late_night` |
| `hours_friday_late_night` | 2 | [9] Gran Caffe, [10] Thirsty Dice | `outdoor_yes` vs `gluten_reviews` |

### Common Conditions (Use with Anchors)

| Condition | Count | Usage |
|-----------|-------|-------|
| `wifi_free` | 40 | Combine with rare anchors |
| `outdoor_yes` | 37 | Disambiguation or OR options |
| `coffee_reviews` | 50 | OR options (all restaurants match) |
| `friendly_reviews` | 49 | OR options |

## Request Structure Patterns

### G01-G04: Simple Structures (Manual Design)

These use straightforward AND/OR combinations with carefully selected conditions.

### G05: Triple OR with Anchor
```json
{
  "op": "AND",
  "args": [
    {"aspect": "anchor1", "evidence": {...}},
    {"aspect": "anchor2", "evidence": {...}},
    {"op": "OR", "args": [
      {"aspect": "opt1", "evidence": {"kind": "review_text", "pattern": "..."}},
      {"aspect": "opt2", "evidence": {"kind": "review_text", "pattern": "..."}},
      {"aspect": "opt3", "evidence": {"kind": "review_text", "pattern": "..."}}
    ]}
  ]
}
```

### G06: Nested OR+AND with Anchor
```json
{
  "op": "AND",
  "args": [
    {"aspect": "anchor", "evidence": {...}},
    {"op": "OR", "args": [
      {"op": "AND", "args": [{"aspect": "a1"}, {"aspect": "a2"}]},
      {"op": "AND", "args": [{"aspect": "b1"}, {"aspect": "b2"}]}
    ]}
  ]
}
```

### G07: Chained OR with Anchor
```json
{
  "op": "AND",
  "args": [
    {"aspect": "anchor", "evidence": {...}},
    {"op": "OR", "args": [{"aspect": "a1"}, {"aspect": "a2"}]},
    {"op": "OR", "args": [{"aspect": "b1"}, {"aspect": "b2"}]}
  ]
}
```

### G08: Unbalanced with Anchor
```json
{
  "op": "AND",
  "args": [
    {"aspect": "anchor", "evidence": {...}},
    {"aspect": "simple", "evidence": {...}},
    {"op": "OR", "args": [
      {"aspect": "opt1", "evidence": {...}},
      {"op": "AND", "args": [...]}
    ]}
  ]
}
```

## G09/G10: Social Filter Design

G09 (1-hop) and G10 (2-hop) requests use social filters to find restaurants based on friend recommendations.

### Structure Types

**Pure Social Filter** (simpler):
```json
{
  "op": "AND",
  "args": [
    {"aspect": "social_pattern", "evidence": {
      "kind": "review_text",
      "pattern": "boba",
      "social_filter": {"friends": ["Ivy"], "hops": 1},
      "min_matches": 1
    }}
  ]
}
```
Shorthand: `1HOP(['Ivy'], 'boba')`

**Compound Request** (non-social conditions + social filter):
```json
{
  "op": "AND",
  "args": [
    {"aspect": "budget", "evidence": {"kind": "item_meta", "path": ["attributes", "RestaurantsPriceRange2"], "true": "1"}},
    {"aspect": "no_outdoor", "evidence": {"kind": "item_meta", "path": ["attributes", "OutdoorSeating"], "true": "False"}},
    {"aspect": "social_taro", "evidence": {
      "kind": "review_text",
      "pattern": "taro",
      "social_filter": {"friends": ["Grace"], "hops": 1},
      "min_matches": 1
    }}
  ]
}
```
Shorthand: `AND(budget, no_outdoor, 1HOP(['Grace'], 'taro'))`

### Design Principles for Compound Requests

For compound requests `AND(conditions..., nHOP(...))`:

1. **Non-social conditions should narrow to 2+ restaurants** (not 1)
   - If non-social conditions uniquely identify a restaurant, the social filter is redundant
   - The social filter should be the **deciding factor** for final selection

2. **Avoid unique non-social conditions**:
   - Bad: `AND(drive_thru, 1HOP(...))` - drive_thru is unique (1 restaurant)
   - Good: `AND(budget, hipster, outdoor, 1HOP(...))` - narrows to several restaurants

### Pattern Selection Guidelines

**Good Patterns** (meaningful, general terms):
| Pattern | Why Good | Example |
|---------|----------|---------|
| `boba` | Domain-specific (bubble tea) | R85: 1HOP(['Ivy'], 'boba') |
| `taro` | Specific ingredient, naturally used | R89: 1HOP(['Grace'], 'taro') |
| `authentic` | Quality descriptor | R88: 1HOP(['Sam'], 'authentic') |
| `grit` | Food item (grit cakes) | R98: 2HOP(['Peter'], 'grit') |
| `writers` | Creative context | R99: 2HOP(['Rose'], 'writers') |

**Bad Patterns** (avoid these):
| Pattern | Why Bad | Problem |
|---------|---------|---------|
| Restaurant name | `hinge` → Hinge Cafe | Pattern IS the answer |
| Common words | `coffee` → Elixr Coffee | Matches restaurant name |
| Vague words | `consider`, `decent`, `totally` | No meaningful signal |
| Hyper-specific | `bison sausage` | Too narrow, not general language |

### Debugging Social Filter Issues

When a social filter request fails:

1. **Check friend_graph**: Verify anchor's friends in `user_mapping.json`
2. **Verify pattern**: Search reviews for the pattern by anchor's social circle
3. **Check hops**: 1-hop = anchor + direct friends; 2-hop = + friends-of-friends

```python
# Find who in anchor's 1-hop network mentions pattern at gold restaurant
friend_graph = user_mapping["friend_graph"]
anchor_friends = friend_graph.get("Grace", [])
one_hop_network = {"Grace"} | set(anchor_friends)

for review in gold_restaurant_reviews:
    if pattern in review["text"] and review["user"]["name"] in one_hop_network:
        print(f"Match: {review['user']['name']} mentions '{pattern}'")
```

## Natural Language for Negative Conditions

When using negative conditions, provide natural justification:

| Condition | Natural Phrase | Justification |
|-----------|----------------|---------------|
| `kids_no` | "adult-oriented" | quiet work, date night |
| `dogs_no` | "without dogs" | allergies, focus |
| `wifi_none` | "digital detox spot" | focused conversation |
| `takeout_no` | "dine-in experience" | full service, atmosphere |
| `outdoor_no` | "indoor-only" | weather, privacy |
| `no_tv` | "without TVs" | quiet, conversation |

## Debugging Multi-Match Issues

When a request matches multiple restaurants:

1. **Identify the matches**: Run validation to see which restaurants match
2. **Find differentiating conditions**: Check condition_matrix.json for conditions that:
   - Gold restaurant has, but others don't (add as positive condition)
   - Others have, but gold doesn't (add as negative condition with `"true": "False"`)
3. **Choose reliable conditions**: Prefer `item_meta` over `review_meta`
4. **Update and re-validate**

Example debugging session:
```python
# Find conditions unique to gold [29] vs matches [4, 40]
for cond_name, cond_data in conditions.items():
    has_29 = 29 in cond_data['satisfying_restaurants']
    has_4 = 4 in cond_data['satisfying_restaurants']
    has_40 = 40 in cond_data['satisfying_restaurants']
    if has_29 and not has_4 and not has_40:
        print(f"Add: {cond_name}")
```

## Validation

After generating or modifying requests:

```bash
.venv/bin/python -m data.validate philly_cafes
```

### Success Criteria
- All 100 requests show ✓
- No ⚠ multi-match warnings
- No ✗ no-match or gold_not_match errors

### Interpreting Errors

| Status | Meaning | Fix |
|--------|---------|-----|
| `✓` | Exactly one match = gold | No action needed |
| `⚠ multi (N)` | N restaurants match | Add disambiguating anchor |
| `✗ no_match` | Zero restaurants match | Relax conditions or fix evidence |
| `✗ gold_not_match` | Wrong restaurant matches | Fix anchor to target gold |

## Tools and Scripts

| Script | Purpose |
|--------|---------|
| `data/scripts/condition_matrix.py` | Generate condition satisfaction matrix |
| `data/scripts/generate_g05_g08_v2.py` | Generate G05-G08 requests with anchors |
| `data/validate.py` | Validate all requests match exactly one restaurant |

## Files Reference

| File | Description |
|------|-------------|
| `data/philly_cafes/condition_matrix.json` | Full condition satisfaction matrix |
| `data/philly_cafes/condition_summary.md` | Human-readable condition summary |
| `data/philly_cafes/requests.jsonl` | All 100 benchmark requests (G01-G10) |
| `data/philly_cafes/groundtruth.jsonl` | Validation results and gold answers |
| `data/philly_cafes/user_mapping.json` | Friend graph for G09/G10 social filters |
