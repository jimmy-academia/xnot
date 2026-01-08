# Request Structure Reference

This document defines the JSON schema for benchmark requests in `requests.jsonl`.

## Overview

Each request represents a user query with a formal logical structure that can be validated against restaurant data.

```json
{
  "id": "R01",
  "group": "G01",
  "scenario": "Busy Parent",
  "text": "Looking for a cafe that's kid-friendly, with a drive-thru, and without TVs",
  "shorthand": "AND(drive_thru, good_for_kids, no_tv)",
  "structure": { ... },
  "gold_restaurant": "e-ZyZc24wgkKafM3pguR2w"
}
```

---

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique request ID (e.g., "R01", "R100") |
| `group` | `str` | Complexity group (G01-G10) |
| `scenario` | `str` | Human-readable persona name |
| `text` | `str` | Natural language request |
| `shorthand` | `str` | Compact logical notation |
| `structure` | `object` | Formal logical structure (see below) |
| `gold_restaurant` | `str` | Business ID of correct answer |

### ID Format

- Format: `R{NN}` where NN is 01-100
- Groups: R01-R10 (G01), R11-R20 (G02), ..., R91-R100 (G10)

### Group Descriptions

| Group | Name | Pattern | Complexity |
|-------|------|---------|------------|
| G01 | Simple AND | `AND(a, b, c)` | Low |
| G02 | Simple OR | `AND(anchor, OR(a, b))` | Low-Medium |
| G03 | AND-OR Combination | `AND(a, OR(b, c))` | Medium |
| G04 | Review Metadata Weighting | `AND(a, review_meta_*)` | Medium-High |
| G05 | Triple OR | `OR(a, b, c)` | Medium |
| G06 | OR of ANDs | `OR(AND(a,b), AND(c,d))` | High |
| G07 | AND of ORs | `AND(unique, OR(a,b), OR(c,d))` | High |
| G08 | Complex Nested | `AND(a, b, c, OR(d, AND(e,f)))` | High |
| G09 | Direct Friends (1-hop) | `1HOP([friends], pattern)` | Medium |
| G10 | Social Circle (2-hop) | `2HOP([friends], pattern)` | High |

---

## Structure Schema

The `structure` field contains a recursive tree of conditions combined with AND/OR operators.

### Logical Node

```json
{
  "op": "AND" | "OR",
  "args": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `op` | `str` | Logical operator: `"AND"` or `"OR"` only |
| `args` | `array` | Child nodes (conditions or nested logical nodes) |

**Operator semantics:**
- `AND`: All children must be satisfied (False dominates → Unknown → True)
- `OR`: At least one child must be satisfied (True dominates → Unknown → False)

### Negation (No NOT Operator)

Negation is handled at the **evidence level**, not via a `NOT` operator. This keeps the structure simpler with only two operators.

| Negation Type | Evidence Field | Example |
|---------------|----------------|---------|
| Boolean false | `"true": "False"` | `{"path": ["attributes", "HasTV"], "true": "False"}` |
| Not contains | `"not_contains": "..."` | `{"path": ["attributes", "GoodForMeal"], "not_contains": "'breakfast': True"}` |
| Not equal | `"not_true": "..."` | `{"path": ["attributes", "WiFi"], "not_true": "free"}` |

See [Evidence Types](evidence_types.md) for full documentation on negation approaches.

### Condition Node

```json
{
  "aspect": "condition_name",
  "evidence": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `aspect` | `str` | Human-readable condition name |
| `evidence` | `object` | Evidence specification (see [Evidence Types](evidence_types.md)) |

---

## Structure Examples

### G01: Simple AND

Three conditions that must all be satisfied.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "drive_thru", "evidence": {"kind": "item_meta", "path": ["attributes", "DriveThru"], "true": "True"}},
    {"aspect": "good_for_kids", "evidence": {"kind": "item_meta", "path": ["attributes", "GoodForKids"], "true": "True"}},
    {"aspect": "no_tv", "evidence": {"kind": "item_meta", "path": ["attributes", "HasTV"], "true": "False"}}
  ]
}
```

**Shorthand**: `AND(drive_thru, good_for_kids, no_tv)`

### G02: Simple OR

Anchor condition with disjunction.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "full_bar", "evidence": {"kind": "item_meta", "path": ["attributes", "Alcohol"], "true": "u'full_bar'"}},
    {"op": "OR", "args": [
      {"aspect": "music_reviews", "evidence": {"kind": "review_text", "pattern": "music"}},
      {"aspect": "live_music", "evidence": {"kind": "item_meta", "path": ["attributes", "Music"], "contains": "'live': True"}}
    ]}
  ]
}
```

**Shorthand**: `AND(full_bar, OR(music_reviews, live_music))`

### G04: Credibility-Weighted Review

Review pattern with credibility weighting.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "drive_thru", "evidence": {"kind": "item_meta", "path": ["attributes", "DriveThru"], "true": "True"}},
    {"aspect": "hipster", "evidence": {"kind": "item_meta", "path": ["attributes", "Ambience"], "contains": "'hipster': True"}},
    {"aspect": "no_dogs", "evidence": {"kind": "item_meta", "path": ["attributes", "DogsAllowed"], "not_true": "True"}},
    {"aspect": "coffee_by_experts", "evidence": {
      "kind": "review_text",
      "pattern": "coffee",
      "weight_by": {"field": ["user", "review_count"]}
    }}
  ]
}
```

**Shorthand**: `AND(drive_thru, hipster, no_dogs, coffee_by_experts)`

### G05: Triple OR

At least one of three conditions must be satisfied.

```json
{
  "op": "OR",
  "args": [
    {"aspect": "wifi_free", "evidence": {"kind": "item_meta", "path": ["attributes", "WiFi"], "contains": "free"}},
    {"aspect": "drive_thru", "evidence": {"kind": "item_meta", "path": ["attributes", "DriveThru"], "true": "True"}},
    {"aspect": "outdoor_seating", "evidence": {"kind": "item_meta", "path": ["attributes", "OutdoorSeating"], "true": "True"}}
  ]
}
```

**Shorthand**: `OR(wifi_free, drive_thru, outdoor_seating)`

### G06: OR of ANDs

Disjunction of conjunctions (at least one AND branch must succeed).

```json
{
  "op": "OR",
  "args": [
    {"op": "AND", "args": [
      {"aspect": "cat_mediterranean", "evidence": {"kind": "item_meta", "path": ["categories"], "contains": "Mediterranean"}},
      {"aspect": "alcohol_full_bar", "evidence": {"kind": "item_meta", "path": ["attributes", "Alcohol"], "contains": "full_bar"}}
    ]},
    {"op": "AND", "args": [
      {"aspect": "goodformeal_latenight", "evidence": {"kind": "item_meta", "path": ["attributes", "GoodForMeal"], "contains": "'latenight': True"}},
      {"aspect": "ambience_romantic", "evidence": {"kind": "item_meta", "path": ["attributes", "Ambience"], "contains": "'romantic': True"}}
    ]}
  ]
}
```

**Shorthand**: `OR(AND(cat_mediterranean, alcohol_full_bar), AND(goodformeal_latenight, ambience_romantic))`

### G07: AND of ORs

Unique constraint plus two OR branches that must all succeed.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "cat_themed_cafes", "evidence": {"kind": "item_meta", "path": ["categories"], "contains": "Themed Cafes"}},
    {"op": "OR", "args": [
      {"aspect": "cat_arts", "evidence": {"kind": "item_meta", "path": ["categories"], "contains": "Arts & Entertainment"}},
      {"aspect": "goodformeal_latenight", "evidence": {"kind": "item_meta", "path": ["attributes", "GoodForMeal"], "contains": "'latenight': True"}}
    ]},
    {"op": "OR", "args": [
      {"op": "NOT", "args": [
        {"aspect": "goodformeal_breakfast", "evidence": {"kind": "item_meta", "path": ["attributes", "GoodForMeal"], "contains": "'breakfast': True"}}
      ]},
      {"aspect": "ambience_romantic", "evidence": {"kind": "item_meta", "path": ["attributes", "Ambience"], "contains": "'romantic': True"}}
    ]}
  ]
}
```

**Shorthand**: `AND(cat_themed_cafes, OR(cat_arts, goodformeal_latenight), OR(no_goodformeal_breakfast, ambience_romantic))`

### G08: Complex Nested

Multiple flat conditions plus an OR with nested AND.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "HasTV", "evidence": {"kind": "item_meta", "path": ["attributes", "HasTV"], "true": "True"}},
    {"aspect": "wifi_free", "evidence": {"kind": "item_meta", "path": ["attributes", "WiFi"], "contains": "free"}},
    {"aspect": "RestaurantsReservations", "evidence": {"kind": "item_meta", "path": ["attributes", "RestaurantsReservations"], "true": "True"}},
    {"op": "OR", "args": [
      {"aspect": "Alcohol", "evidence": {"kind": "item_meta", "path": ["attributes", "Alcohol"], "true": "True"}},
      {"op": "AND", "args": [
        {"aspect": "ambience_classy", "evidence": {"kind": "item_meta", "path": ["attributes", "Ambience"], "contains": "'classy': True"}},
        {"aspect": "OutdoorSeating", "evidence": {"kind": "item_meta", "path": ["attributes", "OutdoorSeating"], "true": "True"}}
      ]}
    ]}
  ]
}
```

**Shorthand**: `AND(HasTV, wifi_free, RestaurantsReservations, OR(Alcohol, AND(ambience_classy, OutdoorSeating)))`

### G09: Social Filter (1-hop)

Direct friend recommendation.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "social_cozy", "evidence": {
      "kind": "review_text",
      "pattern": "cozy",
      "social_filter": {"friends": ["Alice"], "hops": 1},
      "min_matches": 1
    }}
  ]
}
```

**Shorthand**: `1HOP(['Alice'], 'cozy')`

### G10: Social Filter (2-hop)

Extended social circle.

```json
{
  "op": "AND",
  "args": [
    {"aspect": "social_recommend", "evidence": {
      "kind": "review_text",
      "pattern": "recommend",
      "social_filter": {"friends": ["Bob"], "hops": 2},
      "min_matches": 1
    }}
  ]
}
```

**Shorthand**: `2HOP(['Bob'], 'recommend')`

---

## Shorthand Notation

The `shorthand` field provides a compact representation of the logical structure.

### Syntax

| Pattern | Meaning |
|---------|---------|
| `AND(a, b, c)` | All conditions must be satisfied |
| `OR(a, b)` | At least one condition must be satisfied |
| `AND(a, OR(b, c))` | a AND (b OR c) |
| `OR(AND(a,b), AND(c,d))` | (a AND b) OR (c AND d) |
| `1HOP(['Name'], 'pattern')` | Anchor + direct friends mention pattern |
| `2HOP(['Name'], 'pattern')` | Anchor + friends + friends-of-friends mention pattern |

**Social Filter Format**: Always use list syntax with quotes: `1HOP(['Alice'], 'boba')` not `1HOP(Alice, boba)`.

### Condition Names

Condition names in shorthand follow these conventions:

| Pattern | Evidence Type | Example |
|---------|--------------|---------|
| `attr_name` | item_meta (positive) | `drive_thru`, `wifi_free`, `budget` |
| `no_attr` | item_meta (negative) | `no_tv`, `no_dogs`, `no_outdoor` |
| `thing_reviews` | review_text | `coffee_reviews`, `cozy` |
| `review_meta_*` | review_meta | `review_meta_elite_status_love` |
| `hours_day_time` | item_meta_hours | `hours_monday_afternoon` |

**Negation Convention**: Use `no_` prefix for negative conditions. The structure uses `"true": "False"` or `"not_contains"` at the evidence level.

---

## Validation

Requests are validated using `data/validate.py`:

```bash
python -m data.validate philly_cafes
```

### Validation Status

| Status | Meaning |
|--------|---------|
| `ok` | Exactly one restaurant matches, and it's the gold answer |
| `no_match` | No restaurant satisfies all conditions |
| `multi_match` | Multiple restaurants satisfy conditions |
| `gold_not_match` | Gold restaurant doesn't satisfy conditions |

### Three-Value Logic

Each condition evaluates to:
- `1`: Satisfied
- `0`: Unknown (missing data)
- `-1`: Not satisfied

Operators reduce child values:
- **AND**: False dominates → Unknown → True
- **OR**: True dominates → Unknown → False

---

## Creating New Requests

### Guidelines

1. **Start with anchor**: Identify unique or rare conditions first
2. **Add complexity**: Layer OR conditions for evaluation challenge
3. **Validate uniqueness**: Ensure exactly one restaurant matches
4. **Write natural text**: Request should read naturally
5. **Update shorthand**: Keep notation accurate

### Template

```json
{
  "id": "R{NN}",
  "group": "G{XX}",
  "scenario": "Persona Name",
  "text": "Looking for a cafe that...",
  "shorthand": "AND(...)",
  "structure": {
    "op": "AND",
    "args": [
      {"aspect": "condition_name", "evidence": {...}},
      ...
    ]
  },
  "gold_restaurant": "business_id_here"
}
```

### Checklist

- [ ] ID follows format R01-R100
- [ ] Group matches request complexity
- [ ] Scenario is descriptive persona name
- [ ] Text is natural language
- [ ] Shorthand matches structure
- [ ] Structure uses valid evidence types
- [ ] Gold restaurant satisfies all conditions
- [ ] No other restaurant satisfies all conditions

---

## Related Files

| File | Purpose |
|------|---------|
| `data/philly_cafes/requests.jsonl` | 100 benchmark requests |
| `data/philly_cafes/groundtruth.jsonl` | Validation results |
| `data/philly_cafes/condition_matrix.json` | Condition satisfaction matrix |
| `data/validate.py` | Validation implementation |

---

## Implementation Reference

Source code: `data/validate.py`

| Function | Purpose |
|----------|---------|
| `validate_request()` | Check single request against all restaurants |
| `evaluate_structure()` | Recursive AND/OR evaluation |
| `evaluate_condition()` | Single condition evaluation |
| `reduce_tv()` | Three-value logic reduction |
