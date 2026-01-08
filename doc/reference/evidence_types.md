# Evidence Types Reference

This document defines all evidence types supported by the validation system (`data/validate.py`). Evidence types determine how conditions are evaluated against restaurant data.

## Overview

| Evidence Type | Purpose | Data Source | Usage |
|---------------|---------|-------------|-------|
| `item_meta` | Match restaurant attributes | Restaurant JSON fields | ~47% |
| `item_meta_hours` | Match operating hours | `hours` field | ~3% |
| `review_text` | Pattern match in reviews | Review text | ~33% |
| `review_meta` | Check reviewer metadata | Review/user fields | ~12% |
| `social_filter` | Social graph filtering | `user_mapping.json` | ~5% |

---

## 1. item_meta

Evaluates conditions against restaurant metadata fields.

### Schema

```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["field", "nested_field"],
    "true": "value_for_match",
    "false": "value_for_no_match",
    "not_true": "value_that_should_not_match",
    "neutral": "value_for_unknown",
    "contains": "substring_to_find",
    "missing": 0
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[str]` | Path to value in restaurant JSON (e.g., `["attributes", "WiFi"]`) |
| `true` | `any` | Value that results in +1 (satisfied). Use `"False"` for negative boolean checks. |
| `false` | `any` | Value that results in -1 (not satisfied) |
| `not_true` | `any` | Value that should NOT match (None passes, match fails) |
| `not_contains` | `str` | Substring that should NOT be present (None passes) |
| `neutral` | `any` | Value that results in 0 (unknown) |
| `contains` | `str` | Substring to find in string representation |
| `missing` | `int` | Return value when path doesn't exist (default: 0) |

### Negation Approaches

There are two ways to express negative conditions:

| Approach | Use Case | Example |
|----------|----------|---------|
| `"true": "False"` | Boolean attribute should be False | `{"path": ["attributes", "HasTV"], "true": "False"}` |
| `"not_contains"` | Substring should NOT be present | `{"path": ["attributes", "GoodForMeal"], "not_contains": "'breakfast': True"}` |
| `"not_true"` | Value should NOT equal target | `{"path": ["attributes", "WiFi"], "not_true": "free"}` |

**Note**: The structure uses only `AND` and `OR` operators. Negation is handled at the evidence level, not via a `NOT` operator.

### Evaluation Logic

1. **not_contains check** (if specified): Returns +1 if value is None or doesn't contain substring, -1 if contains
2. **not_true/true_not check** (if specified): Returns +1 if value is None or doesn't match, -1 if matches
3. **contains check** (if specified): Substring search in string representation
4. **Missing value**: Returns `missing` field value (default 0)
5. **Dict of booleans**: OR across children (any True value → +1)
6. **No conditions specified**: Boolean check (True→+1, False→-1, else→0)
7. **Only false specified**: Negative logic (match→-1, no match→+1)
8. **Only true specified**: Positive logic (match→+1, no match→-1)

### Examples

**Boolean attribute (WiFi exists)**:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "WiFi"],
    "true": "free"
  }
}
```

**Negative boolean check (no TV)**:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "HasTV"],
    "true": "False"
  }
}
```

**Negative contains check (not good for breakfast)**:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "GoodForMeal"],
    "not_contains": "'breakfast': True"
  }
}
```

**Nested attribute (quiet ambience)**:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "Ambience"],
    "contains": "'quiet': True"
  }
}
```

**Price level**:
```json
{
  "evidence": {
    "kind": "item_meta",
    "path": ["attributes", "RestaurantsPriceRange2"],
    "true": "1"
  }
}
```

### Yelp Attribute Parsing

Yelp stores attributes as strings that need parsing:
- `"True"` → `True`
- `"False"` → `False`
- `"u'free'"` → `"free"`
- `"{'romantic': False, 'casual': True}"` → `{"romantic": False, "casual": True}`

The validation system handles this automatically via `parse_attr_value()`.

---

## 2. item_meta_hours

Evaluates operating hours conditions.

### Schema

```json
{
  "evidence": {
    "kind": "item_meta_hours",
    "path": ["hours", "Monday"],
    "true": "12:00-17:00",
    "missing": 0
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[str]` | Path to day's hours (e.g., `["hours", "Monday"]`) |
| `true` | `str` | Required time range in "H:M-H:M" format |
| `missing` | `int` | Return value when hours not available (default: 0) |

### Evaluation Logic

Checks if restaurant hours **contain** the required time range:
- Restaurant hours: `"8:00-22:00"` (8 AM to 10 PM)
- Required: `"12:00-17:00"` (noon to 5 PM)
- Result: +1 (restaurant is open during required window)

Handles overnight hours (end < start means crosses midnight).

### Examples

**Open Monday afternoon**:
```json
{
  "evidence": {
    "kind": "item_meta_hours",
    "path": ["hours", "Monday"],
    "true": "12:00-17:00"
  }
}
```

**Open Saturday morning**:
```json
{
  "evidence": {
    "kind": "item_meta_hours",
    "path": ["hours", "Saturday"],
    "true": "7:00-12:00"
  }
}
```

---

## 3. review_text

Pattern matching in review text.

### Schema (Basic)

```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "regex_pattern"
  }
}
```

### Schema (With Credibility Weighting)

```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "regex_pattern",
    "weight_by": {
      "field": ["user", "review_count"]
    },
    "credibility_percentile": 50,
    "min_credible_matches": 2
  }
}
```

### Schema (With Social Filter)

```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "pattern",
    "social_filter": {
      "friends": ["Alice", "Bob"],
      "hops": 1,
      "min_matches": 1
    }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | `str` | Regex pattern to match (case-insensitive) |
| `weight_by` | `object` | Enable credibility-count evaluation |
| `weight_by.field` | `list[str]` | Path to credibility field (e.g., `["user", "review_count"]`) |
| `credibility_percentile` | `int` | Percentile threshold for "credible" (default: 50) |
| `min_credible_matches` | `int` | Minimum credible reviewers required (default: 2) |
| `social_filter` | `object` | Enable social graph filtering |
| `social_filter.friends` | `list[str]` | Names of friends to filter by |
| `social_filter.hops` | `int` | 1 for direct friends, 2 for friends-of-friends |
| `social_filter.min_matches` | `int` | Minimum qualifying reviews (default: 1) |

### Evaluation Logic

**Basic mode**: Returns +1 if any review contains pattern, -1 otherwise.

**Credibility-count mode** (when `weight_by` specified):
1. Extract credibility values from all reviews
2. Compute threshold at specified percentile of non-zero values
3. Count reviews where: value >= threshold AND pattern matches
4. Return +1 if count >= `min_credible_matches`, else -1

**Social filter mode** (when `social_filter` specified):
1. Uses synthetic data from `user_mapping.json`
2. Filter reviews to those from users in social circle
3. Check if pattern is mentioned by qualifying reviewers

### Examples

**Simple pattern match**:
```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "cozy|comfortable"
  }
}
```

**Elite reviewer credibility**:
```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "\\bwork\\b",
    "weight_by": {"field": ["user", "elite"]},
    "credibility_percentile": 50,
    "min_credible_matches": 2
  }
}
```

**Direct friend recommendation**:
```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "cozy",
    "social_filter": {
      "friends": ["Alice"],
      "hops": 1,
      "min_matches": 1
    }
  }
}
```

### Credibility Fields

Common fields for `weight_by`:

| Field Path | Description |
|------------|-------------|
| `["user", "review_count"]` | Total reviews by user |
| `["user", "fans"]` | Number of followers |
| `["user", "elite"]` | List of elite years (counted) |
| `["useful"]` | Useful votes on review |
| `["funny"]` | Funny votes on review |
| `["cool"]` | Cool votes on review |

---

## 4. review_meta

Evaluates review metadata without pattern matching.

### Schema

```json
{
  "evidence": {
    "kind": "review_meta",
    "path": ["user", "elite"],
    "op": "not_empty",
    "value": null,
    "min_stars": null,
    "agg": "any",
    "count": 1,
    "missing": 0
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[str]` | Path to metadata field |
| `op` | `str` | Operator: `"not_empty"`, `"gte"`, `"lte"` |
| `value` | `number` | Threshold for `gte`/`lte` operators |
| `min_stars` | `number` | Only consider reviews with stars >= this |
| `agg` | `str` | Aggregation: `"any"`, `"all"`, `"count"` |
| `count` | `int` | For `agg="count"`, required matches |
| `missing` | `int` | Return when no reviews (default: 0) |

### Operators

| Operator | Description |
|----------|-------------|
| `not_empty` | Value exists and is non-empty (for lists: has valid entries) |
| `gte` | Value >= threshold |
| `lte` | Value <= threshold |

### Aggregation Modes

| Mode | Logic |
|------|-------|
| `any` | At least one review matches → +1 |
| `all` | All reviews match → +1 |
| `count` | At least N reviews match → +1 |

### Examples

**Has elite reviewer**:
```json
{
  "evidence": {
    "kind": "review_meta",
    "path": ["user", "elite"],
    "op": "not_empty",
    "agg": "any"
  }
}
```

**Multiple high-value reviews**:
```json
{
  "evidence": {
    "kind": "review_meta",
    "path": ["useful"],
    "op": "gte",
    "value": 10,
    "agg": "count",
    "count": 3
  }
}
```

**High-star reviews only**:
```json
{
  "evidence": {
    "kind": "review_meta",
    "path": ["user", "fans"],
    "op": "gte",
    "value": 50,
    "min_stars": 4,
    "agg": "any"
  }
}
```

---

## 5. social_filter

Social graph filtering for review qualification. Used within `review_text` evidence type.

### Supported via review_text

Social filtering is specified as a sub-field of `review_text`:

```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "recommend",
    "social_filter": {
      "friends": ["Bob", "Carol"],
      "hops": 2,
      "min_matches": 1
    }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `friends` | `list[str]` | User names to filter by |
| `hops` | `int` | Social distance: 1 (direct) or 2 (extended) |
| `min_matches` | `int` | Minimum qualifying reviews (default: 1) |

### Hop Logic

The `friends` list contains **anchor names** - the starting point(s) for social graph traversal. The `hops` parameter determines how far to traverse.

**1-hop** (anchor + direct friends):
- Reviewer IS one of the anchors (0-hop), OR
- Reviewer is a friend of any anchor (anchor appears in reviewer's friend list)
- Example: `friends=["Grace"]` → Grace's reviews qualify, plus reviews from Grace's friends (Emma, Ivy)

**2-hop** (anchor + friends + friends-of-friends):
- All 1-hop reviewers qualify, PLUS
- Reviewer is a friend-of-friend of any anchor
- Example: `friends=["Grace"]` → Grace, Emma, Ivy, plus Carol (Emma's friend) and Kate (Ivy's friend)

### Data Source

Social data is stored in `data/philly_cafes/user_mapping.json`:

```json
{
  "friend_graph": {
    "Grace": ["Emma", "Ivy"],
    "Emma": ["Carol", "Grace"],
    "Ivy": ["Grace", "Kate"],
    ...
  }
}
```

### Examples

**Direct friend recommendation (G09)** - 1HOP(['Grace'], 'taro'):
```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "taro",
    "social_filter": {
      "friends": ["Grace"],
      "hops": 1,
      "min_matches": 1
    }
  }
}
```
This matches if Grace OR any of Grace's friends (Emma, Ivy) mention "taro" in their review.

**Extended social circle (G10)** - 2HOP(['Alice'], 'vanilla bean'):
```json
{
  "evidence": {
    "kind": "review_text",
    "pattern": "vanilla bean",
    "social_filter": {
      "friends": ["Alice"],
      "hops": 2,
      "min_matches": 1
    }
  }
}
```
This matches if Alice, Alice's friends, OR friends-of-friends mention "vanilla bean".

---

## Three-Value Logic

All evidence types return one of three values:

| Value | Meaning | Description |
|-------|---------|-------------|
| `1` | Satisfied | Condition is met |
| `0` | Unknown | Cannot determine (missing data) |
| `-1` | Not satisfied | Condition is not met |

### Logical Operators

Conditions are combined using AND/OR:

```json
{
  "op": "AND",
  "args": [
    {"evidence": {...}},
    {"op": "OR", "args": [...]}
  ]
}
```

**AND reduction**: False dominates → Unknown → True
**OR reduction**: True dominates → Unknown → False

---

## Adding New Evidence Types

To add a new evidence type:

1. Add evaluation function in `data/validate.py`:
   ```python
   def evaluate_my_type(item, evidence_spec, reviews=None) -> int:
       # Return 1, 0, or -1
   ```

2. Register in `evaluate_condition()`:
   ```python
   elif kind == "my_type":
       return evaluate_my_type(item, evidence_spec, reviews)
   ```

3. Document in this file with schema, fields, and examples.

4. Add test cases in requests.jsonl.

See [Adding Evidence Types Guide](../guides/add_evidence_type.md) for detailed instructions.

---

## Implementation Reference

Source code: [data/validate.py](../../data/validate.py)

| Function | Lines | Purpose |
|----------|-------|---------|
| `evaluate_item_meta_rule()` | 386-454 | item_meta evaluation |
| `hours_contains()` | 96-107 | item_meta_hours evaluation |
| `evaluate_review_text_pattern()` | 457-479 | Basic review_text |
| `evaluate_credibility_count()` | 136-185 | Credibility-weighted review_text |
| `evaluate_review_meta()` | 188-256 | review_meta evaluation |
| `evaluate_social_filter()` | 482-520 | social_filter evaluation |
| `check_social_filter()` | 45-77 | Social graph traversal |
| `evaluate_condition()` | 523-587 | Main dispatcher |

---

## Related Documentation

- [Design Rationale](../paper/design_rationale.md) - Evidence type design decisions
- [Request Structure](request_structure.md) - Request JSON schema
- [Adding Evidence Types](../guides/add_evidence_type.md) - How to extend
