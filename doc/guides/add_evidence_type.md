# Adding New Evidence Types

Guide to extending the validation system with custom evidence types for specialized evaluation logic.

## Overview

Evidence types define how conditions are evaluated against item data. The framework includes 5 built-in types:
- `item_meta` - Attribute matching
- `item_meta_hours` - Operating hours
- `review_text` - Pattern matching in text
- `review_meta` - Review metadata checks
- `social_filter` - Social graph filtering (via review_text)

This guide shows how to add custom evidence types for domain-specific needs.

---

## When to Add a New Evidence Type

Add a custom evidence type when:
- Your domain has unique data structures
- Existing types don't support your evaluation logic
- You need specialized aggregation or scoring

**Examples**:
- Geographic distance evaluation
- Price range comparisons
- Inventory availability checks
- Custom scoring algorithms

---

## Implementation Steps

### Step 1: Define the Schema

Design the JSON schema for your evidence type:

```json
{
  "evidence": {
    "kind": "my_custom_type",
    "path": ["field", "nested_field"],
    "param1": "value1",
    "param2": 42,
    "missing": 0
  }
}
```

**Required fields**:
- `kind`: Unique identifier for your type

**Optional fields**:
- `path`: Navigation path into item data
- `missing`: Return value when data is missing (default: 0)
- Custom parameters for your logic

### Step 2: Write Evaluation Function

Add function to `data/validate.py`:

```python
def evaluate_my_custom_type(item: dict, evidence_spec: dict, reviews: list = None) -> int:
    """Evaluate my custom condition.

    Args:
        item: Item dict (restaurant, product, etc.)
        evidence_spec: Evidence specification from request structure
        reviews: Optional list of review dicts

    Returns:
        1 (satisfied), 0 (unknown), -1 (not satisfied)
    """
    # Extract parameters
    path = evidence_spec.get("path", [])
    param1 = evidence_spec.get("param1")
    param2 = evidence_spec.get("param2", 0)
    missing_val = evidence_spec.get("missing", 0)

    # Navigate to value
    value = get_nested_value(item, path)
    if value is None:
        return missing_val

    # Your evaluation logic here
    if meets_condition(value, param1, param2):
        return 1
    else:
        return -1
```

### Step 3: Register in Dispatcher

Add case to `evaluate_condition()` in `data/validate.py`:

```python
def evaluate_condition(item: dict, condition: dict, reviews: list = None) -> int:
    """Evaluate a single condition against an item."""
    evidence_spec = condition.get("evidence", {})
    kind = evidence_spec.get("kind", "item_meta")

    if kind == "item_meta":
        # ... existing code ...

    elif kind == "my_custom_type":
        return evaluate_my_custom_type(item, evidence_spec, reviews)

    return 0
```

### Step 4: Add Test Cases

Create requests using your new type:

```json
{
  "id": "R_TEST",
  "group": "G_CUSTOM",
  "scenario": "Test Scenario",
  "text": "Testing my custom evidence type",
  "shorthand": "CUSTOM(param1, param2)",
  "structure": {
    "op": "AND",
    "args": [
      {
        "aspect": "custom_condition",
        "evidence": {
          "kind": "my_custom_type",
          "path": ["data", "field"],
          "param1": "value",
          "param2": 42
        }
      }
    ]
  },
  "gold_item": "item_id"
}
```

### Step 5: Validate

```bash
python -m data.validate my_benchmark
```

### Step 6: Document

Add to `doc/reference/evidence_types.md`:

```markdown
## N. my_custom_type

Description of what this evidence type evaluates.

### Schema

\`\`\`json
{
  "evidence": {
    "kind": "my_custom_type",
    "path": ["field", "nested"],
    "param1": "value",
    "param2": 42,
    "missing": 0
  }
}
\`\`\`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `list[str]` | Path to value in item |
| `param1` | `str` | First parameter |
| `param2` | `int` | Second parameter |
| `missing` | `int` | Return when value missing |

### Evaluation Logic

1. Navigate to value using path
2. Apply custom logic with parameters
3. Return 1 if satisfied, -1 otherwise

### Examples

**Basic usage**:
\`\`\`json
{
  "evidence": {
    "kind": "my_custom_type",
    "path": ["attributes", "custom"],
    "param1": "expected_value"
  }
}
\`\`\`
```

---

## Example: Geographic Distance

A complete example implementing geographic distance evaluation.

### Schema

```json
{
  "evidence": {
    "kind": "geo_distance",
    "path": ["coordinates"],
    "reference": {"lat": 39.9526, "lng": -75.1652},
    "max_distance_km": 5.0,
    "missing": 0
  }
}
```

### Implementation

```python
import math

def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km

    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def evaluate_geo_distance(item: dict, evidence_spec: dict) -> int:
    """Evaluate geographic distance condition.

    Checks if item is within max_distance_km of reference point.

    Returns: 1 if within distance, -1 if outside, 0 if missing coordinates
    """
    path = evidence_spec.get("path", ["coordinates"])
    reference = evidence_spec.get("reference", {})
    max_distance = evidence_spec.get("max_distance_km", 10.0)
    missing_val = evidence_spec.get("missing", 0)

    # Get item coordinates
    coords = get_nested_value(item, path)
    if not coords or not isinstance(coords, dict):
        return missing_val

    item_lat = coords.get("lat") or coords.get("latitude")
    item_lng = coords.get("lng") or coords.get("longitude")

    if item_lat is None or item_lng is None:
        return missing_val

    # Get reference coordinates
    ref_lat = reference.get("lat")
    ref_lng = reference.get("lng")

    if ref_lat is None or ref_lng is None:
        return 0

    # Calculate distance
    distance = haversine_distance(
        float(item_lat), float(item_lng),
        float(ref_lat), float(ref_lng)
    )

    return 1 if distance <= max_distance else -1
```

### Registration

```python
# In evaluate_condition()
elif kind == "geo_distance":
    return evaluate_geo_distance(item, evidence_spec)
```

### Usage

```json
{
  "aspect": "near_city_center",
  "evidence": {
    "kind": "geo_distance",
    "path": ["location"],
    "reference": {"lat": 39.9526, "lng": -75.1652},
    "max_distance_km": 2.0
  }
}
```

---

## Example: Price Range Comparison

### Schema

```json
{
  "evidence": {
    "kind": "price_range",
    "path": ["price"],
    "min": 10.00,
    "max": 50.00,
    "currency": "USD"
  }
}
```

### Implementation

```python
def evaluate_price_range(item: dict, evidence_spec: dict) -> int:
    """Evaluate price range condition.

    Checks if item price falls within [min, max] range.
    """
    path = evidence_spec.get("path", ["price"])
    min_price = evidence_spec.get("min", 0)
    max_price = evidence_spec.get("max", float('inf'))
    missing_val = evidence_spec.get("missing", 0)

    value = get_nested_value(item, path)
    if value is None:
        return missing_val

    try:
        price = float(value)
    except (ValueError, TypeError):
        return missing_val

    if min_price <= price <= max_price:
        return 1
    return -1
```

---

## Example: Inventory Check

### Schema

```json
{
  "evidence": {
    "kind": "inventory",
    "path": ["stock"],
    "min_quantity": 1,
    "check_availability": true
  }
}
```

### Implementation

```python
def evaluate_inventory(item: dict, evidence_spec: dict) -> int:
    """Evaluate inventory availability condition."""
    path = evidence_spec.get("path", ["stock"])
    min_qty = evidence_spec.get("min_quantity", 1)
    check_avail = evidence_spec.get("check_availability", True)
    missing_val = evidence_spec.get("missing", 0)

    value = get_nested_value(item, path)

    if value is None:
        return missing_val

    # Handle different stock representations
    if isinstance(value, bool):
        return 1 if value else -1

    if isinstance(value, dict):
        if check_avail and "available" in value:
            return 1 if value["available"] else -1
        if "quantity" in value:
            return 1 if value["quantity"] >= min_qty else -1

    try:
        qty = int(value)
        return 1 if qty >= min_qty else -1
    except (ValueError, TypeError):
        return missing_val
```

---

## Best Practices

### Return Values

Always return one of three values:
- `1`: Condition satisfied
- `0`: Unknown (missing data, can't determine)
- `-1`: Condition not satisfied

### Handle Missing Data

Always check for `None` and use `missing` parameter:

```python
value = get_nested_value(item, path)
if value is None:
    return evidence_spec.get("missing", 0)
```

### Type Safety

Handle various input types gracefully:

```python
try:
    numeric_value = float(value)
except (ValueError, TypeError):
    return missing_val
```

### Reuse Utilities

Use existing helper functions:
- `get_nested_value()` - Navigate nested dicts
- `parse_attr_value()` - Parse Yelp attribute strings
- `match_value()` - Flexible value comparison

### Document Thoroughly

For each new type, document:
- Purpose and use cases
- Complete schema with all fields
- Evaluation logic step-by-step
- Multiple examples

---

## Testing Checklist

- [ ] Function handles missing data correctly
- [ ] Function handles invalid data types
- [ ] Returns 1/0/-1 only
- [ ] Registered in evaluate_condition()
- [ ] Test requests pass validation
- [ ] Documentation added to evidence_types.md
- [ ] Examples cover common use cases

---

## Related Documentation

- [Evidence Types Reference](../reference/evidence_types.md)
- [Request Structure Reference](../reference/request_structure.md)
- [Creating New Benchmarks](create_new_benchmark.md)
