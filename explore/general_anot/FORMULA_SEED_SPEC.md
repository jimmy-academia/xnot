# Formula Seed Specification

## Overview

The Formula Seed is the output of Phase 1 - a complete, executable specification that Phase 2 interprets to evaluate a restaurant against the task query.

**Design Principles:**
1. **Declarative, not code** - Operations are declared, not Python strings
2. **Preserve semantics** - All meaning from the query is retained
3. **Unambiguous** - Phase 2 can execute without guessing
4. **Namespace clarity** - Clear distinction between extraction fields, metadata, and context

---

## Structure

```json
{
  "task_name": "G1a-v2",
  "filter": { ... },
  "extract": { ... },
  "compute": [ ... ],
  "output": [ ... ]
}
```

---

## 1. Filter Section

Defines how to find relevant reviews before extraction.

```json
{
  "filter": {
    "keywords": ["allergy", "allergic", "peanut", "nut", "anaphylaxis", "epipen"]
  }
}
```

---

## 2. Extract Section

Defines what to extract from each relevant review. Each field includes full semantic descriptions so the extraction LLM knows what each value means.

```json
{
  "extract": {
    "fields": [
      {
        "name": "incident_severity",
        "type": "enum",
        "values": {
          "none": "No allergic reaction described",
          "mild": "Minor symptoms (stomach upset, mild discomfort)",
          "moderate": "Visible symptoms (hives, swelling, needed medication)",
          "severe": "Life-threatening (anaphylaxis, EpiPen, ER visit)"
        }
      },
      {
        "name": "account_type",
        "type": "enum",
        "values": {
          "none": "No incident mentioned",
          "firsthand": "Personal experience ('I had', 'my child experienced')",
          "secondhand": "Reported by someone else ('I heard', 'friend told me')",
          "hypothetical": "Concern without actual incident"
        }
      },
      {
        "name": "safety_interaction",
        "type": "enum",
        "values": {
          "none": "No staff interaction about allergies",
          "positive": "Staff asked about allergies AND successfully accommodated",
          "negative": "Staff dismissive or refused to accommodate",
          "betrayal": "Staff CLAIMED safe BUT customer still had reaction"
        }
      }
    ]
  }
}
```

---

## 3. Compute Section

Ordered list of computation steps. Each step produces a named value that later steps can reference.

### Namespaces

| Prefix | Source | Examples |
|--------|--------|----------|
| `extraction.*` | LLM extraction output | `extraction.incident_severity`, `extraction.account_type` |
| `meta.*` | Review metadata | `meta.stars`, `meta.useful`, `meta.year` |
| `context.*` | Restaurant context | `context.categories`, `context.name` |
| `$NAME` | Defined filter reference | `$IS_INCIDENT` |
| (none) | Previously computed value | `N_MILD`, `TRUST_SCORE` |

### Operation Types

#### `define_filter` - Create reusable filter
```json
{
  "name": "IS_INCIDENT",
  "op": "define_filter",
  "extraction": {
    "account_type": "firsthand",
    "incident_severity": {"in": ["mild", "moderate", "severe"]}
  }
}
```

#### `count` - Count matching extractions
```json
{
  "name": "N_MILD",
  "op": "count",
  "where": {
    "extraction.incident_severity": "mild",
    "extraction.account_type": "firsthand"
  }
}
```

With metadata filter:
```json
{
  "name": "N_RECENT",
  "op": "count",
  "where": {
    "$IS_INCIDENT": true,
    "meta.year": {">=": 2023}
  }
}
```

#### `sum` - Sum expression over matching extractions
```json
{
  "name": "TOTAL_WEIGHT",
  "op": "sum",
  "expr": "(5 - meta.stars) + log(meta.useful + 1)",
  "where": {"$IS_INCIDENT": true}
}
```

#### `max` / `min` - Aggregate over matching extractions
```json
{
  "name": "MOST_RECENT_YEAR",
  "op": "max",
  "field": "meta.year",
  "where": {"$IS_INCIDENT": true},
  "default": 2020
}
```

#### `lookup` - Match source against table
```json
{
  "name": "CUISINE_MODIFIER",
  "op": "lookup",
  "source": "context.categories",
  "table": {
    "Thai": 2.0,
    "Vietnamese": 1.8,
    "Chinese": 1.5,
    "Italian": 0.5,
    "Pizza": 0.5
  },
  "match": "substring_max",
  "default": 1.0
}
```

Match modes:
- `"exact"` - Direct key lookup
- `"substring_first"` - First key found as substring
- `"substring_max"` - Highest value among all substring matches

#### `expr` - Arithmetic on computed values
```json
{
  "name": "TRUST_SCORE",
  "op": "expr",
  "expr": "max(0.1, min(1.0, TRUST_RAW))"
}
```

Available functions: `max`, `min`, `abs`, `log`, `sqrt`

#### `const` - Literal value
```json
{
  "name": "BASE_RISK",
  "op": "const",
  "value": 2.0
}
```

#### `case` - Conditional rules
```json
{
  "name": "TRAJECTORY_MULTIPLIER",
  "op": "case",
  "rules": [
    {"when": "RECENT_RATIO > 0.7", "then": 1.3},
    {"when": "RECENT_RATIO < 0.3 and N_TOTAL_INCIDENTS > 0", "then": 0.7},
    {"else": 1.0}
  ]
}
```

With source field:
```json
{
  "name": "VERDICT",
  "op": "case",
  "source": "FINAL_RISK_SCORE",
  "rules": [
    {"when": "< 4.0", "then": "Low Risk"},
    {"when": "< 8.0", "then": "High Risk"},
    {"else": "Critical Risk"}
  ]
}
```

### Filter Operators

| Operator | Example | Meaning |
|----------|---------|---------|
| (none) | `"field": "value"` | Exact match |
| `in` | `"field": {"in": ["a", "b"]}` | Value in list |
| `>=` | `"field": {">=": 2023}` | Greater or equal |
| `>` | `"field": {">": 0}` | Greater than |
| `<=` | `"field": {"<=": 10}` | Less or equal |
| `<` | `"field": {"<": 2023}` | Less than |
| `!=` | `"field": {"!=": "none"}` | Not equal |

---

## 4. Output Section

List of computed values to return.

```json
{
  "output": [
    "N_TOTAL_INCIDENTS",
    "TRUST_SCORE",
    "ADJUSTED_INCIDENT_SCORE",
    "TRAJECTORY_MULTIPLIER",
    "CUISINE_IMPACT",
    "FINAL_RISK_SCORE",
    "VERDICT"
  ]
}
```

---

## Complete Example (G1a-v2)

```json
{
  "task_name": "G1a-v2",

  "filter": {
    "keywords": ["allergy", "allergic", "peanut", "nut", "anaphylaxis", "epipen"]
  },

  "extract": {
    "fields": [
      {
        "name": "incident_severity",
        "type": "enum",
        "values": {
          "none": "No allergic reaction described",
          "mild": "Minor symptoms (stomach upset, mild discomfort)",
          "moderate": "Visible symptoms (hives, swelling, needed medication)",
          "severe": "Life-threatening (anaphylaxis, EpiPen, ER visit)"
        }
      },
      {
        "name": "account_type",
        "type": "enum",
        "values": {
          "none": "No incident mentioned",
          "firsthand": "Personal experience ('I had', 'my child experienced')",
          "secondhand": "Reported by someone else ('I heard', 'friend told me')",
          "hypothetical": "Concern without actual incident"
        }
      },
      {
        "name": "safety_interaction",
        "type": "enum",
        "values": {
          "none": "No staff interaction about allergies",
          "positive": "Staff asked about allergies AND successfully accommodated",
          "negative": "Staff dismissive or refused to accommodate",
          "betrayal": "Staff CLAIMED safe BUT customer still had reaction"
        }
      }
    ]
  },

  "compute": [
    {"name": "IS_INCIDENT", "op": "define_filter",
     "extraction": {"account_type": "firsthand", "incident_severity": {"in": ["mild", "moderate", "severe"]}}},

    {"name": "N_MILD", "op": "count", "where": {"extraction.incident_severity": "mild", "extraction.account_type": "firsthand"}},
    {"name": "N_MODERATE", "op": "count", "where": {"extraction.incident_severity": "moderate", "extraction.account_type": "firsthand"}},
    {"name": "N_SEVERE", "op": "count", "where": {"extraction.incident_severity": "severe", "extraction.account_type": "firsthand"}},
    {"name": "N_TOTAL_INCIDENTS", "op": "expr", "expr": "N_MILD + N_MODERATE + N_SEVERE"},

    {"name": "N_POSITIVE", "op": "count", "where": {"extraction.safety_interaction": "positive"}},
    {"name": "N_NEGATIVE", "op": "count", "where": {"extraction.safety_interaction": "negative"}},
    {"name": "N_BETRAYAL", "op": "count", "where": {"extraction.safety_interaction": "betrayal"}},
    {"name": "N_ALLERGY_REVIEWS", "op": "count"},

    {"name": "TRUST_RAW", "op": "expr", "expr": "1.0 + (N_POSITIVE * 0.1) - (N_NEGATIVE * 0.2) - (N_BETRAYAL * 0.5)"},
    {"name": "TRUST_SCORE", "op": "expr", "expr": "max(0.1, min(1.0, TRUST_RAW))"},

    {"name": "MILD_WEIGHT", "op": "expr", "expr": "2 * (1.5 - TRUST_SCORE)"},
    {"name": "MODERATE_WEIGHT", "op": "expr", "expr": "5 * (1.3 - 0.3 * TRUST_SCORE)"},
    {"name": "SEVERE_WEIGHT", "op": "const", "value": 15},
    {"name": "ADJUSTED_INCIDENT_SCORE", "op": "expr", "expr": "(N_MILD * MILD_WEIGHT) + (N_MODERATE * MODERATE_WEIGHT) + (N_SEVERE * SEVERE_WEIGHT)"},

    {"name": "N_RECENT", "op": "count", "where": {"$IS_INCIDENT": true, "meta.year": {">=": 2023}}},
    {"name": "N_OLD", "op": "count", "where": {"$IS_INCIDENT": true, "meta.year": {"<": 2023}}},
    {"name": "RECENT_RATIO", "op": "expr", "expr": "N_RECENT / N_TOTAL_INCIDENTS if N_TOTAL_INCIDENTS > 0 else 0"},
    {"name": "TRAJECTORY_MULTIPLIER", "op": "case", "rules": [
      {"when": "RECENT_RATIO > 0.7", "then": 1.3},
      {"when": "RECENT_RATIO < 0.3 and N_TOTAL_INCIDENTS > 0", "then": 0.7},
      {"else": 1.0}
    ]},

    {"name": "CUISINE_MODIFIER", "op": "lookup", "source": "context.categories",
     "table": {"Thai": 2.0, "Vietnamese": 1.8, "Chinese": 1.5, "Asian Fusion": 1.5,
               "Indian": 1.3, "Japanese": 1.2, "Korean": 1.2, "Mexican": 1.0,
               "Italian": 0.5, "American": 0.5, "Pizza": 0.5},
     "match": "substring_max", "default": 1.0},
    {"name": "SILENCE_PENALTY", "op": "expr", "expr": "CUISINE_MODIFIER * 0.5 if N_ALLERGY_REVIEWS == 0 else 0"},
    {"name": "CUISINE_IMPACT", "op": "expr", "expr": "(CUISINE_MODIFIER * 0.5) + SILENCE_PENALTY"},

    {"name": "MOST_RECENT_YEAR", "op": "max", "field": "meta.year", "where": {"$IS_INCIDENT": true}, "default": 2020},
    {"name": "INCIDENT_AGE", "op": "expr", "expr": "2025 - MOST_RECENT_YEAR"},
    {"name": "RECENCY_DECAY", "op": "expr", "expr": "max(0.3, 1.0 - (INCIDENT_AGE * 0.15))"},

    {"name": "TOTAL_WEIGHT", "op": "sum", "expr": "(5 - meta.stars) + log(meta.useful + 1)", "where": {"$IS_INCIDENT": true}},
    {"name": "CREDIBILITY_FACTOR", "op": "expr", "expr": "TOTAL_WEIGHT / N_TOTAL_INCIDENTS if N_TOTAL_INCIDENTS > 0 else 1.0"},

    {"name": "INCIDENT_IMPACT", "op": "expr", "expr": "ADJUSTED_INCIDENT_SCORE * TRAJECTORY_MULTIPLIER * RECENCY_DECAY * CREDIBILITY_FACTOR"},
    {"name": "TRUST_IMPACT", "op": "expr", "expr": "(1.0 - TRUST_SCORE) * 3.0"},
    {"name": "POSITIVE_CREDIT", "op": "expr", "expr": "N_POSITIVE * TRUST_SCORE * 0.5"},
    {"name": "BASE_RISK", "op": "const", "value": 2.0},
    {"name": "RAW_RISK", "op": "expr", "expr": "BASE_RISK + INCIDENT_IMPACT + TRUST_IMPACT + CUISINE_IMPACT - POSITIVE_CREDIT"},
    {"name": "FINAL_RISK_SCORE", "op": "expr", "expr": "max(0.0, min(20.0, RAW_RISK))"},

    {"name": "VERDICT", "op": "case", "source": "FINAL_RISK_SCORE", "rules": [
      {"when": "< 4.0", "then": "Low Risk"},
      {"when": "< 8.0", "then": "High Risk"},
      {"else": "Critical Risk"}
    ]}
  ],

  "output": ["N_TOTAL_INCIDENTS", "TRUST_SCORE", "ADJUSTED_INCIDENT_SCORE", "TRAJECTORY_MULTIPLIER",
             "RECENCY_DECAY", "CREDIBILITY_FACTOR", "CUISINE_IMPACT", "INCIDENT_IMPACT",
             "TRUST_IMPACT", "POSITIVE_CREDIT", "FINAL_RISK_SCORE", "VERDICT"]
}
```
