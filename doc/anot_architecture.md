# ANoT Architecture

ANoT (Adaptive Network of Thought) is a three-phase evaluation method for ranking tasks over structured multi-source data.

## Overview

```
Phase 1: ReAct Exploration    → Global Plan (N branches)
Phase 2: Branch Expansion     → Fully Expanded LWT Script
Phase 3: Parallel Execution   → Scores → Top-K Ranking
```

---

## Phase 1: ReAct Exploration

**Goal:** Discover data structure and generate a global plan.

The LLM explores the data using lightweight tools (executed in Python, not by LLM):

| Tool | Description | Example |
|------|-------------|---------|
| `count(path)` | Length of array/dict | `count("items")` → `10` |
| `keys(path)` | List of keys at path | `keys("items[0]")` → `["item_id", "attributes", ...]` |
| `union_keys(path)` | Union of keys across items | `union_keys("items[*].attributes")` → `["DriveThru", "WiFi", ...]` |
| `sample(path)` | Sample value (truncated) | `sample("items[0].item_name")` → `"Cafe Roma"` |

**ReAct Loop:**
```
THOUGHT: I need to find items with drive-thru...
ACTION: union_keys("items[*].attributes")
RESULT: ["DriveThru", "WiFi", "NoiseLevel", ...]
THOUGHT: Found DriveThru attribute!
PLAN:
N = 10
RELEVANT_ATTR = DriveThru
(0) = evaluate item 0: check [attributes][DriveThru]
...
(10) = aggregate scores, return top-5
```

**Output:** Structured plan with:
- `n_items`: Number of items
- `relevant_attr`: Key attribute to check
- `branches`: List of (idx, instruction) tuples

---

## Phase 2: Branch Expansion

**Goal:** Expand the global plan into executable LWT (Lightweight Template) script.

This phase uses **no LLM calls** - pure Python string expansion based on local item data.

**Expansion Rules:**
```python
# If attribute exists and is True
(0)=LLM("Cafe Roma has DriveThru=True. Output: 1")

# If attribute exists and is False
(1)=LLM("Front Street Cafe has DriveThru=False. Output: -1")

# If attribute doesn't exist
(2)=LLM("MilkBoy has no DriveThru. Output: -1")
```

**Output:** Fully expanded LWT script:
```
(0)=LLM("Tria Cafe Rittenhouse has no DriveThru. Output: -1")
(1)=LLM("Front Street Cafe has no DriveThru. Output: -1")
...
(5)=LLM("Milkcrate Cafe has DriveThru=True. Output: 1")
...
(10)=LLM("Scores: {0}, {1}, ..., {9}. Return top-5 indices")
```

---

## Phase 3: Parallel Execution

**Goal:** Execute all branches in parallel and aggregate results.

**Execution Model:**
- Uses `asyncio` for parallel LLM calls
- DAG-based layer execution (independent steps run concurrently)
- Results cached by step index

**Scoring Convention:**
| Score | Meaning |
|-------|---------|
| `1` | Matches user criteria |
| `0` | Unknown/uncertain |
| `-1` | Does not match |

**Output:** Top-K indices ranked by score (descending), then by index (ascending for ties).

---

## Key Files

| File | Purpose |
|------|---------|
| `methods/anot.py` | Main ANoT implementation |
| `methods/shared.py` | Shared utilities (parse_script, build_execution_layers) |
| `prompts/task_descriptions.py` | Standard task descriptions |

---

## Example Trace

For request "I need a cafe with a drive-thru option":

```
Phase 1 (46s):
  → count("items") = 10
  → keys("items[0]") = ["item_id", "attributes", ...]
  → union_keys("items[*].attributes") = ["DriveThru", ...]
  → Plan: N=10, RELEVANT_ATTR=DriveThru

Phase 2 (0.2ms):
  → Expanded 11 LWT steps (10 items + 1 aggregation)

Phase 3 (5.3s):
  → Steps 0-9 executed in parallel
  → Scores: [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
  → Top-5: [6, 1, 2, 3, 4] (item 6 = Milkcrate Cafe has DriveThru)
```

---

## Advantages

1. **Adaptive:** Discovers relevant fields dynamically
2. **Efficient:** Phase 2 uses no LLM calls
3. **Parallel:** Phase 3 executes all branches concurrently
4. **Traceable:** Structured logging of all phases
