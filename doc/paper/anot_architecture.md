# ANoT Architecture

ANoT (Adaptive Network of Thought) is a three-phase evaluation method for ranking tasks over structured multi-source data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                          │
│     "cafe with drive-thru, kid-friendly, without TVs"   │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 1: Strategy Extraction                │
│  • Analyze request to extract CONDITIONS                 │
│  • Classify condition types: [ATTR], [AMBIENCE], etc.    │
│  • Output STRATEGY (not LWT) + message for Phase 2       │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 2: LWT Generation                     │
│  • Receive STRATEGY from Phase 1                         │
│  • Generate LWT steps with path-based data access        │
│  • Use {(context)}[item][attr] for single value access   │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 3: Execution                          │
│  • Substitute {(context)}[path] with actual values       │
│  • Parallel DAG execution within layers                  │
│  • Final step outputs ranking indices                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Final Ranking                          │
│                    6 (1-indexed)                         │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Strategy Extraction

**Goal:** Analyze request and output evaluation STRATEGY (not LWT).

### Key Design
Phase 1 sees the SCHEMA (1-2 example items) to understand available fields, but does NOT scan all items. It outputs a strategy describing what conditions to check.

### Input
- `query`: User's natural language request
- `schema_compact`: 1-2 example items showing available fields
- `n_items`: Total number of items to evaluate

### Process
1. Extract ALL conditions from request
2. Classify each condition by type:
   - `[ATTR]` - Direct attribute lookup (e.g., GoodForKids=True)
   - `[AMBIENCE]` - Nested in Ambience dict (e.g., Ambience.hipster=True)
   - `[MEAL]` - Nested in GoodForMeal dict
   - `[HOURS]` - Operating hours check
   - `[REVIEW_TEXT]` - Keyword search in reviews
3. Identify logic: AND/OR relationships between conditions
4. DO NOT attempt to find matching items - Phase 2 handles evaluation

### Output
Two strings:
- `strategy`: Structured strategy with conditions and logic
- `message`: Notes for Phase 2

### Example Output

```
===STRATEGY===
CONDITIONS:
  1. [ATTR] attributes.GoodForKids = True
  2. [ATTR] attributes.DriveThru = True
  3. [ATTR] attributes.HasTV = False

LOGIC: AND(1, 2, 3)
TOTAL_ITEMS: 10

===MESSAGE===
Brief notes for Phase 2 (optional)
```

---

## Phase 2: LWT Generation

**Goal:** Generate LWT steps from strategy using ReAct loop.

### Input
- `strategy`: Evaluation strategy from Phase 1 (conditions + logic)
- `message`: Notes from Phase 1
- `n_items`: Total number of items to evaluate
- `query`: Full data dict for read() tool access

### Process
1. Receive strategy (starts with EMPTY LWT)
2. Use tools to probe data structure if needed
3. Generate LWT steps for each condition with path-based access
4. Generate final aggregation step
5. Call `done()` to finish

### Tools

| Tool | Description |
|------|-------------|
| `done()` | Finish and return final LWT |
| `lwt_list()` | Show current LWT steps with indices |
| `lwt_insert(idx, "step")` | Insert step at index |
| `lwt_set(idx, "step")` | Replace step at index |
| `read(path)` | Probe data structure (e.g., `read("items[0].attributes")`) |

### Path Syntax for LWT Steps

Use `{(context)}[path]` to access restaurant data in Phase 3:
- `{(context)}[1]` - Item 1 (1-indexed)
- `{(context)}[1][attributes]` - Item 1's attributes dict
- `{(context)}[1][attributes][GoodForKids]` - Single value (True/False)
- `{(context)}[1][attributes][Ambience][hipster]` - Nested value

**CRITICAL:** List ALL items explicitly. Do NOT use "..." - it won't be expanded!

### Example LWT Generation

For 10 items with conditions GoodForKids=True, WiFi=free:

```
lwt_insert(1, "(c1)=LLM('GoodForKids: 1={(context)}[1][attributes][GoodForKids], 2={(context)}[2][attributes][GoodForKids], 3={(context)}[3][attributes][GoodForKids], ... 10={(context)}[10][attributes][GoodForKids]. Which are True? Output: [indices]')")
lwt_insert(2, "(c2)=LLM('WiFi: 1={(context)}[1][attributes][WiFi], 2={(context)}[2][attributes][WiFi], ... 10={(context)}[10][attributes][WiFi]. Which are free? Output: [indices]')")
lwt_insert(3, "(final)=LLM('c1={(c1)}, c2={(c2)}. Score: +1 per match. Output top-5: [best,2nd,3rd,4th,5th]')")
done()
```

---

## Phase 3: Execution

**Goal:** Execute LWT steps and produce final ranking.

### Process
1. Parse LWT script into steps
2. Build DAG based on dependencies (`{(step_id)}` references)
3. Execute steps in parallel within each layer
4. Substitute variables in prompts
5. Return final step's output

### Variable Substitution

Three variable types defined in `utils/parsing.py:substitute_variables()`:

| Variable | Maps to | Description |
|----------|---------|-------------|
| `{(context)}` | items dict | Restaurant data (1-indexed) |
| `{(input)}` / `{(items)}` | items dict | Same as context |
| `{(query)}` | user_query | User's request text |
| `{(step_id)}` | cache[step_id] | Previous step output |

### Path-Based Data Access

Use `{(context)}[path]` to access single leaf values:
```
# Access item 1's GoodForKids attribute
{(context)}[1][attributes][GoodForKids] → True

# Access nested Ambience attribute
{(context)}[2][attributes][Ambience][hipster] → False

# Access categories list
{(context)}[3][categories] → ["Cafe", "Coffee & Tea"]
```

### Example Substitution

**Before (LWT step):**
```
1={(context)}[1][attributes][GoodForKids], 2={(context)}[2][attributes][GoodForKids]. Which are True?
```

**After (filled prompt):**
```
1=True, 2=False. Which are True?
```

### Parallel Execution

Steps are organized into layers based on dependencies:
- Layer 0: Steps with no dependencies (run in parallel)
- Layer 1: Steps depending on Layer 0 (run in parallel after Layer 0)
- ...and so on

Uses `asyncio.gather()` for concurrent LLM calls within each layer.

---

## Debug Mode

Control debug output via `ANOT_DEBUG` environment variable:

| Level | Name | Output |
|-------|------|--------|
| 0 | OFF | Rich table display only |
| 1 | SUMMARY | Phase transitions, final results |
| 2 | VERBOSE | Per-phase progress, LWT steps |
| 3 | FULL | Complete LLM prompts and responses |

**Usage:**
```bash
ANOT_DEBUG=2 python main.py --method anot --data philly_cafes --limit 1
```

**Debug Output Format:**
```
[HH:MM:SS.mmm] [P1:R00] Planning for: cafe with drive-thru...
[HH:MM:SS.mmm] [P1:R00] Skeleton: (final)=LLM("User wants...")
[HH:MM:SS.mmm] [P2:R00] ReAct expansion: 1 initial steps...
[HH:MM:SS.mmm] [P2:R00] ReAct done after 1 iterations
[HH:MM:SS.mmm] [P3:R00] Executing 1 steps in 1 layers...
[HH:MM:SS.mmm] [P3:R00] Final ranking: 6
```

---

## Example Trace

For request "cafe with drive-thru, kid-friendly, without TVs" with 10 items:

```
Phase 1 (5s):
  → Extract conditions from request
  → Output strategy:
    CONDITIONS:
      1. [ATTR] DriveThru = True
      2. [ATTR] GoodForKids = True
      3. [ATTR] HasTV = False
    LOGIC: AND(1, 2, 3)
    TOTAL_ITEMS: 10

Phase 2 (8s):
  → Receive strategy, start with empty LWT
  → Generate LWT steps:
    (c1)=LLM('DriveThru: 1={(context)}[1][attributes][DriveThru], ..., 10={(context)}[10][attributes][DriveThru]. Which are True? Output: [indices]')
    (c2)=LLM('GoodForKids: 1={(context)}[1][attributes][GoodForKids], ..., 10={(context)}[10][attributes][GoodForKids]. Which are True? Output: [indices]')
    (c3)=LLM('HasTV: 1={(context)}[1][attributes][HasTV], ..., 10={(context)}[10][attributes][HasTV]. Which are False? Output: [indices]')
    (final)=LLM('c1={(c1)}, c2={(c2)}, c3={(c3)}. Score items +1 per match. Output top-5.')
  → Call done()

Phase 3 (4s):
  → Layer 0: Execute c1, c2, c3 in parallel
    c1 filled: '1=True, 2=False, ..., 10=True. Which are True?' → [1, 5, 10]
    c2 filled: '1=True, 2=True, ..., 10=False. Which are True?' → [1, 2, 5]
    c3 filled: '1=True, 2=False, ..., 10=True. Which are False?' → [2, 5, 7]
  → Layer 1: Execute final
    final filled: 'c1=[1, 5, 10], c2=[1, 2, 5], c3=[2, 5, 7]. Score items...'
    → Output: [5, 1, 2, 10, 7]
  → Final ranking: 5 (1-indexed)
```

---

## Package Structure

ANoT is organized as a Python package for maintainability:

```
methods/anot/
├── __init__.py      Re-exports AdaptiveNetworkOfThought, create_method
├── core.py          Main class with phases and display
├── helpers.py       Utility functions (DAG building, formatting)
├── prompts.py       LLM prompt constants
└── tools.py         Phase 2 tool implementations
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `AdaptiveNetworkOfThought` | `methods/anot/core.py` | Main class orchestrating all phases |
| `phase1_plan()` | `methods/anot/core.py` | Extract strategy (conditions + logic) |
| `phase2_expand()` | `methods/anot/core.py` | Generate LWT from strategy |
| `phase3_execute()` | `methods/anot/core.py` | Parallel LWT execution |
| `build_execution_layers()` | `methods/anot/helpers.py` | Build DAG for parallel execution |
| `format_items_compact()` | `methods/anot/helpers.py` | Format items for Phase 1 |
| `tool_lwt_list()` | `methods/anot/tools.py` | Show LWT steps |
| `tool_lwt_insert()` | `methods/anot/tools.py` | Insert LWT step |
| `tool_read()` | `methods/anot/tools.py` | Read query data |
| `PHASE1_PROMPT` | `methods/anot/prompts.py` | Phase 1 prompt template |
| `PHASE2_PROMPT` | `methods/anot/prompts.py` | Phase 2 prompt template |
| `substitute_variables()` | `utils/parsing.py` | Variable substitution |

---

## LWT Syntax

LWT (Lightweight Template) is the execution script format:

```
(step_id)=LLM("prompt text with {(variable)} substitution")
```

### Variable Types
- `{(context)}[path]` - Access restaurant data by path
- `{(step_id)}` - Access previous step output
- `{(query)}` - Access user request text

### Examples
```
# Check attribute across all items (using path-based access)
(c1)=LLM("GoodForKids: 1={(context)}[1][attributes][GoodForKids], 2={(context)}[2][attributes][GoodForKids]. Which are True? Output: [indices]")

# Reference previous step output
(final)=LLM("c1={(c1)}, c2={(c2)}. Score items +1 per match. Output top-5.")

# Access nested attributes
(c2)=LLM("Ambience.hipster: 1={(context)}[1][attributes][Ambience][hipster]. Which are True?")
```

### Key Rules
1. Use 1-indexed item numbers: `{(context)}[1]`, `{(context)}[2]`, etc.
2. List ALL items explicitly - "..." notation won't be expanded
3. Access leaf values to minimize token usage

---

## Key Design Principles

1. **Strategy-Centric Design** - Phase 1 extracts strategy (conditions + logic), Phase 2 generates LWT
2. **Path-Based Data Access** - Use `{(context)}[item][attr]` to access single leaf values
3. **Explicit Item Listing** - LWT must list all items explicitly (no "..." shorthand)
4. **Parallel Execution** - Phase 3 executes independent steps concurrently
5. **Partial Match Ranking** - If no perfect match, rank by condition count
