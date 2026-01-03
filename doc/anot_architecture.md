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
│              Phase 1: Planning                           │
│  • Extract conditions from request                       │
│  • Prune items that fail attribute checks                │
│  • Generate LWT skeleton + message for Phase 2           │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 2: ReAct Expansion                    │
│  • Check NEEDS_EXPANSION in message                      │
│  • If no → call done() immediately                       │
│  • If yes → use tools to expand LWT, then done()         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 3: Execution                          │
│  • Execute LWT steps with variable substitution          │
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

## Phase 1: Planning

**Goal:** Extract conditions, prune items, and generate LWT skeleton.

### Input
- `context`: User's natural language request
- `items`: Compact item representations with attributes

### Process
1. Extract conditions from request (e.g., DriveThru=True, GoodForKids=True)
2. Find items that match ALL conditions
3. Generate LWT skeleton with matching indices
4. Output message explaining conditions and remaining items

### Output
Two strings:
- `lwt_skeleton`: List of LWT step strings
- `message`: Natural language explanation for Phase 2

### Prompt

```
Analyze the user request and rank items.

[USER REQUEST]
{context}

[ITEMS]
{items_compact}

[TASK]
1. Extract conditions (e.g., DriveThru=True, GoodForKids=True, HasTV=False)
2. Find which items match ALL conditions
3. Output the matching item indices

[OUTPUT FORMAT]
===LWT_SKELETON===
(final)=LLM("User wants: {context}. Item(s) that match: [LIST INDICES]. Output the best index.")

===MESSAGE===
CONDITIONS: <list>
REMAINING: <indices of matching items>
NEEDS_EXPANSION: no
```

### Example Output

```
===LWT_SKELETON===
(final)=LLM("User wants: cafe with drive-thru, kid-friendly, without TVs. Item(s) that match: [5]. Output the best index.")

===MESSAGE===
CONDITIONS: DriveThru=True, GoodForKids=True, HasTV=False
REMAINING: 5
NEEDS_EXPANSION: no
```

---

## Phase 2: ReAct Expansion

**Goal:** Expand LWT skeleton if needed using tools, then finalize.

### Input
- `lwt_skeleton`: LWT steps from Phase 1
- `message`: Natural language explanation from Phase 1
- `query`: Full data dict for read() tool access

### Process
1. Read message to check NEEDS_EXPANSION
2. If "no" → call `done()` immediately (pass-through)
3. If "yes" → use tools to expand LWT, then `done()`

### Tools

| Tool | Description |
|------|-------------|
| `done()` | Finish and return final LWT |
| `lwt_list()` | Show current LWT steps with indices |
| `lwt_insert(idx, "step")` | Insert step at index |
| `lwt_set(idx, "step")` | Replace step at index |
| `read(path)` | Get data at path (e.g., `read("items[5].item_data")`) |

### Prompt

```
Check if the LWT skeleton needs expansion, then call done().

[MESSAGE FROM PHASE 1]
{message}

[CURRENT LWT]
{lwt_skeleton}

[TOOLS]
- done() → finish (call this when skeleton is complete)
- lwt_insert(idx, "step") → add step (only if needed)
- read(path) → get data (only if needed)

[DECISION]
Look at NEEDS_EXPANSION in the message:
- If "no" → just call done() now
- If "yes" → use tools to add steps, then done()

What is your action?
```

### Pass-Through Case

For simple attribute-only conditions (most common case):
1. Phase 1 sets `NEEDS_EXPANSION: no`
2. Phase 2 sees this and calls `done()` immediately
3. Returns original skeleton unchanged

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

Two types of variables:
- `{(input)}["key"][idx]` - Access query data
- `{(step_id)}` - Access previous step output

Examples:
```
# Query data access
{(input)}["items"][5]["attributes"]["DriveThru"]

# Step output access
{(5.check)} → substituted with step 5.check's output
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

For request "cafe with drive-thru, kid-friendly, without TVs":

```
Phase 1 (15s):
  → Extract conditions: DriveThru=True, GoodForKids=True, HasTV=False
  → Check items 0-9 against conditions
  → Item 5 (Milkcrate Cafe) passes all conditions
  → Skeleton: (final)=LLM("...Item(s) that match: [5]...")
  → Message: CONDITIONS: ..., REMAINING: 5, NEEDS_EXPANSION: no

Phase 2 (3s):
  → Read message, see NEEDS_EXPANSION: no
  → Call done() immediately
  → Return skeleton unchanged

Phase 3 (4s):
  → Execute: (final)=LLM("User wants: cafe with... Item(s): [5]...")
  → LLM output: "5"
  → Final ranking: 6 (1-indexed)
```

---

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `phase1_plan()` | `methods/anot.py` | Extract conditions, prune, generate skeleton |
| `phase2_expand()` | `methods/anot.py` | ReAct loop with LWT tools |
| `phase3_execute()` | `methods/anot.py` | Parallel LWT execution |
| `tool_lwt_list()` | `methods/anot.py` | Show LWT steps |
| `tool_lwt_insert()` | `methods/anot.py` | Insert LWT step |
| `tool_read()` | `methods/anot.py` | Read query data |
| `substitute_variables()` | `utils/parsing.py` | Variable substitution |

---

## LWT Syntax

LWT (Lightweight Template) is the execution script format:

```
(step_id)=LLM("prompt text with {(variable)} substitution")
```

Examples:
```
(final)=LLM("User wants: cafe with drive-thru. Items: [5]. Output best index.")
(5.check)=LLM("Does item 5 have {(input)}[\"items\"][5][\"attributes\"][\"DriveThru\"]} = True?")
(agg)=LLM("Results: {(5.check)}. Count matches.")
```

---

## Key Design Principles

1. **LLM-to-LLM Communication** - Phases communicate via strings (skeleton + message), not parsed Python objects
2. **Condition-Based Pruning** - Phase 1 prunes items based on attribute conditions
3. **Pass-Through Case** - Phase 2 can complete instantly if no expansion needed
4. **Parallel Execution** - Phase 3 executes independent steps concurrently
5. **Pure Variable Passing** - LLM outputs passed as-is to subsequent steps
