# Logging Infrastructure

This document describes the two-level logging system for debugging and analysis.

## Overview

| Level | Scope | Output File | Purpose |
|-------|-------|-------------|---------|
| Global | All methods | `usage.jsonl` | LLM call tracking with context |
| Method-specific | ANoT only | `anot_trace.jsonl` | Phase-level structured output |

---

## Global Debug Logging

### Location
- **File:** `utils/usage.py`
- **Output:** `results/{run}/usage.jsonl`

### Schema

Each LLM call is recorded with:

```json
{
  "timestamp": "2026-01-02T21:31:10.431448",
  "model": "gpt-5-nano",
  "provider": "openai",
  "prompt_tokens": 452,
  "completion_tokens": 2218,
  "total_tokens": 2670,
  "cost_usd": 0.00139,
  "latency_ms": 16278.11,
  "context": {
    "method": "anot",
    "phase": 1,
    "step": "explore_0"
  },
  "prompt_preview": "You are exploring data to plan a RANKING task...",
  "response_preview": "THOUGHT: I will start by counting the number of items..."
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 timestamp |
| `model` | string | Model name (e.g., "gpt-5-nano") |
| `provider` | string | "openai", "anthropic", or "local" |
| `prompt_tokens` | int | Input token count |
| `completion_tokens` | int | Output token count |
| `total_tokens` | int | Total tokens used |
| `cost_usd` | float | Estimated cost in USD |
| `latency_ms` | float | API call latency in milliseconds |
| `context` | object | Optional call context (method, phase, step) |
| `prompt_preview` | string | First 200 chars of prompt |
| `response_preview` | string | First 200 chars of response |

### Usage

```python
from utils.llm import call_llm

# With context for tracing
response = call_llm(
    prompt="...",
    system="...",
    context={"method": "anot", "phase": 1, "step": "explore_0"}
)
```

---

## ANoT-Specific Trace

### Location
- **File:** `methods/anot.py`
- **Output:** `results/{run}/anot_trace.jsonl`

### Schema

Each evaluation request generates a trace:

```json
{
  "request_id": "R00",
  "context": "I need a cafe with a drive-thru option - I can't get my kids out of the car",

  "phase1": {
    "exploration_rounds": [
      {"round": 0, "action": "count(\"items\")", "result": "10", "latency_ms": 0.07},
      {"round": 1, "action": "keys(\"items[0]\")", "result": "[...]", "latency_ms": 0.03},
      {"round": 2, "action": "union_keys(\"items[*].attributes\")", "result": "[...]", "latency_ms": 0.07}
    ],
    "plan": {
      "n_items": 10,
      "relevant_attr": "DriveThru",
      "n_branches": 11
    },
    "latency_ms": 46062.06
  },

  "phase2": {
    "expanded_lwt": [
      "(0)=LLM(\"Tria Cafe Rittenhouse has no DriveThru. Output: -1\")",
      "(1)=LLM(\"Front Street Cafe has no DriveThru. Output: -1\")",
      "(5)=LLM(\"Milkcrate Cafe has DriveThru=True. Output: 1\")"
    ],
    "latency_ms": 0.21
  },

  "phase3": {
    "step_results": {
      "0": {"output": "-1", "latency_ms": 2606.00},
      "5": {"output": "1", "latency_ms": 2096.96}
    },
    "final_scores": [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
    "top_k": [6, 1, 2, 3, 4],
    "latency_ms": 5349.06
  }
}
```

### Phase Details

| Phase | Key Fields | Description |
|-------|------------|-------------|
| **phase1** | `exploration_rounds`, `plan` | ReAct exploration actions and resulting plan |
| **phase2** | `expanded_lwt` | Fully expanded LWT script steps |
| **phase3** | `step_results`, `final_scores`, `top_k` | Execution results and ranking |

---

## Enabling Debug Output

Set the `KNOT_DEBUG` environment variable:

```bash
KNOT_DEBUG=1 python main.py --method anot --candidates 10 --limit 1 --dev
```

This enables verbose console output showing:
- Exploration rounds (Phase 1)
- Branch expansions (Phase 2)
- Step executions (Phase 3)

---

## Viewing Logs

```bash
# View usage log (first entry)
head -1 results/dev/XXX_anot/usage.jsonl | python -m json.tool

# View ANoT trace
cat results/dev/XXX_anot/anot_trace.jsonl | python -m json.tool

# Count LLM calls
wc -l results/dev/XXX_anot/usage.jsonl
```
