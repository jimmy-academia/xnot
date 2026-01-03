# Logging Infrastructure

This document describes the logging system for benchmarking and analysis.

## Overview

| File | Scope | Purpose |
|------|-------|---------|
| `results_{n}.jsonl` | All methods | Predictions + per-request usage |
| `usage.jsonl` | All methods | Consolidated per-request usage (merged across runs) |
| `anot_trace.jsonl` | ANoT only | Phase-level structured trace |
| `config.json` | All methods | Run configuration and aggregate stats |

---

## Per-Request Usage (`usage.jsonl`)

### Location
- **Output:** `results/{run}/usage.jsonl`

### Purpose
Consolidated per-request usage across all runs. Supports incremental runs with different `--candidates` or `--limit` values. Records are keyed by `(request_id, n_candidates)` - re-running overwrites previous records.

### Schema

```json
{
  "request_id": "R00",
  "n_candidates": 10,
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "tokens": 1801,
  "cost_usd": 0.0123,
  "latency_ms": 2345,
  "timestamp": "2026-01-02T14:30:22"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Request identifier (e.g., "R00", "R01") |
| `n_candidates` | int | Number of candidates in this run |
| `prompt_tokens` | int | Total input tokens for this request |
| `completion_tokens` | int | Total output tokens for this request |
| `tokens` | int | Total tokens (prompt + completion) |
| `cost_usd` | float | Estimated cost in USD |
| `latency_ms` | float | Total latency in milliseconds |
| `timestamp` | string | ISO 8601 timestamp of last run |

### Merge Behavior

When running incrementally:
- `python main.py --method cot --candidates 10` → adds 10-candidate records
- `python main.py --method cot --candidates 15` → adds 15-candidate records
- Re-running same `(request_id, n_candidates)` → overwrites previous record

---

## Results Files (`results_{n}.jsonl`)

Per-request predictions with embedded usage:

```json
{
  "request_id": "R00",
  "pred_indices": [5, 0, 1, 2, 3],
  "gold_idx": 5,
  "shuffled_gold_pos": 6,
  "gold_restaurant": "abc123",
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "tokens": 1801,
  "cost_usd": 0.0123,
  "latency_ms": 2345
}
```

---

## ANoT-Specific Trace (`anot_trace.jsonl`)

### Location
- **File:** `methods/anot.py`
- **Output:** `results/{run}/anot_trace.jsonl`

### Schema

Each evaluation request generates a detailed trace:

```json
{
  "request_id": "R00",
  "context": "I need a cafe with a drive-thru option",

  "phase1": {
    "exploration_rounds": [
      {"round": 0, "action": "count(\"items\")", "result": "10", "latency_ms": 0.07, "prompt_tokens": 1250, "completion_tokens": 320},
      {"round": 1, "action": "keys(\"items[0]\")", "result": "[...]", "latency_ms": 0.03, "prompt_tokens": 1420, "completion_tokens": 185},
      {"round": 2, "action": "union_keys(\"items[*].attributes\")", "result": "[...]", "latency_ms": 0.07, "prompt_tokens": 1580, "completion_tokens": 210}
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
      "(0)=LLM(\"Tria Cafe has no DriveThru. Output: -1\")",
      "(5)=LLM(\"Milkcrate Cafe has DriveThru=True. Output: 1\")",
      "(10)=LLM(\"Scores: {(0)}, {(1)}, ... Return top-5 indices\")"
    ],
    "latency_ms": 0.21
  },

  "phase3": {
    "step_results": {
      "0": {"output": "-1", "latency_ms": 2606.00, "prompt_tokens": 340, "completion_tokens": 45},
      "5": {"output": "1", "latency_ms": 2096.96, "prompt_tokens": 380, "completion_tokens": 42},
      "10": {"output": "5,0,1,2,3", "latency_ms": 1500.00, "prompt_tokens": 520, "completion_tokens": 65}
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
| **phase1** | `exploration_rounds`, `plan` | ReAct exploration with per-round token usage |
| **phase2** | `expanded_lwt` | Fully expanded LWT script steps (no LLM calls) |
| **phase3** | `step_results`, `final_scores`, `top_k` | Execution results with per-step token usage |

### Token Tracking

Both Phase 1 and Phase 3 record per-step token usage:
- `prompt_tokens`: Input tokens for the LLM call
- `completion_tokens`: Output tokens from the LLM response

This enables fine-grained cost analysis and optimization of individual steps.

---

## ANoT Rich Terminal Display

ANoT provides a live terminal display during evaluation:

```
                    ANoT: 10 candidates, k=5
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Req ┃ Context                         ┃ Phase ┃ Status     ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ R00 │ cafe with drive-thru...         │   ✓   │ 5,0,1,2,3  │
│ R01 │ upscale cafe experience...      │  P3   │ executing  │
│ R02 │ coat check service...           │  P1   │ round 3/10 │
└─────┴─────────────────────────────────┴───────┴────────────┘
Progress: 1/3 | Tokens: 45,678 | $0.1234
```

---

## Scaling Experiment Output

When running `--scaling`:

```
run_dir/
  results_10.jsonl      # Predictions for 10 candidates
  results_15.jsonl      # Predictions for 15 candidates
  ...
  usage.jsonl           # Merged usage across all scale points
  scaling_summary.json  # Summary with Hits@1, Hits@5 per scale
  config.json           # Run configuration
```

---

## Viewing Logs

```bash
# View usage summary
cat results/dev/XXX_anot/usage.jsonl | python -m json.tool

# View ANoT trace for first request
head -1 results/dev/XXX_anot/anot_trace.jsonl | python -m json.tool

# Count requests in usage
wc -l results/dev/XXX_anot/usage.jsonl

# Filter usage by candidate count
grep '"n_candidates": 10' results/dev/XXX_anot/usage.jsonl
```
