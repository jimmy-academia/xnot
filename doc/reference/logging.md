# Logging Infrastructure

This document describes the logging system for benchmarking and analysis.

## Overview

| File | Scope | Purpose |
|------|-------|---------|
| `results_{n}.jsonl` | All methods | Predictions + per-request usage |
| `usage.jsonl` | All methods | Consolidated per-request usage (merged across runs) |
| `anot_trace.jsonl` | ANoT only | Phase-level structured trace |
| `debug.log` | ANoT only | Full debug log (LLM prompts/responses, always-on) |
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

Each evaluation request generates a detailed trace following the three-phase architecture:

```json
{
  "request_id": "R00",
  "context": "Looking for a cafe with a drive-thru, kid-friendly, without TVs",

  "phase1": {
    "skeleton": [
      "(final)=LLM(\"User wants: cafe with drive-thru, kid-friendly, without TVs. Item(s) that match: [5]. Output the best index.\")"
    ],
    "message": "CONDITIONS: [\"DriveThru=True\", \"GoodForKids=True\", \"HasTV=False\"]\nREMAINING: [5]\nNEEDS_EXPANSION: no",
    "latency_ms": 15234.56
  },

  "phase2": {
    "expanded_lwt": [
      "(final)=LLM(\"User wants: cafe with drive-thru, kid-friendly, without TVs. Item(s) that match: [5]. Output the best index.\")"
    ],
    "react_iterations": 1,
    "latency_ms": 3012.45
  },

  "phase3": {
    "step_results": {
      "final": {
        "output": "5",
        "latency_ms": 3799.84,
        "prompt_tokens": 340,
        "completion_tokens": 45
      }
    },
    "top_k": [6],
    "final_output": "5",
    "latency_ms": 3801.23
  }
}
```

### Phase Details

| Phase | Key Fields | Description |
|-------|------------|-------------|
| **phase1** | `skeleton`, `message`, `latency_ms` | Planning: extract conditions, prune items, generate LWT skeleton |
| **phase2** | `expanded_lwt`, `react_iterations`, `latency_ms` | ReAct expansion using tools (lwt_list, lwt_insert, read, done) |
| **phase3** | `step_results`, `top_k`, `final_output`, `latency_ms` | Parallel DAG execution with per-step token usage |

### Phase 1 Output Format

The `message` field contains structured information for Phase 2:
- `CONDITIONS`: List of attribute conditions extracted from user request
- `REMAINING`: Indices of items that match all conditions
- `NEEDS_EXPANSION`: "yes" or "no" - tells Phase 2 whether to expand the LWT

### Phase 2 ReAct Tools

| Tool | Description |
|------|-------------|
| `done()` | Finish and return final LWT |
| `lwt_list()` | Show current LWT steps with indices |
| `lwt_insert(idx, "step")` | Insert step at index |
| `lwt_set(idx, "step")` | Replace step at index |
| `read(path)` | Read data (e.g., `read("items[5].item_data")`) |

### Token Tracking

Phase 3 records per-step token usage:
- `prompt_tokens`: Input tokens for the LLM call
- `completion_tokens`: Output tokens from the LLM response

This enables fine-grained cost analysis and optimization of individual steps.

---

## Terminal Progress Display

### ANoT Rich Display

ANoT provides a live Rich terminal display during evaluation:

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

### Baseline Methods Progress

Baseline methods (cot, ps, listwise, weaver) show a simple one-line progress during evaluation:

```
Progress: 49/50 | Tokens: 608,357 | $0.1991
```

This updates in-place after each request completes, showing:
- **Progress**: completed/total requests
- **Tokens**: cumulative token count
- **Cost**: cumulative cost in USD

---

## Scaling Experiment Output

When running without `--candidates` (scaling experiment):

```
run_dir/
  results_10.jsonl      # Predictions for 10 candidates
  results_20.jsonl      # Predictions for 20 candidates
  results_30.jsonl      # Predictions for 30 candidates
  results_40.jsonl      # Predictions for 40 candidates
  results_50.jsonl      # Predictions for 50 candidates
  usage.jsonl           # Merged usage across all scale points
  scaling_summary.json  # Summary with Hits@1, Hits@5 per scale
  config.json           # Run configuration
```

### Scaling Summary Table

The scaling experiment prints a compact summary table:

```
======================================================================
SCALING EXPERIMENT SUMMARY: cot
======================================================================

                         Scaling Results
┌────┬─────┬─────┬─────┬─────┬─────┬─────┬─────────────────────┬────────┐
│ N  │ Req │ @1  │ @2  │ @3  │ @4  │ @5  │ Usage (tok, $, time)│ Status │
├────┼─────┼─────┼─────┼─────┼─────┼─────┼─────────────────────┼────────┤
│ 10 │ 100 │ 80% │ 86% │ 88% │ 90% │ 92% │ 608k, $0.20, 45s    │ ok     │
│ 20 │ 100 │ 76% │ 82% │ 86% │ 88% │ 90% │ 912k, $0.30, 68s    │ ok     │
│ 30 │ 100 │ 72% │ 80% │ 84% │ 86% │ 88% │ 1.2M, $0.40, 92s    │ ok     │
│ 40 │ 100 │ 68% │ 76% │ 82% │ 84% │ 86% │ 1.5M, $0.50, 115s   │ ok     │
│ 50 │ 100 │ 64% │ 72% │ 78% │ 82% │ 84% │ 1.8M, $0.60, 140s   │ ok     │
└────┴─────┴─────┴─────┴─────┴─────┴─────┴─────────────────────┴────────┘
```

Column descriptions:
- **N**: Number of candidates
- **Req**: Number of requests
- **@1-@5**: Hits@K accuracy (integer percent)
- **Usage**: Tokens (k/M), cost ($), latency (seconds)
- **Status**: ok, skipped, context_exceeded, no_requests

---

## ANoT Debug Log (`debug.log`)

### Overview

ANoT writes a detailed debug log to `{run_dir}/debug.log` on every run. This is always-on and requires no configuration.

### Behavior

| Feature | Behavior |
|---------|----------|
| **Enabled** | Always (no env var needed) |
| **Location** | `{run_dir}/debug.log` |
| **Mode** | Overwrite (each run replaces previous) |
| **Terminal output** | None (file only) |
| **Detail level** | Full (equivalent to old ANOT_DEBUG=3) |

### Content

The debug log includes:
- **Timestamps** for each operation (`[HH:MM:SS.mmm]`)
- **Phase and request prefixes** (`[P1:R00]`, `[P2:R01]`, etc.)
- **Full LLM prompts and responses** for every call
- **Item data** passed to each phase
- **Skeleton generation** and expansion steps

### Example Output

```
=== ANoT Debug Log @ 2026-01-03T17:09:15.746004 ===
[17:09:15.746] [INIT:R00] Ranking 10 items for: Looking for a cafe with a drive-thru...
[17:09:15.746] [P1:R00] Planning for: Looking for a cafe with a drive-thru...
[17:09:15.746] [P1:R00] Compact items:
Item 0: "Tria Cafe Rittenhouse" - Alcohol='beer_and_wine', Ambience=...

============================================================
[17:09:27.887] [P1:R00] LLM Call: plan
============================================================
PROMPT:
Analyze the user request and rank items.
...
----------------------------------------
RESPONSE:
===LWT_SKELETON===
(final)=LLM("User wants: ...")
===MESSAGE===
CONDITIONS: ["DriveThru=True", "GoodForKids=True", "HasTV=False"]
REMAINING: [5]
============================================================
```

### Performance

File I/O adds negligible latency (~0.01-0.1ms per write) compared to LLM calls (~6,000ms average). The synchronous writes do not materially impact async execution.

### Debugging Workflow

```bash
# Check debug log after a run
cat results/dev/XXX_anot/debug.log | less

# Search for specific request
grep "R00" results/dev/XXX_anot/debug.log

# Find LLM calls
grep "LLM Call:" results/dev/XXX_anot/debug.log
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

---

## Related Documentation

- [Configuration Reference](configuration.md) - Results directory structure
- [Run Experiments Guide](../guides/run_experiments.md) - How to run evaluations
- [ANoT Architecture](../paper/anot_architecture.md) - Three-phase design details
