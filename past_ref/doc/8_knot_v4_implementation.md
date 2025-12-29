# Experiment 8: KNoT v4 - Iterative Hierarchical Planning

**Date:** December 27, 2024

**Goal:** Implement v4 approach with hierarchical planning, AND/OR parsing, and structured debug logging.

## Background

| Version | Clean Accuracy | Notes |
|---------|---------------|-------|
| Original KNoT | 30% (12/40) | Heavy +1 bias |
| v3 (AND/OR parsing) | 35% (14/40) | +5pp improvement |
| Divide approach | 22.5% (9/40) | Worse than baseline |
| **v4 Target** | >50% | |

Previous approaches (iterative, divide) didn't help because they don't address the core issue: proper AND/OR logic parsing and structured script generation.

## v4 Architecture

```
Request → Stage 1 (Hierarchical Planning) → Stage 2 (Script Gen + Refinement) → Stage 3 (Execute)
```

### Stage 1: Hierarchical Planning (6 phases)

| Phase | Description |
|-------|-------------|
| i | Parse request → MUST/SHOULD conditions + AND/OR logic |
| ii | Summarize data → review count, structure |
| iii | High-level plan skeleton |
| iv | Validate plan ↔ request (check-fix loop) |
| v | Expand into DAG structure with review blocks |
| vi | Final plan validation |

### Stage 2: Script Generation (4 phases)

| Phase | Description |
|-------|-------------|
| a | Generate initial script from DAG |
| b | Overall structure check-fix (loop) |
| c | Local refinement per block |
| d | Final validation |

### Expected Script Structure

```
# Review blocks (one per review × condition)
(0)=LLM("Extract speed evidence from R0: {(input)[0]}")
(1)=LLM("Extract food evidence from R0: {(input)[0]}")
...

# Aggregation block (one per condition)
(15)=LLM("Aggregate speed evidence: {(0)}, {(3)}, ... → POSITIVE/NEGATIVE/NEUTRAL")
(16)=LLM("Aggregate food evidence: {(1)}, {(4)}, ... → POSITIVE/NEGATIVE/NEUTRAL")
...

# Decision block (applies AND/OR logic)
(18)=LLM("Apply logic: speed={15} AND (food={16} OR value={17}) → -1/0/1")
```

## Debug Logging System

### DebugLogger Class (`utils/logger.py`)

- Buffers log entries in memory
- Flushes at phase boundaries and on interrupt
- Handles SIGINT/SIGTERM for graceful shutdown

### File Organization

**During execution:**
```
results/{run_dir}/
  temp_debug/
    {item_id}_{request_id}.jsonl
```

**After consolidation:**
```
results/{run_dir}/
  debug/
    by_item/{item_id}_{request_id}.jsonl
    by_phase/1_i.jsonl, 1_ii.jsonl, ...
    summary.json
```

### Log Entry Format

```jsonl
{"ts": "...", "item_id": "...", "request_id": "...", "phase": "1.i", "event": "start", "data": {}}
{"ts": "...", "item_id": "...", "request_id": "...", "phase": "1.i", "event": "llm_call", "data": {"prompt": "...", "response": "..."}}
{"ts": "...", "item_id": "...", "request_id": "...", "phase": "1.i", "event": "end", "data": {"result": {...}}}
```

## Files Created/Modified

| File | Action |
|------|--------|
| `.gitignore` | Added `results/` |
| `utils/__init__.py` | Created - package init |
| `utils/logger.py` | Created - DebugLogger class + consolidate_logs() |
| `knot.py` | Added KnowledgeNetworkOfThoughtV4 class (~500 lines) |
| `main.py` | Added v4 to --knot-approach, pass run_dir, consolidation step |

## Usage

```bash
# Run v4 with debug logging
python3 main.py --method knot --knot-approach v4 --attack none --limit 5 --parallel \
  --requests data/requests/complex_requests.json --data data/processed/complex_data.jsonl

# Debug output (set KNOT_DEBUG=1)
KNOT_DEBUG=1 python3 main.py --method knot --knot-approach v4 --attack none --limit 1
```

## Key Design Decisions

1. **Hierarchical check-fix loops**: Both Stage 1 (planning) and Stage 2 (script gen) use check-fix patterns to ensure correctness

2. **DAG-based structure**: Plan is expanded into a directed acyclic graph where:
   - Review blocks can run in parallel
   - Aggregation depends on review blocks
   - Decision depends on aggregation

3. **Explicit AND/OR parsing**: Phase 1.i extracts structured logic from natural language requests

4. **Debug logging with graceful shutdown**: Buffer + flush pattern minimizes latency while preserving logs on interrupt

## Next Steps

1. Run v4 on clean data and compare to v3 (35% baseline)
2. Analyze debug logs to identify failure patterns
3. Tune prompts based on failure analysis
