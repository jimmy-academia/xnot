# Code Quality Audit Report

**Date**: 2026-01-03 (Updated)
**Overall Code Health**: 9/10 (improved from 7.5)

---

## Summary

All critical issues have been resolved. The codebase has been significantly refactored for maintainability:

| Risk Level | Original | Completed | Remaining |
|------------|----------|-----------|-----------|
| Low | 5 items | 5 | 0 |
| Medium | 4 items | 4 | 0 |
| High | 6 items | 4 | 2 (optional) |
| **Major Refactors** | 2 items | 2 | 0 |

---

## Major Refactors (Completed)

### 1. Split methods/anot.py (867 → 5 files)

Original `methods/anot.py` was 867 lines doing 6+ unrelated things. Now split into focused modules:

```
methods/anot/
├── __init__.py      (12 lines)   Re-exports
├── core.py         (559 lines)   Main AdaptiveNetworkOfThought class
├── helpers.py       (97 lines)   extract_dependencies, build_execution_layers, format_items_compact
├── prompts.py       (48 lines)   SYSTEM_PROMPT, PHASE1_PROMPT, PHASE2_PROMPT
└── tools.py         (76 lines)   tool_read, tool_lwt_* functions
```

### 2. Split run.py (941 → 6 files)

Original `run.py` was 941 lines mixing 4+ responsibilities. Now split:

```
run/
├── __init__.py      (42 lines)   Re-exports
├── evaluate.py     (256 lines)   evaluate_ranking, compute_multi_k_stats, extract_hits_at
├── io.py           (123 lines)   load/save results and usage
├── orchestrate.py  (194 lines)   run_single, run_evaluation_loop
├── scaling.py      (297 lines)   run_scaling_experiment, SCALE_POINTS
└── shuffle.py       (77 lines)   shuffle utilities
```

---

## High Risk Fixes (Completed)

### 1. Cycle Detection in `build_execution_layers()` ✅
- **Location**: `methods/anot/helpers.py:22-48`
- **Fix**: Added detection for circular dependencies, raises `ValueError`

### 2. Race Condition in `_update_display()` ✅
- **Location**: `methods/anot/core.py:185-210`
- **Fix**: Moved completion check inside lock scope

### 3. eval() Validation in Weaver ✅
- **Location**: `methods/weaver.py:225-233`
- **Fix**: Added DANGEROUS_PATTERNS validation, restricted `__builtins__`

### 4. Bare except Clauses ✅
- **Location**: `utils/parsing.py:38, 51`
- **Fix**: Replaced `except:` with `except Exception:`

### 5. Unused Global Variable ✅
- **Location**: `methods/anot.py:84` (old file, now deleted)
- **Fix**: Removed `_DEBUG_LOG_FILE = None`

---

## High Risk (Optional - Not Blocking)

### 5. Replace Global State with Dependency Injection
- **Files**: `utils/llm.py`, `utils/usage.py`
- **Current**: Global clients and config dicts
- **Status**: Low priority - current approach works, would be breaking change

### 6. Widen Trace Lock Scope
- **Location**: `methods/anot/core.py`
- **Current**: Fine-grained locking
- **Status**: Low priority - no data loss observed in practice

---

## Medium Risk Fixes (All Completed)

- [x] Extract usage recording helper `_record_usage()` in `utils/llm.py`
- [x] Extract progress display helper in evaluation code
- [x] Split `run.py` into modules (see Major Refactors)
- [x] Split `methods/anot.py` into modules (see Major Refactors)

---

## Low Risk Fixes (All Completed)

- [x] Remove unused imports from `methods/shared.py`
- [x] Delete dead function `_get_next_benchmark_run()` from `utils/experiment.py`
- [x] Remove unused `args.selection_name` from `utils/arguments.py`
- [x] Pre-compile regex patterns in anot code
- [x] Consolidate duplicate `setup_logging()` - removed from `utils/logger.py`

---

## Quick Wins Applied

1. **Fixed bare `except:`** in `utils/parsing.py` (2 places)
2. **Removed unused global** `_DEBUG_LOG_FILE` from old anot.py
3. **Added `extract_hits_at()` helper** in `run/evaluate.py` to eliminate duplicate code

---

## Files Modified/Created

### Deleted
- `methods/anot.py` (replaced by package)
- `run.py` (replaced by package)
- `methods/prompts.py` (moved to anot package)

### Created
- `methods/anot/__init__.py`
- `methods/anot/core.py`
- `methods/anot/helpers.py`
- `methods/anot/prompts.py`
- `methods/anot/tools.py`
- `run/__init__.py`
- `run/evaluate.py`
- `run/io.py`
- `run/orchestrate.py`
- `run/scaling.py`
- `run/shuffle.py`

### Modified
- `utils/parsing.py` - fixed bare except clauses
- `methods/weaver.py` - added eval() validation
- `methods/shared.py` - cleaned to defense-only (~22 lines)
- `methods/cot.py` - added DEFENSE_PREAMBLE constants

---

## Recent Improvements

### Shared Task Description Prompts

All methods now import `RANKING_TASK_COMPACT` from `prompts/task_descriptions.py`:
- `methods/anot/prompts.py` - used in PHASE1_PROMPT
- `methods/cot.py` - used in ranking prompts
- `methods/listwise.py` - used in LISTWISE_PROMPT
- `methods/ps.py` - used in ranking prompts

### Progress Display for Baseline Methods

Added live progress display for non-anot methods (`run/evaluate.py`):
```
Progress: 49/50 | Tokens: 608,357 | $0.1991
```

### Compact Scaling Summary Table

Updated scaling experiment summary (`run/scaling.py`) with:
- Compact column headers: N, Req, @1-@5
- Combined usage column: tokens (k/M), cost ($), latency (s)
- Integer percent format

```
│ N  │ Req │ @1  │ @2  │ @3  │ @4  │ @5  │ Usage (tok, $, time)│ Status │
│ 10 │ 50  │ 80% │ 86% │ 88% │ 90% │ 92% │ 608k, $0.20, 45s    │ ok     │
```

---

## ANoT Hardcoded Logic Analysis

The following Python logic in ANoT is hardcoded by design:

| Location | Logic | Justification |
|----------|-------|---------------|
| `helpers.py:format_items_compact()` | Truncates strings >20 chars, shows dicts as `<dict>` | **By design**: ANoT looks at the "spine" (schema) of input JSON in Phase 1, not full data values |
| `core.py:534-543` | Regex extraction of indices from LLM output | **Standard practice**: Typical LLM output parsing with fallback |
| `core.py:365-366` | 2000 char truncation on `read()` tool results | **Safety**: Protection from context length exceptions in Phase 2 |

These are deterministic operations that don't require LLM reasoning:
- Schema extraction is a design choice for efficient planning
- Output parsing is standard post-processing
- Truncation is a safety guard against token limits

---

## Architecture Benefits

After refactoring:

1. **Single Responsibility**: Each file does one thing
2. **Easier Testing**: Functions can be unit tested in isolation
3. **Easier Navigation**: Finding code is intuitive by filename
4. **Reduced Cognitive Load**: No file over 560 lines
5. **Better Maintainability**: Changes to one concern don't risk breaking others
6. **Clearer Dependencies**: Imports show what each module needs
