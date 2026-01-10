# SCALE Framework Technical Design

**Objective**: Repurpose the ANoT codebase into the SCALE benchmark to test typical LLM reasoning failure modes.

## 1. Task Definition & Scope
We have defined 100 tasks across 10 Failure Groups (see `SCALE_TASKS.md`).
- **Input**: `ScaleContext` (Unified wrapper for Restaurant/Product + Reviews).
- **Output**: `ScaleResult` (Binary Verdict + Evidence Scores).

## 2. Architecture Migration Plan

### A. Data Layer (`anot/data/`)
*Current State*: `loader.py` loads specific Yelp fields for ranking.
*Target State*:
1.  Create `scale/dataset.py`:
    - `ScaleContext` dataclass (Domain-Agnostic).
    - `YelpLoader` and `AmazonLoader` adapters.
    - **Slicer**: Logic to generate N=10, 50, 100, 250 contexts from a single source.

### B. Method Layer (`anot/methods/`)
*Current State*: `BaseMethod.evaluate_ranking` expects query + items list.
*Target State*:
1.  Modify `base.py` to support `evaluate_reasoning(query, context)`:
    - Returns structured JSON `{verdict: int, premises: {...}}`.
2.  Update `methods/cot.py`, `methods/react.py`, etc., to output this computed structure instead of a ranking list.

### C. Evaluation Layer (`anot/run/`)
*Current State*: `evaluate.py` computes Hits@K.
*Target State*:
1.  Create `run/evaluate_scale.py`:
    - Implements the new scoring formula: `Verdict * Avg(Premise_Scores)`.
    - Tracks "Process Failure" vs "Outcome Failure".

### D. The 100 Task Definitions (`scale/tasks/`)
Instead of 100 hardcoded functions, we will use a **Configuration-Driven Approach**:
- `tasks_config.json`: Defines the parameters for the 100 tasks (e.g., Target Attribute, Time Window).
- `TaskFactory`: A Python class that reads the config and generates the specific `GroundTruth` logic for that instance.

## 3. Execution Standard
- **Config**: 100 Tasks x 5 Context Sizes x 3 Models.
- **Output**: `results/scale/run_<timestamp>/` containing:
    - `summary.csv`: Aggregated scores by Group (G1-G10).
    - `failures.json`: Detailed trace of where Evidence failed despite correct Verdict.

## 4. Next Steps
1. User Review of this Architecture.
2. Build `scale/core.py` (Abstractions).
3. Build `TaskFactory` and the first 10 G1 tasks.
