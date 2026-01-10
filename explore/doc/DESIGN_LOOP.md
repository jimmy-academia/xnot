# SCALE Task Design Loop Protocol

**Objective**: Ensure every task is "Fail-Positive" (Standard LLMs fail to solve it) before being finalized.

## 1. Parameters & Constraints
-   **Model**: `gpt-5-nano` (Strictly).
-   **Mode**: `Async` (Parallel execution for speed).
-   **Query Format**: Large text with detailed explanations (see `explore/tasks.py` examples).

## 2. The Loop Algorithm

For each Task Candidate (T):

### A. Design Phase (Manual)
1.  **Define Query (Q)**: Detailed natural language question.
2.  **Define Reasoning Primitives (P)**: 
    -   Identify the specific atoms of information required (e.g., "Menu Price", "Review Sentiment X").
    -   Define the logical operator (e.g., `AND`, `IF/ELSE`, `AVG`).
3.  **Define Ground Truth**: The correct Verdict + Correct Premise Values.

### B. Test Phase (Automated)
Run `explore/design_loop.py` using `gpt-5-nano`.
-   Input: Context + Query.
-   Output: Model Reasoning + Final Answer.

### C. Verdict Phase (Human Review)
Evaluate the result using the **Score Formula**:
$$ Score = Verdict \times Avg(P_{correctness}) $$

| Result | Status | Action |
| :--- | :--- | :--- |
| **Pass** (Score ~ 1.0) | **BAD** | **Refine**. Task is too easy. Add complexity, distractors, or deeper inference. |
| **Fail** (Score < 1.0) | **GOOD** | **Freeze**. Record as a valid benchmark task. |
| **Ambiguous** | **WARN** | **Clarify**. Tighten definition or logic. |

## 3. Usage Guide

```bash
# Run design loop asynchronously
python explore/design_loop.py --task G1.1 --async
```
