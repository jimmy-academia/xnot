# Experiment 1: gpt-4o-mini Baseline Comparison

**Date:** December 2024

**Goal:** Compare knot vs cot methods using gpt-4o-mini for all LLM calls.

**Data:** Real Yelp restaurant reviews (20 restaurants selected via automated pipeline)

## Configuration

| Method | Model | Description |
|--------|-------|-------------|
| cot | gpt-4o-mini | Chain-of-Thought: single-pass reasoning |
| knot | gpt-4o-mini (planner + worker) | Knowledge Network of Thought: dynamic script generation + execution |

## Results

| Method | Accuracy | Notes |
|--------|----------|-------|
| cot | 40% | Single-pass reasoning |
| knot | **56%** | Dynamic script generation + execution |

**Improvement: +16 percentage points**

## Key Findings

1. **knot outperforms cot by 16 percentage points** on real Yelp data
2. knot's 2-phase approach (knowledge → script → execute) provides better reasoning decomposition
3. Both methods use same underlying model, difference is in reasoning structure

## Methodology

- **Data Source:** Yelp Academic Dataset
- **Selection:** Automated pipeline (`data/select_real_data.py`) selects restaurants with varied review sentiment
- **Request Types (5):**
  - R0: Speed/wait time
  - R1: Consistency
  - R2: Romantic ambiance
  - R3: Value for money
  - R4: Food quality
- **Gold Labels:** Computed automatically based on keyword matching in reviews
- **Output:** -1 (not recommend), 0 (unclear), 1 (recommend)

## Architecture Comparison

### Chain-of-Thought (cot)
```
User Query → LLM (single pass) → Answer
```

### Knowledge Network of Thought (knot)
```
User Query → Generate Knowledge (planner) → Generate Script (planner) → Execute Script (worker) → Answer
```

The knot approach decomposes reasoning into explicit steps, allowing for more structured analysis of multi-faceted restaurant reviews.
