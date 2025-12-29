# XNoT Experiment Plan

## Current Results

| Method | Model | Accuracy |
|--------|-------|----------|
| KNoT | gpt-4o-mini | 56% |
| CoT | gpt-4o-mini | 40% |

## Target Specs (3-day sprint)

### Model
- **gpt-5-nano** for everything (planner, worker, baseline runner)

### Data
- **Structure-rich text** format
- Focus on JSON first; expand to HTML, LaTeX later

### Related Work
- StrucText-Eval (ACL 2025): identified benchmark problem but not solutions
- Paper: https://arxiv.org/pdf/2406.10621

---

## Task Domain

### Primary: Customer Review Recommendation
Expand later to: user support triage, public forum opinion mining

### Rich Data Structure

**Context** = multiple conditions:
- Condition on metadata
- Condition on review content (with varying severity)
- Condition on reviewer metadata (e.g., "friend of friend" has higher priority)

**Query** = restaurant info:
- Metadata with multiple subcategories (Yelp existing data)
- Multiple reviews, each containing:
  - Reviewer metadata (social friend list, other Yelp/synthetic info)
  - Review text
  - Review metadata (star count, datetime, etc.)

---

## Attack & Heterogeneity

Focus on review text first; other fields later.

### Heterogeneity
- Very long review (adding irrelevant real review pieces or duplicating text)

### Attacks
- False positive/negative review (LLM-generated)
- Prompt injection attack (LLM-generated)

---

## Problem & Motivation

Real-world applications need to handle information that:
1. Has rich metadata
2. Is sourced from multiple users

**"Multiple users"** → data has heterogeneity and potentially adversarial content
**"Rich metadata"** → data is structured text

### Why Baselines Fail

1. Most prompt schemes and agents work with pure text using full context → gets lost in rich structure and heterogeneity
2. Most prompt schemes cannot handle adversarial cases (prompting for adversarial awareness harms normal case performance - CoT preliminary tests)

---

## NoT Method Plan

### Phase 1: Seed Workflow

**Given:** Context + data schema

**Conduct:**
1. Plan (iterative check)
2. Translate to LWT (iterative check)

*Iterative check ensures context (user requirement) is correctly followed*

### Phase 1b: Workflow Adaptation

**Given:** Query (rich structure)

**Conduct:**
1. Understand structure
2. Understand content condition (check heterogeneity/potential attack)
3. Add steps in seed workflow (to deal with heterogeneity/potential attack)

### Phase 2: Execution

Run LWT script and obtain the final answer.

### Why NoT Works

**Need to split context cleanly for reliable performance:**
- Example: One review is very long but lacks key points; other short reviews mention key points but get forgotten

**Context + query change requires dynamic input selection and dynamic LLM operation:**
- Example 1: Context says "prioritize friend reviews" → compare query metadata (friend field) vs reviewer metadata (friend field)
- Example 2: Context says "find parking" → check restaurant metadata
- Example 3: Context says "check if delicious or has queue" → check review text
- If query has heterogeneity or potential adversarial content → handle accordingly

---

## Experiment Steps

### 1. Data & Benchmark Preparation

#### Query Data
- Use real-world Yelp data
- Select 20 restaurants, 20 reviews each
- Define metadata fields + reviewer fields

#### Complex Context
- Different priority levels
- AND/OR logic
- Use of different fields in query (JSON data)

#### Ground Truth
- Define aggregation logic
- Locate correct sub-condition answers
- Operate aggregation logic for final answer

#### Attack/Heterogeneity Methods
- Apply to review text

### 2. Baseline Experiments

**Methods:** CoT, ReACT

**Target:**
- Adjust data until <15% clean accuracy (and lower attacked accuracy)

### 3. Method Refinement

**Target:**
- ~60% clean accuracy
- Similar attacked accuracy (robustness)
