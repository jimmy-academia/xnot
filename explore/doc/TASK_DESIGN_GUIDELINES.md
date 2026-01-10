# SCALE Task Design Guidelines

## Core Philosophy
1.  **No Simple Tasks**: Assume basic information retrieval is solved. Tasks must require multi-hop reasoning, synthesis, or judgment.
2.  **Fail-Positive Design**: Tasks should be designed so that models "fail" (score low) unless they strictly follow the complex reasoning path.
3.  **Strict Scoring**:
    - **Verdict Gating**: If the final verdict is correct but the reasoning (primitives) is wrong, the score is 0.
    - **Formula**: `Score = Verdict_Match * Avg(Scoring_Primitive_Matches)`.

## Primitive Taxonomy (The "DAG Components")
To organize complexity, we categorize primitives into two types based on their role in evaluation:

### 1. Non-Scoring Primitives (The "Workings")
*Intermediate steps or raw extractions required for the calculation but NOT graded.*
- **Definition**: Any primitive (whether leaf or node) that is too easy, static, or merely structural.
- **Examples**: Raw counts (`n_flags`), static metadata (`base_cuisine_risk`), or simple intermediary sums.
- **Role**: Essential for the chain of thought but excluded from the final score to prevent "grade inflation" from trivial steps.

### 2. Scoring Primitives (The "Test")
*Complex derived values that prove the model's reasoning capability.*
- **Definition**: High-level nodes in the dependency graph that require correctly integrating multiple inputs.
- **Examples**: `severity_index` (integration of stars+flags), `confidence_score` (ratio calculation), `mitigation_score` (sum of weighted safety proofs), `final_risk`.
- **Role**: The "Graded" questions. Failure here indicates a failure in reasoning.

## Scoring Philosophy (The "Hard Way")

### 1. Strict Scoring Formula
We use a multiplicative formula to punish partial failures.
- **Formula**: `Score = Final_Verdict_Match * Avg(Scoring_Primitive_Matches)`
- **Verdict Gating**: If `Final_Verdict_Match` is 0 (wrong decision), total score is **0**. Even if the intermediate math was perfect, the result is useless.

### 2. DAG Premises (Cascading Errors)
Premises should form a **Directed Acyclic Graph (DAG)** of dependencies.
- **Depth**: Deeper graphs (Extraction -> Aggregation A -> Aggregation B -> Verdict) are harder than shallow ones.
- **Dependency**: `Aggregation B` must depend on `Aggregation A`. Choosing to skip A to guess B results in failure for both.

### 3. Targeted Scoring (The Scoring Mask)
To reach the difficulty target preventing "Free Wins":
- **Rule**: ONLY score specific "Scoring Primitives".
- **Mechanism**: Use `scoring_fields` whitelist in the task definition.

### 4. Difficulty Target
- **Zero-Shot Success Rate**: Aim for **0.05 - 0.15** (5-15%).
- **Headroom**: This ensures the benchmark measures *reasoning capability*, not just instruction following. If a task has >50% zero-shot success, it is too easy. Add complexity Motifs.

## The Motif Repertoire
Reusable logical patterns ("Motifs") to increase complexity and realism.

### 1. Risk Score Matrix (Arithmetic Reasoning)
Instead of vague qualitative rules ("if many bad reviews"), use a strict points system.
- **Base Score**: Fixed points based on static metadata (e.g., Cuisine Risk).
- **Point Modifiers**: +/- points for specific evidence found in text.
- **Thresholds**: Strict cutoffs for verdicts (e.g., "Score >= 5 is High Risk").

### 2. Recency Weighting (Temporal Reasoning)
Information relevance decays over time.
- **Pattern**: Weight reviews differently based on `review_meta['date']`.
- **Example**:
    - "Recent" (Last 12 months): 1.0x Weight.
    - "Legacy" (Older than 12 months): 0.5x Weight.
- **Purpose**: Forces the model to check the Date field and perform date comparison logic.

### 3. Star-Confidence Weighting (Sentiment Correlation)
Use the star rating to validate or weight the text evidence.
- **Pattern**: Flags are weighted by their associated `stars`.
- **Example**:
    - Negative Flag in 1-Star Review: High Credibility (+5 points).
    - Negative Flag in 4-Star Review: Low Credibility (+2 points).
- **Purpose**: Forces integration of Sentiment (Stars) vs Semantic (Text) signals.

### 4. Assumption of Risk (The "Pessimist" Motif)
Failure to prove safety implies danger.
- **Pattern**: Start with a "High Risk" or "Penalty" score (e.g., `Risk = 10`).
- **Reduction**: Only specific *positive* evidence (e.g., "clean") reduces this risk.
- **Purpose**: Prevents "Free Wins" from empty contexts. If the model finds nothing, it must predict "Risk". This counters the LLM's natural "benefit of the doubt" bias.

## Prompt Clarity Principles

**Critical Rule**: Models should fail due to **complexity**, NOT due to **ambiguity**.

### 1. Explicit Variable Definitions
**Every** intermediate variable MUST be defined before use.
- **Bad**: `RECENCY_FACTOR: 1.0 / (1 + (AvgAge_FlaggedReviews / 5.0))`  ← What is `AvgAge_FlaggedReviews`?
- **Good**: 
  ```
  DEFINITIONS:
  - AvgAge_FlaggedReviews: Average age in years of flagged reviews = Avg(2025 - ReviewYear for each flagged review)
  
  OUTPUT:
  - RECENCY_FACTOR: 1.0 / (1 + (AvgAge_FlaggedReviews / 5.0))
  ```

### 2. Clear Semantic Instructions
Use precise, unambiguous language for ALL operations.
- **Bad**: `Count('keyword1', 'keyword2')` ← Count what? Mentions? Reviews?
- **Good**: `Count reviews containing ANY of: 'keyword1', 'keyword2'`

### 3. Explicit Scoring Markers
Mark each primitive as SCORED or NON-SCORED.
- **Purpose**: Helps the model understand which calculations are critical.
- **Format**:
  ```
  L2: Raw Extraction (NON-SCORED - simple keyword counting)
  L3: Basic Aggregation (SCORED)
  ```

### 4. Default Value Specifications
For conditional logic, specify the default/fallback value.
- **Bad**: `RECENCY_FACTOR: 1.0 / (1 + (AvgAge_FlaggedReviews / 5.0))`  ← What if no flags?
- **Good**: `RECENCY_FACTOR: 1.0 / (1 + (AvgAge_FlaggedReviews / 5.0)). If no flagged reviews, use 0.5 as default.`

### 5. Complete Formula Breakdown
For complex multi-term formulas, show the full calculation step-by-step.
- **Bad**: Single-line formula with 10+ terms
- **Good**:
  ```
  FINAL_RISK_SCORE:
  PreliminaryRisk = 6.5 
                  + (VOLATILITY * 1.2) 
                  + HYGIENE_PENALTY 
                  + (REVIEW_DENSITY * 0.3)
                  ...
  FINAL_RISK_SCORE = PreliminaryRisk - (SAFETY_MARGIN * 0.1)
  ```

### 6. System-Level Instructions
Add meta-instructions in the SYSTEM_PROMPT to ensure compliance.
- **Required**: `You MUST calculate ALL values listed below. Follow formulas exactly. Do not skip any primitive.`

## Design Process (The Loop)
1.  **Draft**: Define the Logic and Prompt.
2.  **Validate**: Run on a single **Logic Trap** case (designed to fail simple reasoning).
3.  **Scale (N=10)**: Check for "Free Wins" (too easy).
4.  **Refine (Hardening)**: Apply "Assumption of Risk" or "Scoring Mask" to drive score down.
5.  **Finalize**: Run full N=50 benchmark. Target 5-15% success.
