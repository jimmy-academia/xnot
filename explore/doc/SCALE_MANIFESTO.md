# SCALE: Structured Context Analysis & Logic Evaluation
## A Benchmark for LLM Reasoning Limits

**Objective**: Systematically identify the reasoning boundaries of LLMs by scaling context length and task complexity across diverse domains (Yelp, Amazon, etc.).

## Core Terminology

| Term | Definition | Scoring |
| :--- | :--- | :--- |
| **Context** | The "World State" for the task. A structured object (Restaurant or Product) + its collection of N Reviews. | N/A |
| **Query** | The natural language question posed to the model. Defines the objective. | N/A |
| **Evidence** | The specific, verifiable facts or calculations extracted from the Context to support the decision. (Replaces "Derivations") | 0.0 - 1.0 (Distance-based) |
| **Verdict** | The final binary decision (Yes/No, Option A/B) derived *strictly* from the Evidence. | 0 or 1 (Binary) |

### Why "Evidence"?
It implies a courtroom-style logic: "You cannot reach a Verdict without Evidence." It is intuitive, supports both qualitative strings ("The review mentions 'rust'") and quantitative stats ("Average rating is 3.5"), and fits the "Audit" use case.

## Scoring Formula

For a single task instance $i$:
$$ Score_i = Verdict\_Correct_i \times \left( \frac{1}{M} \sum_{j=1}^{M} Evidence\_Score_{i,j} \right) $$

*   **The "Gatekeeper"**: If `Verdict` is wrong (0), the total score is 0. No partial credit for lucky guesses or wrong conclusions.
*   **The "Process"**: If `Verdict` is right (1), the score is scaled by the quality of the `Evidence`.
    *   *Example*: Verdict is Correct (1). Expecting "34 verified reviews", model found "30". Evidence score might be 0.88. Final Score = 0.88.

## The 10x10 Task Matrix (Draft)

We target **100 Tasks** organized into **10 Reasoning Primitives**.

| Group | Name | Primitive | Example Query |
| :--- | :--- | :--- | :--- |
| **G01** | **Extraction** | `Search` | "Does this product explicitly support 220V voltage?" |
| **G02** | **Quantitative** | `Agg` | "Is the median rating for 'durability' higher than 4.0?" |
| **G03** | **Temporal** | `Time` | "Has sentiment improved since the 2023 recall?" |
| **G04** | **Entity** | `Cluster` | "Is 'Server Mike' mentioned more positively than 'Manager'?" |
| **G05** | **Conditional** | `Filter` | "Considering only 'Verified Purchases', is the size accurate?" |
| **G06** | **Correlation** | `Link` | "Do users who complain about 'Price' also complain about 'Portion'?" |
| **G07** | **Persona** | `Segment` | "Would a 'Vegetarian' user be satisfied based on menu reviews?" |
| **G08** | **Outlier** | `Anomaly` | "Is the one 1-star review an outlier compared to concurrent reviews?" |
| **G09** | **Contradiction** | `Conflict` | "Does the user review contradict the product description dimensions?" |
| **G10** | **Decision** | `Gate` | "Given a $50 budget and need for 'Fast Shipping', should I buy?" |

## Scaling Dimensions
1.  **Context Size (N)**: 10, 50, 100, 250, 500 reviews. (Testing "Lost in the Middle").
2.  **Domain**: Restaurants (Initial), Products (Expansion), Movies/Books (Future).
3.  **Noise**: Clean Data vs. Injected Noise (Typos, Irrelevant Info).

## Implementation Plan
1.  **Refactor**: Rename `explore/` to `scale/`.
2.  **Base Class**: `ScaleTask` which enforces `compute_evidence()` and `compute_verdict()`.
3.  **Data Loader**: `ScaleContextLoader` abstracting Yelp/Amazon differences.
