# SCALE Benchmark Dimensions

The SCALE benchmark is defined by the following dimensions.

## Target Dimensions

| Symbol | Dimension | Count / Description |
| :--- | :--- | :--- |
| **K** | **Contexts** | **100** Restaurants (Context Units). Each unit is a self-contained set of reviews, item_meta, review_meta, and reviewer_meta. |
| **N** | **Reviews** | **Scalable**. Varies per context (e.g., 50, 100, 500, 1000) to test "Needle-in-Haystack" limits. |
| **Q** | **Queries** | **30 ~ 100**. The distinct questions posed (from the 100 defined tasks). |
| **P** | **Premises** | **Variable**. The intermediate facts required to answer Q. Designed per query. |

## Scoring Hierarchy

We define performance at three levels of aggregation to isolate failure modes.

### 1. The Instance Score ($S_{q,k}$)
*Scope: 1 Query ($q$) on 1 Restaurant ($k$).*

The score for a single execution. It punishes "Right Answer, Wrong Reason" (Hallucination).

$$ S_{q,k} = Verdict_{q,k} \times \underbrace{\left( \frac{1}{|P_q|} \sum_{p \in P_q} Grounding_{p,k} \right)}_{\text{Evidence Quality}} $$

*   **Verdict**: Binary (0/1). Is the final decision correct?
*   **Grounding**: Binary (0/1). Is the specific premise $p$ correctly retrieved/calculated for this restaurant?

### 2. Task Reliability ($\mu_q$)
*Scope: 1 Query ($q$) averaged over all $K$ Restaurants.*

Measures how **robust** the model is at a specific logic type (e.g., "Can it *consistently* detect peanut safety?").

$$ \mu_q = \frac{1}{K} \sum_{k=1}^{K} S_{q,k} $$

### 3. Group Competency ($C_g$)
*Scope: All Queries in one Group ($g$) (e.g., "Health & Safety").*

Measures the model's proficiency in a specific **domain**.

$$ C_g = \frac{1}{|Q_g|} \sum_{q \in Q_g} \mu_q $$

### 4. The SCALE Score
*Scope: The Global Average.*

$$ SCALE = \frac{1}{10} \sum_{g=1}^{10} C_g $$
