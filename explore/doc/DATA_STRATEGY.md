# Data Strategy: Stratified Slices

**Recommendation**: Do **NOT** Randomly Sample 100 contexts from the global dataset.

## The "Excellent Researcher" Approach

A random sample of 100 from 10k introduces too many **Confounding Variables** (Region, Cuisine, Cost of Living, language norms). It makes it impossible to know if a failure is due to *reasoning* or *domain mismatch*.

### 1. The Strategy: "Anchor & Generalize"
We should define the benchmark on a **Primary Slice** first, then prove robustness on a **Transfer Slice**.

*   **Primary Slice**: `Philadelphia` + `Restaurants` (Broad).
    *   *Why City?* Controls for "Price" ($35 is cheap in NY, expensive in Ohio) and "Culture" (Tipping norms).
    *   *Why Broad Category?* "Coffee" is too narrow for tasks like "Date Night" or "Wine List". "Restaurants" covers all 10 Perspectives.

### 2. Stratified Selection (The Core 100)
Select the 100 restaurants to ensure **Variance** across key difficulties. Do not just pick the "Top 100".

We need a distribution of:
1.  **Volume**: High (1k+ reviews) vs. Low (50 reviews).
    *   *Tests*: Needle-in-Haystack limits.
2.  **Sentiment**: High (4.5+) vs. Low (2.0) vs. **Controversial** (3.5 with high variance).
    *   *Tests*: Conflict resolution and bias.
3.  **Topic Density**: Contexts rich in specific signals (e.g., "Allergy mentions", "Wait time mentions").
    *   *Tests*: Specific primitives (G1, G5).

### 3. Implementation Plan
1.  **Filter**: `City = Philadephia`, `Categories contains 'Restaurants'`.
2.  **Stratify**: Bucket into 4 quadrants (High/Low Stars x High/Low Vol). Select 25 from each.
3.  **Validate**: Ensure the "Date Night" and "Allergy" tasks have at least 10 valid targets in this set.

## Why this is Publishable
It demonstrates **Internal Validity** (Controlled Experiment) before claiming **External Validity** (Generalization).
