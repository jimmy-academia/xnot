# Cumulative Ordinal AUPRC

Evaluation metric for ordinal classification tasks with ordered severity levels.

## Problem Setting

For G1.1 (Peanut Allergy Safety), we have three ordinal classes:

```
Low Risk < High Risk < Critical Risk
```

Standard multi-class metrics (accuracy, macro-F1) don't respect this ordering. Misclassifying Critical as Low is worse than misclassifying Critical as High.

## Cumulative Ordinal Approach

Instead of one-vs-rest, we use **cumulative probability thresholds**:

| Threshold | Binary Task | Question |
|-----------|-------------|----------|
| k=1 | (High+Critical) vs Low | "Is it at least High Risk?" |
| k=2 | Critical vs (High+Low) | "Is it Critical Risk?" |

For each threshold, we compute AUPRC (Area Under Precision-Recall Curve).

## Metrics

### Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `auprc_ge_1` | AUPRC(Y≥High) | Can model separate risky from safe? |
| `auprc_ge_2` | AUPRC(Y≥Critical) | Can model identify critical cases? |
| `ordinal_auprc` | mean(auprc_ge_1, auprc_ge_2) | Combined ordinal performance |

### Normalized Metrics

Chance-corrected versions account for class imbalance:

```
nAP = (AP - prevalence) / (1 - prevalence)
```

| Metric | Interpretation |
|--------|----------------|
| `nap_ge_1` | Improvement over random for ≥High |
| `nap_ge_2` | Improvement over random for ≥Critical |
| `ordinal_nap` | Combined chance-corrected performance |

### Auxiliary Metrics

| Metric | Description |
|--------|-------------|
| `severity_ordering_ap` | Among risky (High+Critical), is Critical ranked higher? |

## Expected Values

With GT distribution (78 Low, 21 High, 1 Critical):

| Scenario | ordinal_auprc | ordinal_nap |
|----------|---------------|-------------|
| Perfect predictions | 1.000 | 1.000 |
| Random predictions | ~0.11 | ~0.00 |
| Inverted predictions | ~0.07 | negative |

## Usage

```bash
# With results file
python score_auprc.py results.json

# Demo with synthetic data
python score_auprc.py --demo
```

## Why AUPRC over AUROC?

AUPRC is preferred for imbalanced datasets (our Critical class is only 1%). AUROC can be misleadingly high when the positive class is rare.

---

## Adjusted AUPRC: Process-Reward Scoring

### Motivation

Standard Ordinal AUPRC only evaluates **outcome correctness** - whether the model's final risk score correctly ranks restaurants by their true risk level. However, a model can achieve high AUPRC through:

1. **Correct reasoning**: Understanding the data and applying formulas correctly
2. **Lucky guessing**: Reaching correct conclusions through flawed intermediate steps

For reliable deployment, we need models that reason correctly, not just guess correctly. The **Adjusted AUPRC** metric penalizes models that produce correct outcomes from incorrect reasoning.

### Formula

```
Adjusted_AUPRC = Ordinal_AUPRC × Avg_Primitive_Accuracy
```

Where:
- **Ordinal_AUPRC**: Standard cumulative ordinal AUPRC (ranking quality)
- **Avg_Primitive_Accuracy**: Mean accuracy of intermediate calculations across all samples

```
Avg_Primitive_Accuracy = (1/N) × Σᵢ accuracy_i

accuracy_i = (1/K) × Σₖ 𝟙[|pred_k - gt_k| ≤ τ_k]
```

Where:
- N = number of restaurants
- K = number of scored primitives (intermediate values)
- τ_k = tolerance for primitive k

### Interpretation

| Scenario | Ordinal AUPRC | Prim. Accuracy | Adjusted AUPRC |
|----------|---------------|----------------|----------------|
| Perfect reasoning | 1.0 | 1.0 | 1.0 |
| Correct ranking, wrong steps | 0.8 | 0.5 | 0.4 |
| Wrong ranking, correct steps | 0.3 | 0.9 | 0.27 |
| Random everything | 0.11 | 0.2 | 0.02 |

The multiplicative formulation ensures that **both** ranking quality AND reasoning quality must be high to achieve a high score.

### Why Multiplicative (Not Additive)?

An additive formulation (e.g., `0.5 × AUPRC + 0.5 × Accuracy`) would allow a model with perfect primitive accuracy but random ranking to score 0.5, which is misleadingly high.

The multiplicative formulation treats the two factors as **joint requirements**:
- High AUPRC alone is insufficient (could be lucky)
- High primitive accuracy alone is insufficient (could be wrong outcome)

### Connection to Process Reward Models

This approach is grounded in the **process supervision** paradigm from AI safety research, which advocates rewarding correct reasoning steps rather than just correct final answers.

**Key insight from Lightman et al. (2023)**: Outcome supervision can reward incorrect reasoning that happens to reach the correct answer, leading to unreliable models. Process supervision directly trains models to follow endorsed reasoning patterns.

Our Adjusted AUPRC applies this principle to **evaluation** rather than training:
- Ordinal AUPRC = Outcome Reward (final ranking quality)
- Primitive Accuracy = Process Reward (intermediate step correctness)
- Adjusted AUPRC = Combined signal that requires both

### Primitives Scored

For G1a (Peanut Allergy Safety), the scored primitives are:

| Primitive | Tolerance | Type |
|-----------|-----------|------|
| `n_total_incidents` | 0 | Exact count |
| `incident_score` | 0 | Exact calculation |
| `recency_decay` | 0.1 | Float |
| `credibility_factor` | 0.2 | Float |

Note: `final_risk_score` and `verdict` are NOT included in primitive accuracy as they are captured by AUPRC.

### Usage

```bash
# Score results file (automatically computes Adjusted AUPRC)
python score_auprc.py results/latest/detailed_results.json
```

Output includes:
```
PROCESS-REWARD ADJUSTED METRICS:
--------------------------------------------------
  Primitive Acc:  0.873  (mean accuracy of intermediate calculations)
    - Range:      [0.250, 1.000]
    - Std Dev:    0.189

  Adjusted AUPRC: 0.449  = 0.514 × 0.873
```

---

## References

### Ordinal Classification

1. **Waegeman, W., De Baets, B., & Boullart, L. (2008).** ROC analysis in ordinal regression learning. *Pattern Recognition Letters*, 29(1), 1-9.

2. **Baccianella, S., Esuli, A., & Sebastiani, F. (2009).** Evaluation Measures for Ordinal Regression. *IEEE International Conference on Intelligent Systems Design and Applications (ISDA)*, 283-287.

3. **Brentnall, A. R., & Cuzick, J. (2019).** Cumulative ROC curves for discriminating three or more ordinal outcomes with cutpoints on a shared continuous measurement scale. *PLoS ONE*, 14(8): e0221433.

### Process Supervision and Reward Models

4. **Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023).** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). *arXiv preprint arXiv:2305.20050*.
   - Key finding: Process supervision significantly outperforms outcome supervision for training reliable models on math reasoning.
   - Relevance: Justifies evaluating intermediate reasoning steps, not just final answers.
   - OpenAI blog post: [Improving Mathematical Reasoning with Process Supervision](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)

5. **Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021).** [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). *arXiv preprint arXiv:2110.14168*.
   - Introduced GSM8K dataset and the verifier approach.
   - Key insight: Training verifiers to judge solution correctness improves performance more than scaling model size.

6. **Lanham, T., et al. (2023).** [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702). *arXiv preprint arXiv:2307.13702*.
   - Investigates whether stated reasoning faithfully represents actual model reasoning.
   - Key finding: Models often produce unfaithful reasoning that doesn't match their decision process.
   - Relevance: Motivates measuring intermediate step correctness to detect unfaithful reasoning.

7. **Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., & Sui, Z. (2024).** [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://arxiv.org/abs/2312.08935). *ACL 2024*.
   - Automatic process supervision without human labels.
   - Demonstrates scalable approaches to step-level verification.
