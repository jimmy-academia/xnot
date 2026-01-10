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

## References

1. **Waegeman, W., De Baets, B., & Boullart, L. (2008).** ROC analysis in ordinal regression learning. *Pattern Recognition Letters*, 29(1), 1-9.

2. **Baccianella, S., Esuli, A., & Sebastiani, F. (2009).** Evaluation Measures for Ordinal Regression. *IEEE International Conference on Intelligent Systems Design and Applications (ISDA)*, 283-287.

3. **Brentnall, A. R., & Cuzick, J. (2019).** Cumulative ROC curves for discriminating three or more ordinal outcomes with cutpoints on a shared continuous measurement scale. *PLoS ONE*, 14(8): e0221433.
