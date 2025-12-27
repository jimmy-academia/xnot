# Experiment 6: Complex Requests - CoT vs KNoT

**Date:** December 27, 2024

**Goal:** Compare CoT and KNoT methods on complex requests (C0-C7) with AND/OR logic, testing both normal and defense prompts.

**Data:** 5 restaurants, 8 complex requests = 40 predictions per attack type

## Configuration

| Setting | Value |
|---------|-------|
| Model | gpt-5-nano |
| Data | data/processed/complex_data.jsonl |
| Requests | data/requests/complex_requests.json (C0-C7) |
| Limit | 5 restaurants |
| Runs | 18_cot_complex_normal, 19_cot_complex_defense, 20_knot_complex_normal, 24_knot_complex_defense |

## Results - Combined Comparison

| Attack | CoT Normal | CoT Defense | KNoT Normal | KNoT Defense | Best |
|--------|------------|-------------|-------------|--------------|------|
| clean | 30.0% (12/40) | 27.5% (11/40) | 30.0% (12/40) | 30.0% (12/40) | tie |
| typo_10 | 17.5% (7/40) | 25.0% (10/40) | **30.0%** (12/40) | 27.5% (11/40) | KNoT Normal |
| typo_20 | 22.5% (9/40) | **32.5%** (13/40) | 25.0% (10/40) | 17.5% (7/40) | CoT Defense |
| inject_override | 22.5% (9/40) | 20.0% (8/40) | 20.0% (8/40) | 17.5% (7/40) | CoT Normal |
| inject_fake_sys | 20.0% (8/40) | 25.0% (10/40) | 17.5% (7/40) | **35.0%** (14/40) | KNoT Defense |
| inject_hidden | 17.5% (7/40) | 27.5% (11/40) | **35.0%** (14/40) | 25.0% (10/40) | KNoT Normal |
| inject_manipulation | 27.5% (11/40) | 22.5% (9/40) | 37.5% (15/40) | **47.5%** (19/40) | KNoT Defense |
| fake_positive | 25.0% (10/40) | 20.0% (8/40) | 22.5% (9/40) | 22.5% (9/40) | CoT Normal |
| fake_negative | 30.0% (12/40) | 25.0% (10/40) | 35.0% (14/40) | **45.0%** (18/40) | KNoT Defense |
| **Average** | 23.6% (85/360) | 25.0% (90/360) | 28.1% (101/360) | **29.7%** (107/360) | KNoT Defense |

## Key Findings

### 1. KNoT Outperforms CoT on Complex Requests

Overall accuracy:
- **KNoT Defense: 29.7%** (best)
- KNoT Normal: 28.1%
- CoT Defense: 25.0%
- CoT Normal: 23.6%

KNoT's structured multi-step approach provides ~5-6pp improvement over CoT on complex AND/OR logic requests.

### 2. Defense Prompt Effect Varies by Method

**For KNoT:** Defense helps on injection attacks
- inject_fake_sys: +17.5pp (17.5% -> 35.0%)
- inject_manipulation: +10pp (37.5% -> 47.5%)
- fake_negative: +10pp (35.0% -> 45.0%)

**For CoT:** Defense helps on typos
- typo_10: +7.5pp (17.5% -> 25.0%)
- typo_20: +10pp (22.5% -> 32.5%)
- inject_hidden: +10pp (17.5% -> 27.5%)

### 3. Method-Attack Specialization

| Attack Type | Best Method | Advantage |
|-------------|-------------|-----------|
| typo attacks | Mixed | CoT Defense wins typo_20, KNoT Normal wins typo_10 |
| injection attacks | KNoT Defense | Strong advantage on fake_sys, manipulation |
| fake reviews | KNoT Defense | 45% on fake_negative (vs 25-35% others) |
| clean baseline | Tie | All methods ~30% |

### 4. Complex Requests Are Hard

All methods struggle with complex AND/OR logic:
- Best average: 29.7% (KNoT Defense)
- Random baseline would be 33% (for 3-class classification)
- Complex requests (C0-C7) are significantly harder than simple requests (R0-R4)

## Confusion Matrix Patterns

Common pattern across all methods:
- Heavy bias toward predicting +1 (recommend)
- Gold label -1 often mispredicted as +1
- Neutral (0) predictions are rare

This suggests models are too optimistic in their recommendations.

## Conclusions

1. **KNoT is better for complex logic** - The structured decomposition helps with AND/OR conditions
2. **Defense prompts work differently by method** - KNoT defense helps on injections, CoT defense helps on typos
3. **No single method dominates all attacks** - Attack-specific strategies may be needed
4. **Complex requests need better approaches** - All methods underperform vs random baseline
