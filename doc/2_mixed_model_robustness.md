# Experiment 2: Mixed Model Configuration with Robustness Testing

**Date:** December 2024

**Goal:** Compare knot (planner=gpt-4o-mini, worker=gpt-5-nano) vs cot (gpt-4o-mini) on clean and adversarial inputs.

**Data:** Real Yelp restaurant reviews (3 restaurants, 5 request types each = 15 predictions)

## Configuration

| Method | Planner | Worker | Total Config |
|--------|---------|--------|--------------|
| cot | - | gpt-4o-mini | gpt-4o-mini |
| knot | gpt-4o-mini | gpt-5-nano | Mixed |

## Results

### Clean Performance

| Method | Model Config | Accuracy |
|--------|--------------|----------|
| cot | gpt-4o-mini | 53.3% |
| knot | planner=gpt-4o-mini, worker=gpt-5-nano | 46.7% |

### Adversarial Attack Results

| Attack Type | cot | knot | Winner |
|-------------|-----|------|--------|
| clean | 53.3% | 46.7% | cot |
| typo_10 | 33.3% | 46.7% | **knot** |
| typo_20 | 40.0% | 46.7% | **knot** |
| inject_override | 46.7% | 46.7% | tie |
| inject_fake_sys | 53.3% | 46.7% | cot |
| inject_hidden | 40.0% | 46.7% | **knot** |
| fake_positive | 53.3% | 46.7% | cot |
| fake_negative | 20.0% | 46.7% | **knot** |

**Summary:** knot wins 4/8, ties 1, loses 3

### Attack Types Explained

| Attack | Description |
|--------|-------------|
| typo_10 | 10% character typos in reviews |
| typo_20 | 20% character typos in reviews |
| inject_override | Injected "ignore previous instructions" |
| inject_fake_sys | Fake system message injection |
| inject_hidden | Hidden instruction injection |
| fake_positive | Injected fake positive reviews |
| fake_negative | Injected fake negative reviews |

## Key Findings

1. **cot wins on clean data** (53% vs 47%)
2. **knot is more robust** - consistent 47% across ALL attacks
3. **cot degrades under attack** - drops to 20-33% on some attacks
4. **Trade-off:** Slightly lower clean accuracy for consistent attack resistance

## gpt-5-nano Full Configuration

### Issue Discovered
When using gpt-5-nano for both planner and worker, all outputs were neutral (0).

**Root Cause:** gpt-5-nano uses ~200 "reasoning tokens" internally before generating output. Default `max_tokens=1024` was insufficient.

**Fix Applied:**
```python
# llm.py
MAX_TOKENS_REASONING = 4096  # For gpt-5-nano and o1/o3 models
```

### gpt-5-nano Approach Comparison (After Fix)

| Approach | Accuracy | Notes |
|----------|----------|-------|
| base | 60% (6/10) | Best performance |
| divide | 50% (5/10) | Comparable |
| iterative | 20% (2/10) | Underperforms - critique/revise adds noise |

## Conclusions

1. knot provides **robustness advantage** over cot under adversarial conditions
2. Trade-off: slightly lower clean accuracy for consistent attack resistance
3. gpt-5-nano works after token limit fix; base approach is sufficient
4. Complex planning approaches (iterative, divide) don't improve over base with gpt-5-nano
