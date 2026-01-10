# LLM Reasoning Limit Discovery: Leaderboard & Analysis

This document tracks the "breaking point" of LLM reasoning as we scale task complexity and context length (number of reviews).

## Task Performance Summary (n=50 reviews)

| Task | Category | Avg Score | Status | LLM Failure Mode |
|------|----------|-----------|--------|-----------------|
| **A** | Recent vs Historical | ~0.98 | **SOLVED** | Deprecated. |
| **B** | Credibility Weighting | ~0.95 | **SOLVED** | Deprecated. |
| **C** | Rating-Text Alignment | ~0.55 | **FAILING** | **Attention/Retrieval**: Missing misaligned reviews in dense context. |
| **D** | Cross-Aspect Correlation| ~0.58 | **FAILING** | **Information Extraction**: Failing to link "Food" issues to "Wait" issues correctly. |
| **F** | Expectation Calibration | ~0.75 | **STRUGGLING**| **Logic/Formatting**: Errors in "Price Adjusted Score" calculation. |
| **G** | Conditional Sentiment | ~0.30 | **FAILING** | **Nuance/Resolution**: Struggling with "Worth the Wait" vs "Not Worth It". |
| **H** | Entity Resolution | ~0.20 | **FAILING** | **Entity Linking**: Hallucinating or missing staff names across reviews. |
| **I** | Multi-Hop Reasoning | ~0.65 | **FAILING** | **Cross-Reference**: Inconsistent entity linking across multiple reviews. |
| **J** | Weighted Aspect Priority| TBD | **TESTING** | **Arithmetic/Aspect**: Expected to fail on multi-step weighting. |
| **K** | Entity Sentiment Drift | TBD | **TESTING** | **Temporal/Entity**: Expected to fail on linking staff to dates. |

## Detailed Analysis of Failure Points

### Task C: Retrieval under pressure
With 50+ reviews, LLMs consistently under-count "misaligned" reviews (where text is positive but stars are low). This indicates a "Lost in the Middle" or attention bottleneck when scanning for specific negative sentiment triggers.

### Task G: Conditional Logic
The model struggles to differentiate between a review that *mentions* a wait and one where the wait *defined* the experience.

### Task H: Entity Resolution
The model often "invents" staff names or misses them if they aren't preceded by specific keywords like "server". This is a high-level entity resolution task.

## Next Steps: Harder Tasks (G-L)
We are rotating to more complex temporal and multi-hop reasoning tasks.

- **Task I (Temporal Sequencing)**: Sequence of events in service failure.
- **Task J (Quantitative Multi-Hop)**: Aggregating cross-user metadata.
