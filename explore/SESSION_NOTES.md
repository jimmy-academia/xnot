# G1a ANoT Session Summary (2026-01-10)

## What Was Built

### Files Created in `explore/g1_anot/`
```
g1_anot/
├── __init__.py      # Package exports
├── core.py          # G1aANoT class - hardcoded extraction + Python computation
├── prompts.py       # Extraction prompts for allergy signals
├── tools.py         # keyword_search, get_cuisine_modifier, compute_risk_score
└── helpers.py       # LWT parsing, DAG building (not fully used)
```

### Evaluation Drivers
- `g1_anot_eval.py` - Runs G1a-ANoT evaluation
- `cot_eval.py` - Chain of Thought baseline

## Benchmark Results (100 restaurants, K=200)

| Method | Ordinal AUPRC | Primitive Acc | Adjusted AUPRC |
|--------|---------------|---------------|----------------|
| **Direct LLM** | 0.844 | 0.892 | **0.753** |
| **G1a-ANoT (hardcoded)** | 0.570 | 0.875 | **0.499** |
| Chain of Thought | ~0.78 | 0.783 | ~0.61 |

## Key Findings

### 1. Direct LLM outperformed hardcoded ANoT
- Direct LLM: 0.753 Adjusted AUPRC
- Hardcoded ANoT: 0.499 Adjusted AUPRC
- **Unexpected!** The hardcoded approach was worse

### 2. CoT doesn't help (and may hurt)
- CoT primitive accuracy (0.783) < Direct LLM (0.892)
- Explicit step-by-step prompting doesn't improve formula-following

### 3. Small sample results were misleading
- 30 samples: ANoT showed 1.0 AUPRC (all Low Risk, 1 Critical)
- 100 samples: ANoT dropped to 0.499 (more diverse distribution)

## Critical Insight (User Feedback)

**The hardcoded G1a-ANoT is NOT a true universal ANoT.**

Current implementation:
- Prompts are task-specific (allergy keywords, extraction format)
- Tools are G1a-specific (compute_risk_score uses V1 formula)
- Not generalizable to other 99 tasks

**What universal ANoT needs:**
- Phase 1: Parse ANY formula specification → generate LWT seed dynamically
- Phase 2: Expand seed with context-specific data paths
- Phase 3: Execute with COMPUTE() for arithmetic (avoiding LLM math errors)

## Next Steps

1. **Analyze why Direct LLM beat hardcoded ANoT**
   - Check if LLM extraction is failing on some reviews
   - Compare primitive-level accuracy per restaurant

2. **Design universal ANoT**
   - Formula parser that generates extraction prompts from query
   - Dynamic LWT seed generation
   - COMPUTE() for all arithmetic

3. **Test on more tasks**
   - G1a-v2 (harder formula)
   - Other G-series tasks

## Run Commands

```bash
# Direct LLM baseline
python eval.py --task G1a --limit 20

# G1a-ANoT (hardcoded)
python g1_anot_eval.py --limit 20

# Chain of Thought
python cot_eval.py --limit 20

# Compare methods
python g1_anot_eval.py --compare --limit 20
```

## Results Location
- `results/G1a_k200_*/` - Direct LLM results
- `results/G1a_anot_k200_*/` - ANoT results
- `results/G1a_cot_k200_*/` - CoT results
