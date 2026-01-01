# ANoT Iteration Log

## Round 1

### Issue
1. **Phase 1 Context Analysis returns empty** - LLM took 31s but returned nothing
2. **Phase 2 Script Generation returns empty** - Even with 30s processing time
3. **Script checks wrong conditions** - When script did generate, it checked wrong attributes

### Root Cause
1. Context analysis prompt was too verbose
2. Script generation prompt was too long and complex
3. No fallback when context analysis is empty

### Fix
1. Simplified CONTEXT_ANALYSIS_PROMPT with clear example
2. Simplified SCRIPT_GENERATION_PROMPT significantly
3. Added fallback: if context_analysis is empty, pass "(extract conditions from the user request)"
4. Added raw context to Phase 2 prompt so LLM has the user request

### Result
- Phase 1 now works! Returns proper CONDITIONS list
- Phase 2 now generates scripts (len=662)
- BUT: New issues emerged (see Round 2)

---

## Round 2

### Issue
1. **Script generation returns empty** - Even with good Phase 1 output, Phase 2 returns empty string (len=0)
2. **When script did generate**, it used wrong paths (`{(input)}[NoiseLevel]` instead of `{(input)}[attributes][NoiseLevel]`)
3. **Worker returns verbose text** - Instead of just 0/1/-1, returns "Please provide the statement..."

### Root Cause
1. The LLM (gpt-5-nano) seems to have issues generating scripts in the required format
2. Script generation prompt was too complex
3. The LWT script approach adds unnecessary complexity for simple evaluation

### Fix
Simplified the approach completely:
1. Replaced `phase2_generate_script()` with `phase2_direct_evaluate()`
2. Direct evaluation sends a single prompt with:
   - User request
   - Conditions from Phase 1
   - Actual restaurant data (attributes, hours, sample reviews)
   - Clear instruction to output -1, 0, or 1
3. Bypasses script generation entirely - just Phase 1 → Direct Evaluation → Answer

### Result
- Works! Phase 1 (20s) + Phase 2 Direct Eval (7s) = proper answer
- First request: -1 (NOT RECOMMEND) - correct because "Uncle Bobbie's" has NoiseLevel='average' (not quiet)
- Much faster and more reliable than script-based approach

---

## Round 3

### Issue
1. **Model biased toward -1** - Predicts NOT RECOMMEND 14/20 times vs gold of 12/20
2. **Rarely predicts 1** - Only 2/20 predictions are RECOMMEND
3. **60% accuracy** - 12/20 correct on 1 item × 20 requests
4. **Speed** - ~27s per request (20s Phase 1 + 7s Phase 2) = too slow for practical use

### Root Cause
1. The direct evaluation prompt may be too conservative - tends to find reasons to reject
2. "If any required condition is clearly NOT met, output -1" may be too strict
3. Phase 1 context analysis takes 20+ seconds - adds latency for each request

### Fix for Round 4
1. Adjust evaluation prompt to be more balanced
2. Add explicit guidance for 0 (unclear) vs -1 (definitely not)
3. Consider caching Phase 1 results per context (same context → same conditions)

### Result
Baseline established:
- Accuracy: 60% (12/20)
- Pred distribution: {-1: 14, 0: 4, 1: 2}
- Gold distribution: {-1: 12, 0: 6, 1: 2}
- Errors mostly: false negatives (-1 when should be 0 or 1)

---

## Round 4

### Issue
[To observe after fix]

### Root Cause
[TBD]

### Fix
[TBD]

### Result
[TBD]
