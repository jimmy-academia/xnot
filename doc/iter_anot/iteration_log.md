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
1. **Ranking mode not supported** - `anot.py` only had per-item evaluation (`solve()`)
2. **Excluded from ranking** - `run.py:407` explicitly disabled ranking for anot: `args.method != "anot"`
3. **Wrong query format** - `eval_mode` was set to "dict" for anot, but ranking needs "string"

### Root Cause
1. Original design used per-item evaluation (-1/0/1 scores)
2. Ranking mode requires comparing ALL restaurants at once and outputting best index
3. `format_ranking_query()` needs "string" mode to format restaurants with [N] markers

### Fix
1. **Added ranking prompts to `anot.py`**:
   - `RANKING_CONTEXT_ANALYSIS_PROMPT` - Identifies comparison criteria
   - `DIRECT_RANKING_PROMPT` - Direct comparison prompt

2. **Added ranking methods to `AdaptiveNetworkOfThought` class**:
   - `phase1_analyze_context_ranking()` - Analyze user request for ranking (cached)
   - `phase2_direct_ranking()` - Direct evaluation approach
   - `_parse_ranking_indices()` - Extract indices from LLM output
   - `evaluate_ranking(query, context, k)` - Main entry point, returns index string

3. **Updated `methods/__init__.py`** (lines 45-53):
   ```python
   elif name == "anot":
       anot_instance = create_anot(run_dir=run_dir, debug=debug)
       if getattr(args, 'ranking', True):
           k = getattr(args, 'k', 1)
           return lambda q, c: anot_instance.evaluate_ranking(q, c, k)
       return lambda q, c: anot_instance.solve(q, c)
   ```

4. **Updated `run.py`**:
   - Removed `args.method != "anot"` from ranking_mode check (line 407)
   - Changed eval_mode logic: ranking always uses "string" format (line 406)

### Result
- Ranking mode works! "ANoT RANKING" banner displays
- Phase 1 analyzes context for ranking criteria (~20-30s)
- Phase 2 performs direct ranking comparison (~3-5s)
- Outputs restaurant index (1, 2, 3...) instead of -1/0/1
- Hits@K metrics computed properly
