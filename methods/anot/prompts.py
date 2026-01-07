#!/usr/bin/env python3
"""Prompt constants for ANoT phases - Multi-Step Design."""

from prompts.task_descriptions import RANKING_TASK_COMPACT

# Re-export for use in core.py
__all__ = [
    'SYSTEM_PROMPT', 'STEP1_EXTRACT_PROMPT', 'STEP2_PATH_PROMPT',
    'STEP3_RULEOUT_PROMPT', 'STEP4_SKELETON_PROMPT', 'PHASE2_PROMPT',
    'RANKING_TASK_COMPACT'
]

SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested."

# =============================================================================
# STEP 1: Condition Extraction
# =============================================================================

STEP1_EXTRACT_PROMPT = """Extract conditions from the user request.

[USER REQUEST]
{query}

[OUTPUT FORMAT]
List each condition on a new line:
[ATTR] description of attribute condition
[REVIEW] description of review text search
[HOURS] description of hours condition

Example output:
[ATTR] has drive-thru
[ATTR] kid-friendly
[REVIEW] mentions wifi
"""

# =============================================================================
# STEP 2: Path Resolution (called per condition)
# =============================================================================

STEP2_PATH_PROMPT = """Determine where to find the value for this condition.

[CONDITION]
{condition_description}

[SCHEMA - example items showing available fields]
{schema_compact}

[COMMON FIELDS]
- attributes.GoodForKids: True/False
- attributes.WiFi: "free", "paid", "no", or None
- attributes.DriveThru: True/False
- attributes.NoiseLevel: "quiet", "average", "loud", "very_loud"
- attributes.OutdoorSeating: True/False
- attributes.HasTV: True/False
- attributes.Ambience: dict with keys like hipster, casual, upscale, romantic, etc.
- attributes.GoodForMeal: dict with keys like breakfast, lunch, dinner, brunch, etc.
- hours: dict with day names as keys, values like "8:0-22:0"
- reviews: list of review objects with 'text' field

[TASK]
1. Identify which field to check for this condition
2. If field doesn't exist in schema, use best guess from common fields
3. Determine expected value

[OUTPUT FORMAT]
PATH: attributes.FieldName
EXPECTED: True/False/"value"
TYPE: HARD

Or if it's a review text search:
PATH: reviews
EXPECTED: keyword
TYPE: SOFT
"""

# =============================================================================
# STEP 3: Quick Rule-Out (check hard conditions, prune items)
# =============================================================================

STEP3_RULEOUT_PROMPT = """Check items against hard conditions and identify which pass.

[HARD CONDITIONS]
{hard_conditions}

[ITEMS - relevant attributes only]
{items_compact}

[TASK]
For each item, check if ALL hard conditions are satisfied.
- Item passes if all conditions match
- Item fails if any condition doesn't match
- Missing/None values count as not matching

[OUTPUT FORMAT]
===CANDIDATES===
[list of item numbers that pass all conditions, e.g., 2, 4, 6, 7]

===PRUNED===
item_number: reason
item_number: reason
...
"""

# =============================================================================
# STEP 4: LWT Skeleton Generation (separate steps per item)
# =============================================================================

STEP4_SKELETON_PROMPT = """Generate LWT skeleton for soft conditions on candidate items.

[CANDIDATES]
{candidates}

[SOFT CONDITIONS]
{soft_conditions}

[RULES]
- Generate ONE step per item per soft condition
- Each step checks ONE item independently using {{{{(context)}}}}[item_num][reviews]
- Final step aggregates all results and outputs ranking
- Use variable names like (r2), (r4) for item-specific results

[PATH SYNTAX]
{{{{(context)}}}}[2][reviews] - Item 2's reviews array

[OUTPUT FORMAT]
===LWT_SKELETON===
(r2)=LLM('Item 2 reviews: {{{{(context)}}}}[2][reviews]. {soft_question} Answer: yes/no')
(r4)=LLM('Item 4 reviews: {{{{(context)}}}}[4][reviews]. {soft_question} Answer: yes/no')
...
(final)=LLM('Results: r2={{{{(r2)}}}}, r4={{{{(r4)}}}}, ... Rank items with yes first. Output top-{k}: [best,2nd,...]')

[EXAMPLE for candidates [2,4,6] checking "mentions wifi"]
===LWT_SKELETON===
(r2)=LLM('Item 2 reviews: {{{{(context)}}}}[2][reviews]. Mentions wifi? Answer: yes/no')
(r4)=LLM('Item 4 reviews: {{{{(context)}}}}[4][reviews]. Mentions wifi? Answer: yes/no')
(r6)=LLM('Item 6 reviews: {{{{(context)}}}}[6][reviews]. Mentions wifi? Answer: yes/no')
(final)=LLM('wifi: r2={{{{(r2)}}}}, r4={{{{(r4)}}}}, r6={{{{(r6)}}}}. Rank items with yes first. Output: [best,2nd,3rd,4th,5th]')

[IF NO SOFT CONDITIONS]
===LWT_SKELETON===
(final)=LLM('Candidates: {candidates}. All passed hard conditions. Output top-{k}: [first {k} from list]')
"""

# =============================================================================
# PHASE 2: ReAct Expansion (refine skeleton with read() calls)
# =============================================================================

PHASE2_PROMPT = """Refine the LWT skeleton by checking review content.

[LWT SKELETON]
{lwt_skeleton}

[AVAILABLE TOOLS]
- read("items[2].reviews") - Read item 2's reviews to check if they match
- lwt_set(idx, step) - Modify step at index
- lwt_delete(idx) - Remove step at index (if item clearly doesn't match)
- done() - Finish refinement

[TASK]
1. Optionally read() reviews for items to verify they match soft conditions
2. If an item clearly doesn't match, use lwt_delete() to remove its step
3. Call done() when finished

[PROCESS]
- You can read a few items to verify matching
- Delete steps for items that clearly don't match
- Keep steps for items that might match (let Phase 3 evaluate)
- Call done() to finish
"""
