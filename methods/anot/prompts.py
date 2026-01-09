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

[RULES]
- ONLY extract conditions the user EXPLICITLY wants
- If user doesn't mention hours/timing, do NOT output any [HOURS] line
- Distinguish between review types:
  - "is praised for X", "known for great X" → [REVIEW:POSITIVE] X (sentiment check)
  - "complaints about X", "criticized for X" → [REVIEW:NEGATIVE] X (sentiment check)
  - "has reviews mentioning X", "reviews mention X" → [REVIEW:MENTION] X (keyword check)
- For attribute requirements like "quiet", "trendy", "hipster vibe", use [ATTR] not [REVIEW]
- For social/friend filters like "my friend X reviewed", use [SOCIAL] friend_name, keyword
- If a category isn't mentioned, simply omit it - do NOT write "not specified" or "none"

[OUTPUT FORMAT]
List each condition on a new line:
[ATTR] description of attribute condition
[REVIEW:POSITIVE] topic (only for praised/recommended aspects)
[REVIEW:NEGATIVE] topic (only for complaints/criticisms)
[REVIEW:MENTION] keyword (only for "reviews mention X" / "has reviews mentioning X")
[HOURS] day and time range (only if user explicitly mentions hours)
[SOCIAL] friend_name, keyword (only if user mentions friend's review)

Example 1 - attribute query:
User: "Looking for a quiet cafe with free WiFi"
[ATTR] quiet
[ATTR] free WiFi

Example 2 - positive sentiment review query:
User: "Looking for a cafe praised for their coffee"
[REVIEW:POSITIVE] coffee

Example 3 - keyword mention query:
User: "Looking for a cafe with reviews mentioning 'cozy'"
[REVIEW:MENTION] cozy

Example 4 - hours query:
User: "Looking for a cafe open on Sunday morning"
[HOURS] Sunday morning

Example 5 - social filter query:
User: "Looking for a cafe that my friend Kevin reviewed mentioning 'place'"
[SOCIAL] Kevin, place
"""

# =============================================================================
# STEP 2: Path Resolution (called per condition)
# =============================================================================

STEP2_PATH_PROMPT = """Map the condition to a database field.

CONDITION: {condition_description}

AVAILABLE FIELDS:
- attributes.NoiseLevel: "quiet", "average", "loud" (for quiet/noisy)
- attributes.GoodForKids: True/False (for kid-friendly)
- attributes.DogsAllowed: True/False (for dog-friendly)
- attributes.WiFi: "free", "paid", "no" (for WiFi)
- attributes.HasTV: True/False (for TV/screens)
- attributes.OutdoorSeating: True/False (for outdoor/patio)
- attributes.CoatCheck: True/False (for coat storage)
- attributes.DriveThru: True/False (for drive-through)
- attributes.Alcohol: "full_bar", "beer_and_wine", "none"
- attributes.RestaurantsTakeOut: True/False (for takeout)
- attributes.RestaurantsReservations: True/False (for reservations)
- attributes.RestaurantsGoodForGroups: True/False (for groups)
- attributes.BikeParking: True/False (for bike parking)
- attributes.Ambience.hipster: True/False (for hipster/indie vibe)
- attributes.Ambience.trendy: True/False (for trendy/Instagram)
- attributes.Ambience.casual: True/False
- attributes.Ambience.romantic: True/False
- attributes.RestaurantsPriceRange2: 1-4 (1=cheap, 4=expensive)
- hours.Monday, hours.Tuesday, etc: "8:0-22:0" format
- reviews: search review text for keywords

OUTPUT exactly 3 lines for the condition "{condition_description}":
PATH: [field name from list above]
EXPECTED: [True/False/value the user wants]
TYPE: [HARD or SOFT]

Use HARD for exact matches, SOFT for reviews/hours.
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

STEP4_SKELETON_PROMPT = """Generate LWT to rank candidates {candidates} for: {soft_question}

CRITICAL RULES:
1. NEVER write {{(context)}} or {{(items)}} alone - this dumps 100K+ tokens!
2. ALWAYS use specific paths: {{(context)}}[N][field] where N is item number
3. For names: {{(context)}}[N][name]
4. For reviews: {{(context)}}[N][reviews][0][text][0:200] (slice to 200 chars max!)
5. For stars: {{(context)}}[N][stars]

BAD (crashes): {{(context)}}
BAD (crashes): {{(items)}}
GOOD: {{(context)}}[1][name], {{(context)}}[1][stars]
GOOD: {{(context)}}[2][reviews][0][text][0:150]

===LWT_SKELETON===
(final)=LLM('Candidates: {candidates}. Names: {{(context)}}[{first_candidate}][name]. Rank by {soft_question}. Output numbers: ')"""

# =============================================================================
# PHASE 2: ReAct Expansion (refine skeleton with slice syntax for long reviews)
# =============================================================================

PHASE2_PROMPT = """Review LWT and fix any issues, then call done().

[SKELETON]
{lwt_skeleton}

[CHECK]
1. If you see {{(context)}} or {{(items)}} WITHOUT [N][field] accessor → FIX IT
2. If reviews are accessed, ensure slice like [0:200] is present
3. If OK, output: done()

[FIX TOOL]
update_step("step_id", "new prompt text")

Example fix:
BAD:  (final)=LLM('Rank {{(context)}}...')
FIX:  update_step("final", "Rank items by name: {{(context)}}[1][name], {{(context)}}[2][name]...")

If skeleton looks good (all accesses have [N][field]), just output: done()"""
