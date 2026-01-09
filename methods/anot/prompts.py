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
- attributes.Alcohol: "full_bar", "beer_and_wine", "none"
- attributes.CoatCheck: True/False
- attributes.RestaurantsPriceRange2: 1, 2, 3, 4
- hours: dict with day names as keys, values like "8:0-22:0"
- reviews: list of review objects with 'text' field

[TASK]
1. Identify which field to check for this condition
2. Determine expected value based on what the user WANTS:
   - "trendy" → user wants trendy=True
   - "not kid-friendly" → user wants GoodForKids=False
   - "quiet" → user wants NoiseLevel="quiet"
   - "free WiFi" → user wants WiFi="free"
3. For HOURS conditions: ALWAYS use TYPE: SOFT (requires range checking, not exact match)

[OUTPUT FORMAT]
PATH: attributes.FieldName
EXPECTED: True/False/"value"
TYPE: HARD

For review text search:
PATH: reviews
EXPECTED: keyword
TYPE: SOFT

For hours conditions (ALWAYS use SOFT):
PATH: hours.DayName
EXPECTED: start:min-end:min
TYPE: SOFT

For date/recency conditions (e.g. "reviews since 2020"):
PATH: reviews
EXPECTED: date >= 2020-01-01
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

[CRITICAL RULES]
1. ONLY generate steps for items in CANDIDATES list above - no other items!
2. Generate ONE step per candidate per soft condition
3. MUST end with a (final) step that aggregates results and outputs ranking
4. NEVER use {{(context)}}[N][reviews] - this loads ALL reviews and is TOO LARGE!
5. Use placeholder [NEEDS_SEARCH] for review steps - Phase 2 will use tools to find relevant reviews

[CONDITION TYPES - USE CORRECT FORMAT]
For REVIEW conditions - use placeholder that Phase 2 will expand:
  (rN)=LLM('[NEEDS_SEARCH:N:keyword] Do any reviews mention keyword? yes/no')

For HOURS conditions (check if open during requested time):
  IMPORTANT: Use ACTUAL day names (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
  NEVER use "Today", "Evening", "Morning" - these keys don't exist in data!
  (hN)=LLM('Item N hours Monday: {{(context)}}[N][hours][Monday]. User needs START-END. Is start<=START AND end>=END? yes/no')

For SOCIAL conditions (check friend reviews):
  (sN)=LLM('[NEEDS_SOCIAL_SEARCH:N:friend_name:keyword] Does friend_name mention keyword? yes/no')

[VARIABLE SYNTAX]
- {{(context)}}[N][hours][Day] - Item N's hours for Day (e.g., Monday)
- {{(context)}}[N][reviews][R][text][start:end] - Specific review slice (Phase 2 generates)
- {{(rN)}} or {{(hN)}} - Result from step rN or hN

[OUTPUT FORMAT]
===LWT_SKELETON===
(step_id)=LLM('...')
...
(final)=LLM('Item N={{(rN)}}, ... Output item NUMBERS with yes first: ')

[EXAMPLE: candidates [2,4] with REVIEW condition for "coffee"]
===LWT_SKELETON===
(r2)=LLM('[NEEDS_SEARCH:2:coffee] Do reviews mention coffee? yes/no')
(r4)=LLM('[NEEDS_SEARCH:4:coffee] Do reviews mention coffee? yes/no')
(final)=LLM('Item 2={{(r2)}}, Item 4={{(r4)}}. Output item NUMBERS with yes first: ')

[EXAMPLE: candidates [5,6] with HOURS condition (early morning on Monday)]
===LWT_SKELETON===
(h5)=LLM('Item 5 hours Monday: {{(context)}}[5][hours][Monday]. User needs 7:0-8:0. Is start<=7:0 AND end>=8:0? yes/no')
(h6)=LLM('Item 6 hours Monday: {{(context)}}[6][hours][Monday]. User needs 7:0-8:0. Is start<=7:0 AND end>=8:0? yes/no')
(final)=LLM('Item 5={{(h5)}}, Item 6={{(h6)}}. Output item NUMBERS with yes first: ')

NOTE: For "late night" or "evening" requests, check multiple days or use a representative day like Friday/Saturday.

[EXAMPLE: candidates [3] with SOCIAL condition for friend "Kevin" and keyword "place"]
===LWT_SKELETON===
(s3)=LLM('[NEEDS_SOCIAL_SEARCH:3:Kevin:place] Does Kevin mention place? yes/no')
(final)=LLM('Item 3={{(s3)}}. Output item NUMBERS with yes first: ')

[IF NO SOFT CONDITIONS]
===LWT_SKELETON===
(final)=LLM('Candidates: {candidates}. Output as comma-separated: ')"""

# =============================================================================
# PHASE 2: ReAct Expansion (refine skeleton with slice syntax for long reviews)
# =============================================================================

PHASE2_PROMPT = """Expand LWT skeleton: replace placeholders with actual review slices.

[CONDITIONS]
{conditions}

[SKELETON]
{lwt_skeleton}

[TOOLS]
get_review_lengths(N) → char counts per review
keyword_search(N, "word") → review indices and positions where keyword appears
social_search(N, "friend_name", "keyword") → review indices where friend mentions keyword
update_step("rN", "prompt text") → update step (just the prompt, NOT "LLM(...)")
done() → finish

[SLICE SYNTAX]
{{(context)}}[N][reviews][R][text][start:end]

[CRITICAL TASK]
Find steps with [NEEDS_SEARCH:N:keyword] or [NEEDS_SOCIAL_SEARCH:N:friend:keyword] placeholders.
For EACH such step:
1. Use keyword_search(N, "keyword") or social_search(N, "friend", "keyword") to find matching reviews
2. Use update_step to replace placeholder with actual review slices

[OUTPUT FORMAT FOR update_step]
After keyword_search returns matches like "Review 0: pos 150, Review 2: pos 300":
update_step("r2", "Item 2 review excerpts: {{(context)}}[2][reviews][0][text][100:300], {{(context)}}[2][reviews][2][text][250:450]. Do these mention coffee? yes/no")

[CRITICAL INDEXING - ITEMS ARE 1-INDEXED]
- Item numbers start at 1 (Item 1, Item 2, ...), NOT 0
- Step IDs match item numbers: r1, r2, r3, ... (NOT r0!)
- NEVER use r0, s0, h0 - these don't exist
- Example: For Item 5, use step "r5" or "s5"

[OTHER RULES]
- Output ONE action, then STOP. Wait for system response.
- NEVER write "Observation:" yourself - system provides it.
- ALWAYS use slice syntax [start:end] - NEVER include full review text!
- Use ~200 chars around each match: [pos-100:pos+100]
- If no matches found, update step with "No matches found: no"
- Hours steps (hN) and (final) steps need no changes - skip them.

[PROCESS]
1. Find first step with [NEEDS_SEARCH] or [NEEDS_SOCIAL_SEARCH] placeholder
2. Call keyword_search or social_search to find positions
3. Call update_step with slice syntax (use correct 1-indexed step ID!)
4. Repeat until all placeholders expanded
5. Call done()

Begin. Output Thought and Action:"""
