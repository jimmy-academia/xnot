#!/usr/bin/env python3
"""Prompt constants for ANoT phases - Multi-Step Design."""

from prompts.task_descriptions import RANKING_TASK_COMPACT

# Re-export for use in core.py
__all__ = [
    'SYSTEM_PROMPT', 'STEP1_EXTRACT_PROMPT', 'STEP2_PATH_PROMPT',
    'STEP1C_SEED_PROMPT', 'STEP3_RULEOUT_PROMPT', 'STEP4_SKELETON_PROMPT',
    'PHASE2_PROMPT', 'RANKING_TASK_COMPACT'
]

SYSTEM_PROMPT = "You follow instructions precisely. Output ONLY what is requested - no explanations, no caveats, no clarification requests. If asked yes/no, respond with only 'yes' or 'no'."

# =============================================================================
# STEP 1: Condition Extraction
# =============================================================================

STEP1_EXTRACT_PROMPT = """Extract conditions from the user request, preserving logical structure.

[USER REQUEST]
{query}

[RULES]
- ONLY extract conditions the user EXPLICITLY wants
- PRESERVE LOGICAL STRUCTURE: Use [OR] to group alternatives
- If user says "A or B or C", group them as [OR] A | B | C
- If user says "A and B" or just lists requirements, they are AND (default)
- For nested logic like "(A and B) or (C and D)", use [OR] (A AND B) | (C AND D)
- If user doesn't mention hours/timing, do NOT output any [HOURS] line
- Distinguish between review types:
  - "is praised for X", "known for great X" → [REVIEW:POSITIVE] X (sentiment check)
  - "complaints about X", "criticized for X" → [REVIEW:NEGATIVE] X (sentiment check)
  - "has reviews mentioning X", "reviews mention X" → [REVIEW:MENTION] X (keyword check)
  - "4+ reviews since 2020", "recent activity", "active recently" → [REVIEW:RECENCY] description (date-based count)
- For attribute requirements like "quiet", "trendy", "hipster vibe", use [ATTR] not [REVIEW]
- For social/friend filters like "my friend X reviewed", use [SOCIAL] friend_name, keyword

[OUTPUT FORMAT]
For AND conditions (default), list each on a new line:
[ATTR] description
[REVIEW:POSITIVE] topic

For OR conditions, group alternatives with pipe separator:
[OR] condition1 | condition2 | condition3

For nested OR of ANDs:
[OR] (cond1 AND cond2) | (cond3 AND cond4)

For parallel ORs with AND:
[OR] A | B
[OR] C | D
(means: (A or B) AND (C or D))

Example 1 - simple AND query:
User: "Looking for a quiet cafe with free WiFi"
[ATTR] quiet
[ATTR] free WiFi

Example 2 - simple OR query:
User: "Looking for a place with gelato, or touristy vibe, or upscale"
[OR] gelato | touristy vibe | upscale

Example 3 - AND with parallel ORs:
User: "Looking for a place that serves beer or is good for late night, and is quiet or touristy"
[OR] serves beer | good for late night
[OR] quiet | touristy

Example 4 - OR of ANDs:
User: "Looking for an Italian dinner place, or somewhere with alcohol and divey vibe"
[OR] (Italian AND dinner) | (alcohol AND divey vibe)

Example 5 - AND anchors with OR options:
User: "Looking for a place with TV, WiFi, reservations, that either serves alcohol or has classy outdoor seating"
[ATTR] TV
[ATTR] WiFi
[ATTR] reservations
[OR] alcohol | (classy AND outdoor seating)

Example 6 - recency/activity condition:
User: "Looking for a cafe with strong recent activity (4+ reviews since 2020)"
[REVIEW:RECENCY] 4+ reviews since 2020
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
   - "drive-thru" or "has drive-thru" → DriveThru=True
   - "outdoor seating" or "patio" → OutdoorSeating=True
   - "has TV" or "shows sports" → HasTV=True
   - "trendy" → Ambience.trendy=True
   - "romantic vibe" → Ambience.romantic=True
   - "divey atmosphere" → Ambience.divey=True
   - "not kid-friendly" → GoodForKids=False
   - "quiet" → NoiseLevel="quiet"
   - "free WiFi" → WiFi="free"
   - "serves alcohol" or "full bar" → Alcohol="full_bar" or Alcohol="beer_and_wine"
   - "good for late night" → GoodForMeal.latenight=True
   - "good for dinner" → GoodForMeal.dinner=True
   - "good for brunch" → GoodForMeal.brunch=True
3. For CATEGORY conditions (pub, gift shop, Italian, etc.): Use PATH: categories, TYPE: HARD
4. For HOURS conditions: ALWAYS use TYPE: SOFT (requires range checking, not exact match)
5. NEVER output "value" as the EXPECTED - always specify True, False, or a quoted string

[OUTPUT FORMAT]
PATH: attributes.FieldName
EXPECTED: True/False/"value"
TYPE: HARD

Example outputs:
- For "drive-thru": PATH: attributes.DriveThru, EXPECTED: True, TYPE: HARD
- For "romantic": PATH: attributes.Ambience.romantic, EXPECTED: True, TYPE: HARD
- For "quiet": PATH: attributes.NoiseLevel, EXPECTED: "quiet", TYPE: HARD
- For "serves alcohol": PATH: attributes.Alcohol, EXPECTED: "full_bar", TYPE: HARD
- For "late night": PATH: attributes.GoodForMeal.latenight, EXPECTED: True, TYPE: HARD

For category conditions (cuisine, place type):
PATH: categories
EXPECTED: "category_name"
TYPE: HARD
Examples: "Italian" → PATH: categories, EXPECTED: "Italian", TYPE: HARD
          "gift shop" → PATH: categories, EXPECTED: "Gift Shops", TYPE: HARD
          "pub" → PATH: categories, EXPECTED: "Pubs", TYPE: HARD

For review text search (only for sentiment/mentions):
PATH: reviews
EXPECTED: keyword
TYPE: SOFT

For hours conditions (ALWAYS use SOFT):
PATH: hours.DayName
EXPECTED: start:min-end:min
TYPE: SOFT

For date/recency conditions (e.g. "4+ reviews since 2020", "recent activity"):
PATH: reviews
EXPECTED: count>=4 since 2020-01-01 (or similar: count>=N since YYYY-MM-DD)
TYPE: SOFT
Note: The reviews array contains 'date' field for each review. The evaluator will count reviews matching the date criteria.
"""

# =============================================================================
# STEP 1c: Generate LWT Seed (template with logical structure)
# =============================================================================

STEP1C_SEED_PROMPT = """Create an LWT seed template that evaluates if an item matches the query.

[QUERY]
{query}

[RESOLVED CONDITIONS]
{resolved_conditions}

[TASK]
Create a SINGLE template that:
1. Lists each relevant attribute with path substitution syntax
2. States the logical expression the user wants
3. Asks yes/no for whether the item matches

[PATH SUBSTITUTION SYNTAX]
Use {{{{(context)}}}}[N] to reference item N's data:
- {{{{(context)}}}}[N][attributes][DriveThru] - item N's DriveThru attribute
- {{{{(context)}}}}[N][attributes][Ambience][romantic] - nested attribute
- {{{{(context)}}}}[N][categories] - item N's categories list

[OUTPUT FORMAT]
===LWT_SEED===
Item [N] data:
- field1: {{{{(context)}}}}[N][path1]
- field2: {{{{(context)}}}}[N][path2]
...

Query requires: (logical expression using field names)
Does this item satisfy the query? Answer yes or no.

[EXAMPLE 1: Simple OR]
Query: "Looking for drive-thru or romantic or divey"
Conditions: DriveThru=True OR Ambience.romantic=True OR Ambience.divey=True

===LWT_SEED===
Item [N] data:
- DriveThru: {{{{(context)}}}}[N][attributes][DriveThru]
- romantic: {{{{(context)}}}}[N][attributes][Ambience][romantic]
- divey: {{{{(context)}}}}[N][attributes][Ambience][divey]

Query requires: DriveThru=True OR romantic=True OR divey=True
Does this item satisfy the query? Answer yes or no.

[EXAMPLE 2: AND with nested OR]
Query: "Looking for a place with TV and WiFi that serves alcohol or has outdoor seating"
Conditions: HasTV=True AND WiFi=free AND (Alcohol!=none OR OutdoorSeating=True)

===LWT_SEED===
Item [N] data:
- HasTV: {{{{(context)}}}}[N][attributes][HasTV]
- WiFi: {{{{(context)}}}}[N][attributes][WiFi]
- Alcohol: {{{{(context)}}}}[N][attributes][Alcohol]
- OutdoorSeating: {{{{(context)}}}}[N][attributes][OutdoorSeating]

Query requires: HasTV=True AND WiFi='free' AND (Alcohol!='none' OR OutdoorSeating=True)
Does this item satisfy the query? Answer yes or no.

[EXAMPLE 3: Categories]
Query: "Looking for an Italian restaurant or pub"
Conditions: categories contains 'Italian' OR categories contains 'Pubs'

===LWT_SEED===
Item [N] data:
- categories: {{{{(context)}}}}[N][categories]

Query requires: categories contains 'Italian' OR categories contains 'Pubs'
Does this item satisfy the query? Answer yes or no.
"""

# =============================================================================
# STEP 3: Quick Rule-Out (check hard conditions, prune items)
# =============================================================================

STEP3_RULEOUT_PROMPT = """Check items against conditions and identify which pass.

[CONDITIONS]
{hard_conditions}

[ITEMS - relevant attributes only]
{items_compact}

[TASK]
For each item, check conditions with these rules:
- Regular conditions (no OR): Item must satisfy ALL of them (AND logic)
- OR groups (marked with "OR:"): Item must satisfy AT LEAST ONE option in the group
- Missing/None values count as not matching

Example:
  1. attributes.HasTV = True         <- must have TV
  2. OR: alcohol=full_bar | alcohol=beer_and_wine    <- either one is OK
  3. attributes.WiFi = free          <- must have free WiFi
Item passes if: HasTV=True AND (alcohol=full_bar OR alcohol=beer_and_wine) AND WiFi=free

[OUTPUT FORMAT]
===CANDIDATES===
[list of item numbers that pass, e.g., 2, 4, 6, 7]

===PRUNED===
item_number: reason
item_number: reason
...
"""

# =============================================================================
# STEP 4: LWT Skeleton Generation (duplicate seed for each candidate)
# =============================================================================

STEP4_SKELETON_PROMPT = """Duplicate the LWT seed template for each candidate item.

[CANDIDATES]
{candidates}

[LWT SEED]
{lwt_seed}

[SOFT CONDITIONS (if any)]
{soft_conditions}

[TASK]
1. For each candidate item number, create a step (cN) that replaces [N] with the actual item number
2. Add any additional steps for soft conditions (review analysis, hours checks)
3. End with a (final) step that aggregates all results

[OUTPUT FORMAT]
===LWT_SKELETON===
(cN)=LLM('... item N ...')
...
(final)=LLM('Item X={{(cX)}}, Item Y={{(cY)}}, ... Output item NUMBERS where answer is yes: ')

[EXAMPLE: candidates [2, 5, 7] with seed for OR condition]
LWT Seed:
Item [N] data:
- DriveThru: {{{{(context)}}}}[N][attributes][DriveThru]
- romantic: {{{{(context)}}}}[N][attributes][Ambience][romantic]
Query requires: DriveThru=True OR romantic=True
Does this item satisfy? yes/no

===LWT_SKELETON===
(c2)=LLM('Item 2 data: DriveThru={{{{(context)}}}}[2][attributes][DriveThru], romantic={{{{(context)}}}}[2][attributes][Ambience][romantic]. Query requires: DriveThru=True OR romantic=True. Satisfy? yes/no')
(c5)=LLM('Item 5 data: DriveThru={{{{(context)}}}}[5][attributes][DriveThru], romantic={{{{(context)}}}}[5][attributes][Ambience][romantic]. Query requires: DriveThru=True OR romantic=True. Satisfy? yes/no')
(c7)=LLM('Item 7 data: DriveThru={{{{(context)}}}}[7][attributes][DriveThru], romantic={{{{(context)}}}}[7][attributes][Ambience][romantic]. Query requires: DriveThru=True OR romantic=True. Satisfy? yes/no')
(final)=LLM('Item 2={{{{(c2)}}}}, Item 5={{{{(c5)}}}}, Item 7={{{{(c7)}}}}. Output item NUMBERS where answer is yes: ')

[IF SOFT CONDITIONS EXIST - add review/hours steps]
For REVIEW:MENTION conditions, add steps like:
(r2)=LLM('Item 2 reviews: {{{{(context)}}}}[2][reviews]. Do reviews mention X? yes/no')

For REVIEW:RECENCY conditions (e.g. "4+ reviews since 2020"), add steps like:
(r2)=LLM('Item 2 reviews with dates: {{{{(context)}}}}[2][reviews]. Count reviews where date >= 2020-01-01. Is count >= 4? Answer only: yes or no')

[CRITICAL]
- Replace [N] with actual item numbers from CANDIDATES
- Use double braces {{{{...}}}} for variable substitution
- Keep the logical expression EXACTLY as in the seed
- Reference previous step results with {{{{(cN)}}}} syntax"""

# =============================================================================
# PHASE 2: ReAct Context Expansion (iterate items, build lwt_script)
# =============================================================================

PHASE2_PROMPT = """Expand LWT seed into executable script by iterating over items.

[LWT SEED - template for evaluating one item]
{lwt_seed}

[RESOLVED CONDITIONS]
{conditions}

[LOGICAL STRUCTURE]
{logical_structure}

[NUMBER OF ITEMS]
{n_items}

[TOOLS]
list_items() → show all items with relevant attributes for the conditions
check_item(N) → show item N's full relevant attributes
drop_item(N, "reason") → mark item N as non-candidate (won't be evaluated)
add_step("cN", "prompt") → add LWT step for item N (prompt only, no LLM wrapper)
update_step("cN", "prompt") → modify existing step
get_review_lengths(N) → char counts per review (for review conditions)
keyword_search(N, "word") → find keyword positions in reviews
done() → finish expansion

[VARIABLE SYNTAX IN STEPS]
- {{{{(context)}}}}[N][attributes][FieldName] - item N's attribute
- {{{{(context)}}}}[N][categories] - item N's categories
- {{{{(context)}}}}[N][reviews] - item N's reviews
- {{{{(cN)}}}} - result from step cN

[TASK]
Build an executable LWT script:
1. Use list_items() to see all items with relevant attributes
2. For each item, decide:
   - If item CLEARLY cannot satisfy the logical structure → drop_item(N, reason)
   - If item MIGHT satisfy → add_step("cN", prompt based on seed)
3. If conditions involve reviews, use get_review_lengths() and keyword_search()
4. Add final aggregation step: add_step("final", "Item X={{{{(cX)}}}}, ... Output items with yes:")
5. Call done() when script is complete

[HANDLING REVIEW:RECENCY CONDITIONS]
For date-based review conditions (e.g. "4+ reviews since 2020"):
- The reviews array contains a 'date' field for each review (format: "YYYY-MM-DD HH:MM:SS")
- Create step prompts that explicitly instruct counting by date
- Example: add_step("r7", "Item 7 reviews: {{{{(context)}}}}[7][reviews]. Count reviews where date >= 2020-01-01. Is count >= 4? Answer only: yes or no")
- Do NOT just ask "4+ reviews since 2020?" - be explicit about counting dates

[CRITICAL]
- You can output MULTIPLE actions in one turn for efficiency
- After list_items(), you can batch drop_item and add_step calls
- NEVER write "Observation:" yourself
- Be CONSERVATIVE with drop_item:
  * ONLY drop items that CLEARLY FAIL HARD attribute conditions visible in list_items()
  * For SOFT conditions (reviews, recency, sentiment) - you CANNOT determine from list_items()
  * If item passes HARD conditions but has SOFT conditions → CREATE add_step(), DO NOT drop
- String values like 'True' and boolean True are equivalent (LLM will judge)
- You MUST call add_step("cN", ...) for EACH item that passes HARD conditions
- Do NOT describe what you would do - you must execute actual tool calls
- Pseudocode, plans, and descriptions are NOT valid - only tool calls work
- The final step MUST only reference step IDs that you actually created

[EXAMPLE TRACE - Batch Processing]
Thought: Let me see all items with their relevant attributes.
Action: list_items()
Observation: Item 1: DriveThru=None, romantic=False
             Item 2: DriveThru=None, romantic=True
             Item 7: DriveThru='True', romantic=False
             ... (all items)

Thought: Based on list_items, I can batch process all items.
         Items without any matching OR option should be dropped.
         Items with potential matches get evaluation steps.
Action: drop_item(1, "no OR satisfied")
Action: drop_item(3, "no OR satisfied")
Action: add_step("c2", "Item 2: DriveThru={{{{(context)}}}}[2][attributes][DriveThru], romantic={{{{(context)}}}}[2][attributes][Ambience][romantic]. DriveThru=True OR romantic=True? yes/no")
Action: add_step("c7", "Item 7: DriveThru={{{{(context)}}}}[7][attributes][DriveThru], romantic={{{{(context)}}}}[7][attributes][Ambience][romantic]. DriveThru=True OR romantic=True? yes/no")
Action: add_step("final", "c2={{{{(c2)}}}}, c7={{{{(c7)}}}}. Items with yes:")
Action: done()

Begin. Output Thought and Action:"""
