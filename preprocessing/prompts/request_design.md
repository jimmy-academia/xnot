# Request Design Prompts

Prompts for designing benchmark requests with logical structures.

## Generic Template

### Group Design Kickstart

```
Design {count} requests for group {group_id} with structure pattern: {pattern}

**Selected Items:**
- data/{name}/restaurants.jsonl - {N} items
- data/{name}/reviews.jsonl - reviews for each item

**Reference:**
- doc/reference/evidence_types.md - evidence type specifications
- doc/reference/request_structure.md - JSON schema

**Requirements:**
1. Each request matches exactly 1 item
2. Use bottom-up anchor-first methodology
3. Natural language text signals all conditions
4. Diverse item coverage (don't reuse same gold answer)

**Output for each request:**
- id: R{NN}
- group: {group_id}
- scenario: Persona name (e.g., "Busy Parent", "Wine Enthusiast")
- text: Natural language request
- shorthand: Compact notation
- structure: Full JSON
- gold_{item_type}: Unique ID of matching item
```

### Condition Verification

```
Verify that request {request_id} matches exactly 1 item.

**Request:**
{request_json}

**Check:**
1. Evaluate each condition against all {N} items
2. List items that satisfy ALL conditions
3. Confirm gold item matches
4. If multi-match, suggest additional anchor condition
```

### Natural Language Refinement

```
Rewrite the request text to naturally signal all conditions.

**Current:**
text: "{current_text}"
structure: {structure}

**Missing signals:**
{list_of_unmentioned_conditions}

**Requirements:**
- Sound natural, like a real user query
- Mention all conditions without being robotic
- Keep the persona/scenario intact
```

---

## philly_cafes Actual Prompts

### Session 1: G01 Simple AND

**Prompt:**
```
Design 10 requests for group G01 with structure: AND(condition1, condition2, condition3)

These should test basic attribute matching. Use unique feature combinations identified earlier.

For each request:
1. Start with the unique/rare feature as anchor
2. Add 1-2 common features
3. Verify only 1 restaurant matches
4. Write natural request text

Example:
- Anchor: DriveThru=True (only Milkcrate Cafe)
- Add: GoodForKids=True, HasTV=False
- Result: Only Milkcrate Cafe matches all 3
```

**Response Summary:**
- Created R01-R10 with simple AND structures
- Each used identified unique combinations
- All validated to match exactly 1 restaurant

### Session 2: G04 Credibility Weighting

**Prompt:**
```
Design 10 requests for group G04 with structure: AND(item_meta conditions, review_text with weight_by)

These should test credibility-weighted review pattern matching.

Weight fields available:
- ["user", "review_count"] - experienced reviewers
- ["user", "elite"] - elite status (number of years)
- ["user", "fans"] - popular reviewers
- ["useful"] - community-validated reviews

For each request:
1. Use 1-2 item_meta conditions as anchors
2. Add review_text condition with weight_by
3. Verify only 1 restaurant has enough credible reviewers mentioning pattern
```

**Response Summary:**
- Created R31-R40 with credibility-weighted patterns
- Tested different weight fields
- Calibrated thresholds for single-match

### Session 3: G09-G10 Social Filter

**Prompt:**
```
Design social filter requests for G09 (1-hop) and G10 (2-hop).

**Setup:**
1. First create user_mapping.json with:
   - 20-30 synthetic users with names
   - Friend graph connecting users
   - Restaurant reviews mapping [user_id, pattern_mentioned]

2. For G09 (1HOP):
   - Request specifies friend names directly
   - Only reviews from those friends qualify
   - Pattern must be mentioned

3. For G10 (2HOP):
   - Request specifies friend names
   - Reviews from friends OR friends-of-friends qualify
   - Pattern must be mentioned

**Requirements:**
- Each request matches exactly 1 restaurant
- Varying friend lists and patterns
- Test both inclusion and exclusion
```

**Response Summary:**
- Created user_mapping.json with 25 synthetic users
- Built friend graph with varying connectivity
- Created R81-R90 (1-hop) and R91-R100 (2-hop)
- All validated to match exactly 1 restaurant

### Session 4: Text Rewriting

**Prompt:**
```
Review all 100 requests and identify where the text doesn't signal all conditions.

For example:
- Request R01 text: "I need a cafe with a drive-thru option"
- Structure has: DriveThru=True, GoodForKids=True, HasTV=False
- Missing: GoodForKids and HasTV not mentioned in text

Rewrite texts to signal all conditions naturally:
- "Looking for a cafe with a drive-thru, that's kid-friendly, and without TVs"

Run preprocessing/rewrite_requests.py philly_cafes --analyze first
Then provide rewritten texts for flagged requests.
```

**Response Summary:**
- Identified 23 requests with missing signals
- Rewrote each to include all conditions
- Maintained natural language flow

---

## Usage Notes

### Bottom-Up Anchor-First Methodology

1. **Find anchor**: Unique or rare feature that narrows to 1-5 items
2. **Add conditions**: Layer additional conditions to narrow to exactly 1
3. **Verify uniqueness**: Check no other item satisfies all
4. **Write text**: Natural language mentioning all conditions
5. **Validate**: Run `python -m data.validate {name}`

### Structure Patterns by Group

| Group | Pattern | Example |
|-------|---------|---------|
| G01 | `AND(a, b, c)` | `AND(drive_thru, kids, no_tv)` |
| G02 | `AND(anchor, OR(a, b))` | `AND(full_bar, OR(music, outdoor))` |
| G03 | `AND(a, OR(b, c))` | `AND(wifi, OR(quiet, cozy))` |
| G04 | `AND(a, weight_by)` | `AND(byob, elite_mention_coffee)` |
| G05 | `AND(a, OR(b, c, d))` | `AND(drive_thru, OR(love, coffee, best))` |
| G06 | `AND(a, OR(AND, AND))` | `AND(takeout, OR(AND(a,b), AND(c,d)))` |
| G07 | `AND(a, OR, OR)` | `AND(hipster, OR(a,b), OR(c,d))` |
| G08 | `AND(a, OR(b, AND))` | `AND(quiet, OR(slow, AND(elite, best)))` |
| G09 | `1HOP` | `1HOP(['Alice'], 'cozy')` |
| G10 | `2HOP` | `2HOP(['Bob'], 'recommend')` |

### Common Issues

- **Multi-match**: Add more specific anchor or combination
- **No match**: Check condition values match data format
- **Unnatural text**: Revise to flow better while keeping all signals
