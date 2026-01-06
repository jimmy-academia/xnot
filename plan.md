# Data Fix Plan

## Problem Summary

### Issue 1: Request Text Errors

**G03 (R20-R29): Hours conditions missing in text**
```
Current:  "Looking for a cafe open on  at "
Expected: "Looking for a cafe open on Monday at 7:00 AM"
```
- Template variables `{day}` and `{time}` not substituted
- All 10 G03 requests have this problem

**G04 (R30-R39): Multiple rendering failures**
- `not_true` conditions render as `[unknown condition: {...}]`
  - R30: `no_dogs` → `[unknown condition: {'aspect': 'no_dogs', ...}]`
  - R32: `no_coat_check` → `[unknown condition: {...}]`
- `weight_by` metadata not reflected in text:
  - R30: "where reviews mention 'coffee'" should be "where **expert reviewers** mention 'coffee'"
  - R31: weight_by `["user", "fans"]` → should mention "influencers"

**G05 (R40-R49): OR conditions not rendered**
```
Current:  "[unknown condition: {'op': 'OR', 'args': [...]}]"
Expected: "mentions 'wine list' OR 'cocktail' in reviews"
```
- All 10 G05 requests have raw structure embedded in text

**Other text issues:**
- R07: Duplicate "without WiFi" appears twice
- R39: Duplicate "with a hipster vibe"
- R41: Duplicate "that's kid-friendly"
- Many texts sound mechanical ("Looking for a cafe with X, with Y, with Z")

---

### Issue 2: Query Format Errors

**Reviews missing user metadata (needed for G04):**
```python
# Current in loader.py:175-184
"reviews": [
    {"review_id": r.get("review_id", ""),
     "review": r.get("text", ""),
     "stars": r.get("stars", 0),
     "date": r.get("date", "")}
]

# Missing fields from raw reviews.jsonl:
# - useful, funny, cool (review-level)
# - user.review_count, user.elite, user.fans (user metadata)
```

**Dynamic construction:**
- Data assembled at runtime in `load_dataset()`
- Want pre-prepared file for simpler loading and reproducibility

---

### Issue 3: Query/Context Swap Needed

**Current:**
- `query` = restaurant data (large dict)
- `context` = user request text (small string)

**Proposed:**
- `query` = user request text (what user is asking for)
- `context` = restaurant data (background information)

This is more semantically correct: "query" should be what we're searching for.

---

### Issue 4: Documentation Outdated

- `doc/evaluation_spec.md`: outdated schema (mentions `request_id`, `valid_idx`)
- No documentation for request `structure` format
- No documentation for evidence types (item_meta, item_meta_hours, review_text)
- No documentation for weight_by conditions

---

## Fix Plan: Phased Approach

Each phase has a **validation step** before proceeding to the next.

### Phase 1: Fix Request Text (Critical)

Break into 5 steps, one per group, for incremental validation.

---

**Step 1.1: Fix G01 (R00-R09) - Basic attribute requests** ✅ COMPLETED

Current issues:
- Mechanical phrasing: "Looking for a cafe with X, with Y, with Z"
- R07 had duplicate "without WiFi" (structure had both `no_wifi` and `wifi_none`)

Fix applied:
- Created `data/scripts/fix_g01_requests.py`
- Removed duplicate conditions from structure
- Improved text rendering: adjectives first, then "with" phrases, then restrictions
- Used proper grammar ("that's X" for adjectives, "with X" for noun phrases)

Validation:
```bash
head -10 data/philly_cafes/requests.jsonl | python3 -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]"
# Verified: R07 no longer has duplicate, all texts natural
```

---

**Step 1.2: Fix G02 (R10-R19) - Review text pattern requests** ✅ COMPLETED

Current issues:
- Mechanical phrasing
- Missing mappings for `BusinessAcceptsCreditCards`, `WheelchairAccessible`

Fix applied:
- Extended fix script with `--group` parameter
- Added phrase type classification (adj, with, where, other)
- "where reviews mention 'X'" now properly formatted
- Added missing attribute mappings

Validation:
```bash
sed -n '11,20p' data/philly_cafes/requests.jsonl | python3 -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]"
# Verified: review patterns use "where reviews mention", attributes render correctly
```

---

**Step 1.3: Fix G03 (R20-R29) - Hours conditions** ✅ COMPLETED

Current issues:
- "open on  at " - day in `path[1]`, time in `true` field not extracted
- R29 had `not_true` condition showing as `[unknown condition: ...]`
- Missing mappings for `Alcohol='beer_and_wine'`, `RestaurantsAttire`

Fix applied:
- Added `format_time_range()` to convert "7:0-8:0" → "from 7:00 AM to 8:00 AM"
- Extract day from `path[1]` in `item_meta_hours`
- Added `negate_phrase()` for `not_true` conditions
- Added Alcohol variants (both `u'...'` and `'...'` formats)
- Added RestaurantsAttire mappings

Validation:
```bash
sed -n '21,30p' data/philly_cafes/requests.jsonl | python3 -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]"
# Verified: hours render as "open on Monday from 7:00 AM to 8:00 AM"
# Verified: R29 not_true renders as "without outdoor seating"
```

---

**Step 1.4: Fix G04 (R30-R39) - Negated conditions + weight_by** ✅ COMPLETED

Current issues:
- `not_true` rendered as `[unknown condition: ...]`
- `weight_by` not reflected in text
- R33 had `RestaurantsAttire=u'casual'` unrendered
- R39 had duplicate "with a hipster vibe"

Fix applied:
- Added `weight_by` handling in `review_text` conditions:
  - `['user', 'review_count']` → "experienced reviewers mention"
  - `['user', 'fans']` → "popular reviewers mention"
  - `['user', 'elite']` → "elite reviewers mention"
  - `['useful']` → "helpful reviews mention"
  - `['user', 'friends']` → "well-connected reviewers mention" (for future use)
- `not_true` already handled via `negate_phrase()` from G03
- Added RestaurantsAttire variants
- Duplicates auto-removed by `remove_duplicates()`

Validation:
```bash
sed -n '31,40p' data/philly_cafes/requests.jsonl | python3 -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]"
# Verified: weight_by reflected, not_true rendered, no duplicates
```

---

**Step 1.5: Fix G05 (R40-R49) - OR conditions** ✅ COMPLETED

Current issues:
- OR operators rendered as `[unknown condition: {'op': 'OR', ...}]`
- R41 had duplicate "that's kid-friendly"
- Mixed OR types (review_text + item_meta) needed special handling

Fix applied:
- Added OR handling in `condition_to_phrase()`
- Review-only ORs: "reviews mention 'X' or 'Y'" (compact form)
- Mixed ORs: "trendy or reviews mention 'wine list'"
- Updated `remove_duplicates()` to handle OR conditions
- Duplicates auto-removed

Known edge cases:
- R45: Has structural redundancy (wifi in both OR and standalone condition)
- R49: Mixed OR reads slightly awkward but acceptable

Validation:
```bash
sed -n '41,50p' data/philly_cafes/requests.jsonl | python3 -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]"
# Verified: OR conditions render as "reviews mention 'X' or 'Y'"
```

---

**Phase 1 Implementation Approach:**

Create `data/scripts/render_request_text.py`:
```python
def render_request_text(structure: dict) -> str:
    """Render request structure to natural language."""
    # Handle evidence types: item_meta, item_meta_hours, review_text
    # Handle operators: AND, OR
    # Handle modifiers: not_true, weight_by
```

Then regenerate all texts and validate group by group.

---

### Phase 2: Fix Query Format (For G04 to Work)

**Step 2.1: Update load_dataset() to preserve user metadata**
```python
# In loader.py, update review dict to include:
{
    "review_id": r.get("review_id", ""),
    "review": r.get("text", ""),
    "stars": r.get("stars", 0),
    "date": r.get("date", ""),
    "useful": r.get("useful", 0),
    "funny": r.get("funny", 0),
    "cool": r.get("cool", 0),
    "user": {
        "user_id": r.get("user_id", ""),
        "review_count": r.get("user", {}).get("review_count", 0),
        "elite": r.get("user", {}).get("elite", []),
        "fans": r.get("user", {}).get("fans", 0),
    }
}
```

**Step 2.2: Update format_query() dict mode to include user metadata**

**Step 2.3: (Optional) Pre-generate static data file**
- Create `philly_cafes/items.jsonl` with fully assembled items
- Simplify load_dataset() to just load this file

**Validation 2:**
```bash
# Test that G04 validation passes with user metadata
python -c "from data.validate import *; from data.loader import *; ..."
# Verify: R30-R39 gold restaurants pass their weighted review conditions
```

---

### Phase 3: Swap Query/Context ✅ COMPLETED

**Step 3.1: Update run/evaluate.py** ✅
- Renamed parameter from `context` to `query` in `evaluate_ranking_single`
- Internal variable `query` → `context` for formatted restaurant data
- Now: `query` = user request, `context` = restaurant data

**Step 3.2: Update method interfaces** ✅
- Updated all 18+ methods in `methods/*.py`:
  - cot.py, ps.py, listwise.py, prp.py, finegrained.py, l2m.py, react.py
  - pot.py, cot_table.py, weaver.py, rankgpt.py, selfask.py, setwise.py
  - parade.py, pal.py, decomp.py
- Swapped `{query}` and `{context}` in all prompt templates
- Updated docstrings to reflect new semantics

**Step 3.3: Update prompts/task_descriptions.py** ✅
- Changed `{context}` to `{query}` for user request references

**Step 3.4: Update methods/base.py** ✅
- Updated abstract method docstrings

**Validation 3:** ✅
```bash
# Tested dummy method - no errors
python main.py --method dummy --limit 3 -v

# Tested cot method - runs successfully (tokens consumed)
python main.py --method cot --limit 1 -v --dev
```

---

### Phase 4: Update Documentation

**Step 4.1: Document request structure schema**
- Create `doc/request_schema.md`
- Document all evidence types with examples
- Document weight_by options

**Step 4.2: Update doc/evaluation_spec.md**
- Fix outdated field names
- Add correct data schemas
- Add query/context swap explanation

**Step 4.3: Add data format documentation**
- Create `data/philly_cafes/README.md`
- Document all files: restaurants.jsonl, reviews.jsonl, requests.jsonl, groundtruth.jsonl

**Validation 4:**
```bash
# Documentation review: ensure examples match actual data
# No automated test - manual review
```

---

## Execution Order (Revised)

```
Phase 3 (Query/Context Swap) ✅ COMPLETED
  └── Step 3.1 → Step 3.2 → Step 3.3 → Step 3.4 → Validation 3 ✅

Phase 1 (Request Text) ✅ COMPLETED
  └── Step 1.1 (G01) ✅
  └── Step 1.2 (G02) ✅
  └── Step 1.3 (G03) ✅
  └── Step 1.4 (G04) ✅
  └── Step 1.5 (G05) ✅

Phase 2 (Query Format + Social Feature) - NEXT
  └── Step 2.1: Add user metadata to reviews
  └── Step 2.2: Create synthetic users and friend graph (G06, G07)
  └── Step 2.3: Create social-based requests

Phase 4 (Documentation)
  └── Step 4.1 → Step 4.2 → Step 4.3 → Validation 4
```

**Rationale:**
- Phase 3 done first: clean code refactor with no data dependencies
- Phase 1 completed: all 50 requests now have proper natural language text
- Phase 2 expanded: includes social friend feature for 2 new groups (G06, G07)

---

## Questions for Clarification

1. **Naturalness level:** Should request text be very natural ("I'm looking for a cozy cafe...") or structured ("Requirements: X, Y, Z")?

2. **Pre-generated data:** Should we create a single `items.jsonl` file, or keep dynamic assembly?

3. **Query/context swap scope:** Should we also update CLAUDE.md and all documentation references, or just the code?

4. **Backward compatibility:** Do we need to support the old interface temporarily?
