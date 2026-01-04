#!/usr/bin/env python3
"""Prompt constants for ANoT phases."""

SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested."

PHASE1_PROMPT = """Analyze the user request and rank items.

[USER REQUEST]
{context}

[ITEMS]
{items_compact}

[TASK]
1. Extract conditions (e.g., DriveThru=True, GoodForKids=True, HasTV=False)
2. Find which items match ALL conditions
3. Output the matching item indices

[OUTPUT FORMAT]
===LWT_SKELETON===
(final)=LLM("User wants: {context}. Item(s) that match: [LIST INDICES]. Output the best index.")

===MESSAGE===
CONDITIONS: <list>
REMAINING: <indices of matching items>
NEEDS_EXPANSION: no
"""

PHASE2_PROMPT = """Check if the LWT skeleton needs expansion, then call done().

[MESSAGE FROM PHASE 1]
{message}

[CURRENT LWT]
{lwt_skeleton}

[TOOLS]
- done() → finish (call this when skeleton is complete)
- lwt_insert(idx, "step") → add step (only if needed)
- read(path) → get data (only if needed)

[DECISION]
Look at NEEDS_EXPANSION in the message:
- If "no" → just call done() now
- If "yes" → use tools to add steps, then done()

What is your action?
"""
