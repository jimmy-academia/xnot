#!/usr/bin/env python3
"""Setwise: Zero-shot Ranking via Set Selection.

Reference: A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking
with Large Language Models
Zhuang et al., SIGIR 2024
https://arxiv.org/abs/2310.09497

Approach:
Present a set of candidates and ask the LLM to select the MOST relevant one.
This balances efficiency (fewer comparisons than pairwise) with effectiveness
(considers multiple candidates together unlike pointwise).

Key difference from other approaches:
- Pointwise: Score each item individually
- Pairwise: Compare pairs, aggregate winners
- Listwise: Produce full ranking permutation
- Setwise: Select best from set (single selection)
"""

import re

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT = """You are evaluating a restaurant for a user's request.

Analyze the restaurant against the user's needs and determine if it should be recommended.

Output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)"""

SYSTEM_PROMPT_RANKING = """You are a ranking assistant that selects the most relevant restaurant.

Given a user request and a set of restaurants (each marked with [1], [2], etc.),
identify which restaurant is MOST relevant to the user's needs.

Consider all restaurants together and select the single best match.

Output format: ANSWER: N (where N is the number of the best restaurant)"""


class Setwise(BaseMethod):
    """Setwise: Select the most relevant item from a set of candidates."""

    name = "setwise"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Single-item evaluation (falls back to pointwise)."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT]
{query}

Analyze whether this restaurant matches the user's request."""

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Setwise selection: pick the best restaurant(s) from the set."""
        if k == 1:
            instruction = "select the ONE restaurant that BEST matches the user's request"
        else:
            instruction = f"select the TOP {k} restaurants in order from best to worst match"

        prompt = f"""[USER REQUEST]
{context}

[RESTAURANTS]
{query}

From the restaurants above, {instruction}.
Output your selection as numbers: [best], [second], [third], etc.

[SELECTION]"""

        response = call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
        return self._parse_selection(response, k)

    def _parse_selection(self, response: str, k: int) -> str:
        """Parse selection indices from response."""
        # Extract bracketed numbers [N] or plain numbers
        indices = re.findall(r'\[?(\d+)\]?', response)
        if indices:
            # Dedupe and take top k
            seen = set()
            result = []
            for idx in indices:
                idx_int = int(idx)
                if idx_int not in seen and idx_int > 0:
                    seen.add(idx_int)
                    result.append(str(idx_int))
                    if len(result) >= k:
                        break
            if result:
                return ", ".join(result)
        return "1"  # Fallback
