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
        """Setwise selection: pick the best restaurant from the set."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANTS]
{query}

From the set of restaurants above, select the ONE that BEST matches the user's request.

[SELECTION]"""

        return call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
