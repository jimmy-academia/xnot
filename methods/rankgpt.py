#!/usr/bin/env python3
"""RankGPT: LLMs as Re-Ranking Agents using Permutation Generation.

Reference: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
Sun et al., EMNLP 2023 (Outstanding Paper Award)
https://arxiv.org/abs/2304.09542

Approach:
Listwise permutation-based ranking where the LLM produces a ranked
ordering of all candidates simultaneously, rather than scoring each
individually (pointwise) or comparing pairs (pairwise).
"""

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT = """You are RankGPT, an intelligent assistant that ranks restaurants based on user requirements.

Given a user request and a list of restaurants (each marked with a number like [1], [2], etc.),
rank ALL restaurants from MOST relevant to LEAST relevant.

Output your ranking as a permutation: [X] > [Y] > [Z] > ...
where > means "is more relevant than".

After the ranking, state ANSWER: N where N is the number of the BEST restaurant."""

SYSTEM_PROMPT_RANKING = """You are RankGPT, an intelligent assistant that ranks restaurants.

I will provide you with restaurants, each indicated by a numerical identifier [].
Rank ALL restaurants based on their relevance to the user's request.

Output format:
1. Produce a ranking permutation: [X] > [Y] > [Z] > ... (most to least relevant)
2. End with: ANSWER: N (where N is the best restaurant's number)"""


class RankGPT(BaseMethod):
    """RankGPT: Listwise permutation-based ranking."""

    name = "rankgpt"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Permutation-based evaluation (single item maps to recommend/not)."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT]
{query}

Should this restaurant be recommended? Output ANSWER: 1 (yes), 0 (neutral), or -1 (no)."""

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Listwise permutation ranking of all restaurants."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANTS]
{query}

Rank ALL restaurants from MOST to LEAST relevant to the user's request.

[RANKING]
Permutation (most relevant first):"""

        return call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
