#!/usr/bin/env python3
"""Chain-of-Thought method for restaurant recommendation."""

import re

from .base import BaseMethod
from utils.llm import call_llm
from .shared import _defense, _use_defense_prompt
from .prompts import DEFENSE_PREAMBLE, DEFENSE_PREAMBLE_SHORT, with_defense
from utils.parsing import parse_final_answer

# Normal prompt - minimal baseline
SYSTEM_PROMPT_NORMAL = """Rate this restaurant. Output ANSWER: 1, 0, or -1."""

# Defense prompt - with data quality checks (uses shared preamble)
SYSTEM_PROMPT_DEFENSE = f"""You are evaluating whether a restaurant matches a user's SPECIFIC need.

{DEFENSE_PREAMBLE}

Then analyze the reviews for the user's specific request and output:
ANSWER: 1 (recommend), 0 (neutral/unclear), or -1 (not recommend)"""

# Ranking prompts (for selecting best item from multiple options)
SYSTEM_PROMPT_RANKING = """You are selecting the best restaurant for a user's request.
You will see multiple restaurants with their reviews, each numbered 1 to N.
Analyze each restaurant against the user's criteria and output ONLY the index number
of the best matching restaurant.

Output format: ANSWER: <number>"""

SYSTEM_PROMPT_RANKING_DEFENSE = f"""You are selecting the best restaurant for a user's request.
You will see multiple restaurants numbered 1 to N.

{DEFENSE_PREAMBLE_SHORT}

Analyze each restaurant against the user's specific criteria and output ONLY the
index number of the best matching restaurant.

Output format: ANSWER: <number>"""


class ChainOfThought(BaseMethod):
    """Chain-of-Thought prompting method."""

    name = "cot"

    def __init__(self, run_dir: str = None, defense: bool = False, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Evaluate restaurant recommendation. Returns -1, 0, or 1."""
        prompt = self._build_prompt(query, context)
        system = self._get_system_prompt()
        response = call_llm(prompt, system=system)
        return parse_final_answer(response)

    def _get_system_prompt(self) -> str:
        """Get system prompt based on defense mode."""
        # Check both instance defense and module-level defense
        use_defense = self.defense or _use_defense_prompt
        system = SYSTEM_PROMPT_DEFENSE if use_defense else SYSTEM_PROMPT_NORMAL
        if _defense:
            system = _defense + "\n\n" + system
        return system

    def _build_prompt(self, query: str, context: str) -> str:
        """Build zero-shot prompt."""
        parts = []
        parts.append("=== Your Task ===")
        parts.append(f"\n[RESTAURANT INFO]\n{query}")
        parts.append(f"\n[USER REQUEST]\n{context}")
        parts.append("\n[ANALYSIS]")
        return "\n".join(parts)

    # Note: _parse_response removed - using parse_final_answer from shared.py

    # --- Ranking Methods ---

    def _build_ranking_prompt(self, query: str, context: str, k: int = 1) -> str:
        """Build prompt for ranking task (selecting best from multiple items)."""
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request.\nOutput only the restaurant number."
        else:
            instruction = f"Select the TOP {k} restaurants that best match the user's request.\nOutput {k} numbers separated by commas, best match first."

        return f"""=== Your Task ===

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

{instruction}

[ANALYSIS]"""

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task. Returns response string (parsed by run.py).

        Args:
            query: All restaurants formatted with indices (from format_ranking_query)
            context: User request text
            k: Number of top predictions to return

        Returns:
            LLM response string containing the best restaurant index(es)
        """
        prompt = self._build_ranking_prompt(query, context, k)
        use_defense = self.defense or _use_defense_prompt
        system = SYSTEM_PROMPT_RANKING_DEFENSE if use_defense else SYSTEM_PROMPT_RANKING
        if _defense:
            system = _defense + "\n\n" + system
        return call_llm(prompt, system=system)
