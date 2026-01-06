#!/usr/bin/env python3
"""PaRaDe: Passage Ranking using Demonstrations with LLMs.

Reference: PaRaDe: Passage Ranking using Demonstrations with LLMs
Sun et al., EMNLP 2023
https://arxiv.org/abs/2310.15449

Approach:
Few-shot demonstration-based ranking where the model learns
the evaluation pattern from explicit examples.
"""

import re

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


# Few-shot demonstrations for restaurant evaluation
DEMONSTRATIONS = """=== Example 1 ===
[USER REQUEST]
Looking for a quiet place to work with good coffee and wifi.

[RESTAURANT]
Name: Cafe Studious
Reviews: "Perfect for remote work, quiet atmosphere and strong wifi." "Great lattes, plenty of outlets." "Can get crowded on weekends but weekdays are peaceful."

[REASONING]
The user needs: quiet environment, good coffee, wifi access.
Evidence from reviews:
- Quiet: "quiet atmosphere", "peaceful" on weekdays
- Coffee: "Great lattes"
- Wifi: "strong wifi", "plenty of outlets"
All key requirements are met with positive evidence.

ANSWER: 1

=== Example 2 ===
[USER REQUEST]
Need a romantic dinner spot with vegetarian options.

[RESTAURANT]
Name: Burger Palace
Reviews: "Best burgers in town!" "Great for watching sports with friends." "Loud and fun atmosphere, cold beers."

[REASONING]
The user needs: romantic atmosphere, vegetarian options.
Evidence from reviews:
- Romantic: No mention, described as "loud" and for "watching sports" - opposite of romantic
- Vegetarian: No mention, focus is on burgers and meat
Key requirements are not met.

ANSWER: -1

=== Example 3 ===
[USER REQUEST]
Family-friendly brunch place with outdoor seating.

[RESTAURANT]
Name: Garden Bistro
Reviews: "Nice patio area." "Food was okay, nothing special." "Kids menu available but limited options."

[REASONING]
The user needs: family-friendly, brunch, outdoor seating.
Evidence from reviews:
- Outdoor: "Nice patio area" - positive
- Family: "Kids menu available but limited" - partial match
- Brunch/Food quality: "okay, nothing special" - neutral
Mixed evidence, partially meets requirements.

ANSWER: 0

"""

SYSTEM_PROMPT = """You are evaluating restaurants using the demonstrated pattern.

Follow the same reasoning format as the examples:
1. Identify what the user needs
2. Find evidence in the reviews for each need
3. Determine if requirements are met

Output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)"""

SYSTEM_PROMPT_RANKING = """You are ranking restaurants using the demonstrated evaluation pattern.

For each restaurant, apply the same reasoning:
1. Identify user needs
2. Find evidence in reviews
3. Score how well each meets requirements

Output ANSWER: <number> for the best matching restaurant."""


class PaRaDe(BaseMethod):
    """PaRaDe: Few-shot demonstration-based ranking."""

    name = "parade"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Few-shot evaluation with demonstrations.

        Args:
            query: User request text
            context: Restaurant data
        """
        prompt = f"""{DEMONSTRATIONS}
=== Your Task ===
[USER REQUEST]
{query}

[RESTAURANT]
{context}

[REASONING]"""

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking with demonstration-based evaluation.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        if k == 1:
            instruction = "select the BEST matching restaurant"
        else:
            instruction = f"rank the TOP {k} restaurants from best to worst"

        prompt = f"""{DEMONSTRATIONS}
=== Your Task ===
[USER REQUEST]
{query}

[RESTAURANTS]
{context}

Evaluate each restaurant using the demonstrated pattern, then {instruction}.
Output your ranking as numbers: [best], [second], [third], etc.

[RANKING]"""

        response = call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
        return self._parse_ranking(response, k)

    def _parse_ranking(self, response: str, k: int) -> str:
        """Parse ranking indices from response."""
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
