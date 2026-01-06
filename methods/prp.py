#!/usr/bin/env python3
"""PRP: Pairwise Ranking Prompting for restaurant recommendation.

Reference: Large Language Models are Effective Text Rankers with Pairwise
Ranking Prompting
Qin et al., NAACL 2024
https://arxiv.org/abs/2306.17563

Approach:
Compare items in pairs, aggregate wins, rank by win count.
Complexity: O(n²) comparisons for n items.
"""

import re
import asyncio
from typing import List, Tuple, Dict

from .base import BaseMethod
from utils.llm import call_llm, call_llm_async
from utils.parsing import parse_final_answer


# Pairwise comparison prompt
COMPARISON_PROMPT = """Given this user request, which restaurant is MORE relevant?

User Request:
{query}

Restaurant A:
{item_a}

Restaurant B:
{item_b}

Which restaurant better matches the user's needs?
Analyze briefly, then output your choice.
Output ONLY one of: A, B, or TIE"""

# For single-item evaluation
EVALUATE_PROMPT = """Evaluate if this restaurant matches the user's needs.

User Request:
{query}

Restaurant:
{context}

Does this restaurant match the user's requirements?
Output: 1 (matches), 0 (unclear), or -1 (does not match)

ANSWER:"""


class PairwiseRankingPrompting(BaseMethod):
    """Pairwise Ranking Prompting method.

    Compares all pairs of items, counts wins, ranks by win count.
    """

    name = "prp"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Evaluate single item.

        Args:
            query: User request text
            context: Restaurant data

        Returns:
            -1, 0, or 1
        """
        prompt = EVALUATE_PROMPT.format(query=query, context=context)
        response = call_llm(prompt)

        return parse_final_answer(response)

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task using pairwise comparisons.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        items = self._parse_items(context)

        if not items:
            return "1"  # Fallback

        if len(items) == 1:
            return str(items[0][0])

        # Initialize win counts
        wins: Dict[int, int] = {idx: 0 for idx, _ in items}

        # Build all pairs for parallel comparison
        pairs = []
        for i, (idx_a, item_a) in enumerate(items):
            for idx_b, item_b in items[i+1:]:
                pairs.append((item_a, item_b, idx_a, idx_b))

        # Compare all pairs in parallel using async
        async def compare_all():
            tasks = [
                self._compare_pair_async(item_a, item_b, idx_a, idx_b, query)
                for item_a, item_b, idx_a, idx_b in pairs
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(compare_all())

        # Aggregate wins
        for idx_a, idx_b, winner in results:
            if winner == "A":
                wins[idx_a] += 1
            elif winner == "B":
                wins[idx_b] += 1
            # TIE: no points awarded

        # Rank by wins (descending), then by index for deterministic ties
        ranked = sorted(wins.items(), key=lambda x: (-x[1], x[0]))

        # Return top-k indices
        top_k = [str(idx) for idx, _ in ranked[:k]]
        return ", ".join(top_k)

    def _parse_items(self, query: str) -> List[Tuple[int, str]]:
        """Parse ranking query into (index, item_text) pairs."""
        items = []

        # Split by [N] patterns - match [1], [2], etc.
        pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'

        for match in re.finditer(pattern, query, re.DOTALL):
            idx = int(match.group(1))
            item_text = match.group(2).strip()
            if item_text:  # Only add non-empty items
                items.append((idx, item_text))

        return items

    def _compare_pair(self, item_a: str, item_b: str,
                      idx_a: int, idx_b: int, query: str) -> str:
        """Compare two items and return winner: 'A', 'B', or 'TIE'.

        Args:
            query: User request text
        """
        prompt = COMPARISON_PROMPT.format(
            query=query,
            item_a=item_a,
            item_b=item_b
        )

        response = call_llm(prompt)
        return self._parse_winner(response)

    async def _compare_pair_async(self, item_a: str, item_b: str,
                                   idx_a: int, idx_b: int, query: str) -> Tuple[int, int, str]:
        """Async version: compare two items, return (idx_a, idx_b, winner).

        Args:
            query: User request text
        """
        prompt = COMPARISON_PROMPT.format(
            query=query,
            item_a=item_a,
            item_b=item_b
        )

        response = await call_llm_async(prompt)
        return (idx_a, idx_b, self._parse_winner(response))

    def _parse_winner(self, response: str) -> str:
        """Parse winner from LLM response."""
        response_upper = response.strip().upper()

        # Look for clear A or B at the end (most common pattern)
        lines = response_upper.split('\n')
        last_line = lines[-1].strip() if lines else ""

        # Check last line first
        if last_line in ["A", "B", "TIE"]:
            return last_line

        # Check for patterns like "A." or "B." or "Winner: A"
        if re.search(r'\bA\b[.\s]*$', last_line):
            return "A"
        if re.search(r'\bB\b[.\s]*$', last_line):
            return "B"
        if "TIE" in last_line:
            return "TIE"

        # Fallback: check whole response
        # Count occurrences to avoid ambiguity
        a_count = len(re.findall(r'\bA\b', response_upper))
        b_count = len(re.findall(r'\bB\b', response_upper))

        # Only return if one is clearly dominant at the end
        if "RESTAURANT A" in response_upper and "RESTAURANT B" not in response_upper[-50:]:
            return "A"
        if "RESTAURANT B" in response_upper and "RESTAURANT A" not in response_upper[-50:]:
            return "B"

        # Check for explicit choice patterns
        choice_patterns = [
            (r'CHOOSE[:\s]+A\b', "A"),
            (r'CHOOSE[:\s]+B\b', "B"),
            (r'WINNER[:\s]+A\b', "A"),
            (r'WINNER[:\s]+B\b', "B"),
            (r'ANSWER[:\s]+A\b', "A"),
            (r'ANSWER[:\s]+B\b', "B"),
            (r'OUTPUT[:\s]+A\b', "A"),
            (r'OUTPUT[:\s]+B\b', "B"),
        ]

        for pattern, result in choice_patterns:
            if re.search(pattern, response_upper):
                return result

        # Default to TIE if unclear
        return "TIE"
