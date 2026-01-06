#!/usr/bin/env python3
"""Fine-Grained Relevance Ranking method for restaurant recommendation.

Reference: Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring
Fine-Grained Relevance Labels
Zhuang et al., SIGIR 2024
https://arxiv.org/abs/2310.14122

Approach:
Score each item on a 4-point relevance scale, then rank by scores.
"""

import re
import asyncio
from typing import List, Tuple

from .base import BaseMethod
from utils.llm import call_llm, call_llm_async
from utils.parsing import parse_final_answer


# 4-point relevance scale (from paper)
RELEVANCE_LABELS = {
    "highly relevant": 3,
    "relevant": 2,
    "somewhat relevant": 1,
    "not relevant": 0,
}

# Scoring prompt - used for each item independently
SCORING_PROMPT = """Given this user request and restaurant information, rate the relevance.

User Request:
{query}

Restaurant:
{item}

Rate the relevance using EXACTLY one of these labels:
- Highly Relevant: Restaurant clearly matches all key requirements
- Relevant: Restaurant matches most requirements
- Somewhat Relevant: Restaurant partially matches some requirements
- Not Relevant: Restaurant does not match the requirements

Your rating (output ONLY the label):"""

# For single-item evaluation (non-ranking mode)
EVALUATE_PROMPT = """Given this user request and restaurant information, rate the relevance.

User Request:
{query}

Restaurant:
{context}

Rate the relevance using EXACTLY one of these labels:
- Highly Relevant: Restaurant clearly matches all key requirements
- Relevant: Restaurant matches most requirements
- Somewhat Relevant: Restaurant partially matches some requirements
- Not Relevant: Restaurant does not match the requirements

Your rating (output ONLY the label):"""


class FineGrainedRanker(BaseMethod):
    """Fine-Grained Relevance Ranking method.

    Scores each item independently on a 4-point scale, then ranks by score.
    """

    name = "finegrained"

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

        score = self._parse_relevance(response)

        # Map 4-point scale to -1/0/1
        # 0-1 (not relevant, somewhat relevant) -> -1
        # 2 (relevant) -> 0
        # 3 (highly relevant) -> 1
        if score <= 1:
            return -1
        elif score == 2:
            return 0
        else:
            return 1

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task using pointwise scoring.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        # Parse individual items from context (restaurant data)
        items = self._parse_items(context)

        if not items:
            return "1"  # Fallback

        # Score all items in parallel using async
        async def score_all():
            tasks = [
                self._score_item_async(idx, item_text, query)
                for idx, item_text in items
            ]
            return await asyncio.gather(*tasks)

        scores = asyncio.run(score_all())

        # Rank by score (descending), then by index for ties
        ranked = sorted(scores, key=lambda x: (-x[1], x[0]))

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

    def _score_item(self, item_text: str, query: str) -> int:
        """Score a single item on the 4-point relevance scale.

        Args:
            query: User request text
        """
        prompt = SCORING_PROMPT.format(query=query, item=item_text)
        response = call_llm(prompt)

        return self._parse_relevance(response)

    async def _score_item_async(self, idx: int, item_text: str, query: str) -> Tuple[int, int]:
        """Async version: score a single item, return (idx, score).

        Args:
            query: User request text
        """
        prompt = SCORING_PROMPT.format(query=query, item=item_text)
        response = await call_llm_async(prompt)

        return (idx, self._parse_relevance(response))

    def _parse_relevance(self, response: str) -> int:
        """Parse relevance label from LLM response."""
        response_lower = response.lower().strip()

        # Check for exact matches first (order matters - check "highly relevant" before "relevant")
        if "highly relevant" in response_lower:
            return 3
        elif "not relevant" in response_lower:
            return 0
        elif "somewhat relevant" in response_lower:
            return 1
        elif "relevant" in response_lower:
            return 2

        # Fallback: check for numeric scores
        numbers = re.findall(r'\b([0-3])\b', response)
        if numbers:
            return int(numbers[0])

        # Default to not relevant
        return 0
