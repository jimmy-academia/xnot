#!/usr/bin/env python3
"""CoT-Self-Consistency method for restaurant recommendation."""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseMethod
from utils.llm import call_llm
from .shared import majority_vote
from utils.parsing import parse_final_answer

DEFAULT_N_SAMPLES = 5

SYSTEM_PROMPT = """Rate this restaurant. Output ANSWER: 1, 0, or -1."""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurants for a user's request.
Analyze each restaurant against the user's criteria.

Output format: ANSWER: <n1>, <n2>, <n3>, ... (indices of best matches, best first)"""


class CoTSelfConsistency(BaseMethod):
    """CoT with Self-Consistency (majority voting over multiple samples)."""

    name = "cotsc"

    def __init__(self, run_dir: str = None, n_samples: int = DEFAULT_N_SAMPLES, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)
        self.n_samples = n_samples

    def evaluate(self, query: str, context: str) -> int:
        """Sample multiple reasoning paths, vote on final answer."""
        prompt = self._build_prompt(query, context)

        # Sample in parallel
        answers = []
        with ThreadPoolExecutor(max_workers=self.n_samples) as executor:
            futures = [
                executor.submit(self._sample_once, prompt)
                for _ in range(self.n_samples)
            ]
            for future in as_completed(futures):
                try:
                    answers.append(future.result())
                except Exception:
                    pass  # Skip failed samples

        return majority_vote(answers)

    def _sample_once(self, prompt: str) -> int:
        """Generate one sample."""
        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""=== Your Task ===

[RESTAURANT INFO]
{query}

[USER REQUEST]
{context}

[ANALYSIS]"""

    # --- Ranking Methods ---

    def _build_ranking_prompt(self, query: str, context: str, k: int = 1) -> str:
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request."
        else:
            instruction = f"Select the TOP {k} restaurants that best match."
        return f"""=== Your Task ===

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

{instruction}

[ANALYSIS]"""

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking with self-consistency voting on parsed indices."""
        prompt = self._build_ranking_prompt(query, context, k)

        # Sample multiple ranking responses
        responses = []
        with ThreadPoolExecutor(max_workers=self.n_samples) as executor:
            futures = [
                executor.submit(call_llm, prompt, SYSTEM_PROMPT_RANKING)
                for _ in range(self.n_samples)
            ]
            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception:
                    pass

        if not responses:
            return "1"  # Default fallback

        # Parse indices from each response and vote with position weights
        # Position 1 gets weight k, position 2 gets weight k-1, etc.
        index_scores = Counter()
        for response in responses:
            indices = self._parse_indices(response, max_index=20, k=k)
            for pos, idx in enumerate(indices):
                weight = k - pos  # First position gets highest weight
                index_scores[idx] += weight

        # Return top-k indices by score
        if not index_scores:
            return "1"

        top_indices = [idx for idx, _ in index_scores.most_common(k)]
        return ", ".join(str(i) for i in top_indices)

    def _parse_indices(self, response: str, max_index: int = 20, k: int = 5) -> list:
        """Parse up to k indices from LLM response."""
        if response is None:
            return []
        indices = []
        for match in re.finditer(r'\b(\d+)\b', str(response)):
            idx = int(match.group(1))
            if 1 <= idx <= max_index and idx not in indices:
                indices.append(idx)
                if len(indices) >= k:
                    break
        return indices
