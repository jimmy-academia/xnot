#!/usr/bin/env python3
"""Least-to-Most (L2M) prompting baseline.

Reference: Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
https://arxiv.org/abs/2205.10625

Approach:
1. Decompose: Break user request into simpler sub-questions
2. Solve: Answer each sub-question from simplest to most complex
3. Aggregate: Use all sub-answers to make final recommendation
"""

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT_DECOMPOSE = """You are decomposing a user's restaurant request into simpler sub-questions.

Break down the request into 2-3 specific questions that can be answered by reading restaurant reviews.
Output each question on a new line, numbered 1, 2, 3."""

SYSTEM_PROMPT_SOLVE = """Answer the question based ONLY on the restaurant information provided.
Be concise and factual."""

SYSTEM_PROMPT_AGGREGATE = """Based on all the evidence gathered, make a final recommendation.

Output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)"""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurants for a user's request.
Analyze each restaurant against the user's criteria.

Output format: ANSWER: <n1>, <n2>, <n3>, ... (indices of best matches, best first)"""


class LeastToMost(BaseMethod):
    """Least-to-Most prompting: decompose → solve subproblems → aggregate."""

    name = "l2m"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """L2M evaluation: decompose request, solve sub-questions, aggregate."""
        # Step 1: Decompose user request into sub-questions
        decompose_prompt = f"""User request: {context}

Break this into 2-3 simpler questions to evaluate a restaurant."""

        subquestions_response = call_llm(decompose_prompt, system=SYSTEM_PROMPT_DECOMPOSE)
        subquestions = self._parse_subquestions(subquestions_response)

        # Step 2: Solve each sub-question (least to most complex)
        accumulated_context = f"[RESTAURANT INFO]\n{query}\n\n[USER REQUEST]\n{context}\n"

        for i, subq in enumerate(subquestions):
            solve_prompt = f"""{accumulated_context}
[QUESTION {i+1}]
{subq}

Answer based on the restaurant information above."""

            answer = call_llm(solve_prompt, system=SYSTEM_PROMPT_SOLVE)
            accumulated_context += f"\n[Q{i+1}] {subq}\n[A{i+1}] {answer}\n"

        # Step 3: Aggregate all evidence into final answer
        aggregate_prompt = f"""{accumulated_context}
[FINAL DECISION]
Based on all the above analysis, should this restaurant be recommended for the user's request?"""

        response = call_llm(aggregate_prompt, system=SYSTEM_PROMPT_AGGREGATE)
        return parse_final_answer(response)

    def _parse_subquestions(self, response: str) -> list:
        """Parse numbered sub-questions from LLM response."""
        lines = response.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            # Remove numbering like "1.", "1)", "1:"
            if line and line[0].isdigit():
                # Find where the actual question starts
                for i, ch in enumerate(line):
                    if ch in '.):' and i < 3:
                        line = line[i+1:].strip()
                        break
            if line and len(line) > 5:  # Filter out empty or too short lines
                questions.append(line)
        return questions[:3]  # Cap at 3 sub-questions

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking evaluation with L2M approach."""
        # For ranking, use simplified single-pass L2M
        decompose_prompt = f"""User request: {context}

What are the 2-3 key criteria to evaluate restaurants for this request?"""

        criteria_response = call_llm(decompose_prompt, system=SYSTEM_PROMPT_DECOMPOSE)

        # Build analysis prompt with top-k instruction
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request."
        else:
            instruction = f"Select the TOP {k} restaurants that best match, ranked from best to worst."

        ranking_prompt = f"""[KEY CRITERIA]
{criteria_response}

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

Evaluate each restaurant against the criteria above.
{instruction}

[ANALYSIS]"""

        return call_llm(ranking_prompt, system=SYSTEM_PROMPT_RANKING)
