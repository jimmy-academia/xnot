#!/usr/bin/env python3
"""Plan-and-Solve (PS) prompting baseline.

Reference: Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
by Large Language Models
https://arxiv.org/abs/2305.04091

Approach:
1. Plan: Devise a step-by-step plan to evaluate the restaurant
2. Solve: Execute each step of the plan to reach a conclusion
"""

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT_PLAN = """You are devising a plan to evaluate a restaurant for a user's specific request.

Create a brief 3-step plan to systematically analyze the restaurant.
Output each step on a new line, numbered 1, 2, 3."""

SYSTEM_PROMPT_SOLVE = """Execute the current step of the evaluation plan.
Be concise and factual based on the restaurant information."""

SYSTEM_PROMPT_FINAL = """Based on your step-by-step analysis, make a final recommendation.

Output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)"""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurants for a user's request.
Analyze each restaurant against the user's criteria.

Output format: ANSWER: <n1>, <n2>, <n3>, ... (indices of best matches, best first)"""


class PlanAndSolve(BaseMethod):
    """Plan-and-Solve prompting: devise plan → execute steps → conclude."""

    name = "ps"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """PS evaluation: plan then solve step by step."""
        # Step 1: Devise a plan
        plan_prompt = f"""User request: {context}

Restaurant information is available. Devise a 3-step plan to evaluate if this restaurant matches the user's needs."""

        plan_response = call_llm(plan_prompt, system=SYSTEM_PROMPT_PLAN)
        steps = self._parse_steps(plan_response)

        # Step 2: Execute each step
        accumulated = f"[RESTAURANT INFO]\n{query}\n\n[USER REQUEST]\n{context}\n\n[PLAN]\n{plan_response}\n"

        for i, step in enumerate(steps):
            solve_prompt = f"""{accumulated}
[EXECUTING STEP {i+1}]
{step}

Analyze the restaurant information for this step."""

            step_result = call_llm(solve_prompt, system=SYSTEM_PROMPT_SOLVE)
            accumulated += f"\n[STEP {i+1} RESULT]\n{step_result}\n"

        # Step 3: Final recommendation
        final_prompt = f"""{accumulated}
[FINAL RECOMMENDATION]
Based on all steps above, should this restaurant be recommended?"""

        response = call_llm(final_prompt, system=SYSTEM_PROMPT_FINAL)
        return parse_final_answer(response)

    def _parse_steps(self, response: str) -> list:
        """Parse numbered steps from LLM response."""
        lines = response.strip().split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering like "1.", "1)", "1:"
                for i, ch in enumerate(line):
                    if ch in '.):' and i < 3:
                        line = line[i+1:].strip()
                        break
            if line and len(line) > 5:
                steps.append(line)
        return steps[:3]  # Cap at 3 steps

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking with Plan-and-Solve approach."""
        # Plan phase
        plan_prompt = f"""User request: {context}

Devise a 3-step plan to compare multiple restaurants and select the best match."""

        plan_response = call_llm(plan_prompt, system=SYSTEM_PROMPT_PLAN)

        # Build instruction based on k
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request."
        else:
            instruction = f"Select the TOP {k} restaurants that best match, ranked from best to worst."

        # Solve phase (single pass for ranking)
        ranking_prompt = f"""[EVALUATION PLAN]
{plan_response}

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

Execute the plan above to evaluate each restaurant.
{instruction}

[ANALYSIS]"""

        return call_llm(ranking_prompt, system=SYSTEM_PROMPT_RANKING)
