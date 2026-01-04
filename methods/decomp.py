#!/usr/bin/env python3
"""Decomposed Prompting method for restaurant recommendation.

Reference: Decomposed Prompting: A Modular Approach for Solving Complex Tasks
Khot et al., ICLR 2023
https://arxiv.org/abs/2205.10825

Approach:
LLM generates a decomposition plan, then executes specialized
handlers for each sub-task.
"""

import re
from typing import List, Dict, NamedTuple

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


MAX_STEPS = 6


class PlanStep(NamedTuple):
    """A single step in the decomposition plan."""
    step_num: int
    handler: str
    inputs: List[str]
    output: str


# Decomposer prompt - generates the execution plan
DECOMPOSER_PROMPT = """You are decomposing a restaurant recommendation task into sub-tasks.

Available handlers:
- extract_requirements(context) -> list of user requirements from their request
- analyze_reviews(reviews, criteria) -> analysis of how reviews relate to criteria
- count_evidence(analysis) -> count of positive vs negative evidence
- make_decision(evidence, requirements) -> final recommendation (-1, 0, or 1)

Task: Given restaurant reviews and a user's requirements, determine if the restaurant matches their needs.

Generate a step-by-step plan using these handlers. You MUST end with make_decision.
Format each step as: Step N: [handler_name] input1, input2 -> output_name

Example plan:
Step 1: [extract_requirements] context -> requirements
Step 2: [analyze_reviews] reviews, requirements -> analysis
Step 3: [count_evidence] analysis -> evidence
Step 4: [make_decision] evidence, requirements -> answer

Now generate your plan:"""

# Handler prompts - specialized prompts for each sub-task
HANDLER_PROMPTS = {
    "extract_requirements": """Extract the key requirements from this user request.
Output a numbered list of specific criteria the restaurant must meet.

User request:
{context}

Requirements:""",

    "analyze_reviews": """Analyze these restaurant reviews for the following criteria.
For each criterion, identify relevant evidence (positive, negative, or neutral).

Criteria to check:
{criteria}

Reviews:
{reviews}

Analysis:""",

    "count_evidence": """Based on this analysis, count the evidence.

Analysis:
{analysis}

Summary:
- Number of requirements with POSITIVE evidence:
- Number of requirements with NEGATIVE evidence:
- Number of requirements with UNCLEAR/NO evidence:

Evidence summary:""",

    "make_decision": """Based on this evidence and requirements, make a recommendation.

Evidence:
{evidence}

Original requirements:
{requirements}

Decision rules:
- Output 1 if the restaurant clearly matches most requirements (positive > negative)
- Output -1 if the restaurant clearly does NOT match (negative > positive)
- Output 0 if evidence is mixed, unclear, or insufficient

Your recommendation (explain briefly, then output ANSWER: -1, 0, or 1):""",
}

# Ranking prompts
DECOMPOSER_PROMPT_RANKING = """You are decomposing a restaurant selection task into sub-tasks.

Available handlers:
- extract_requirements(context) -> list of user requirements
- analyze_all_restaurants(restaurants, requirements) -> analysis of each restaurant
- rank_restaurants(analysis, requirements) -> ranking from best to worst match
- select_best(ranking) -> index of best matching restaurant

Task: Given multiple restaurants and a user's requirements, select the BEST matching restaurant.

Generate a step-by-step plan. You MUST end with select_best.
Format: Step N: [handler_name] input1, input2 -> output_name

Now generate your plan:"""

HANDLER_PROMPTS_RANKING = {
    "extract_requirements": HANDLER_PROMPTS["extract_requirements"],

    "analyze_all_restaurants": """Analyze each restaurant for the following requirements.

Requirements:
{requirements}

Restaurants:
{restaurants}

For each restaurant, note how well it matches the requirements.

Analysis:""",

    "rank_restaurants": """Based on this analysis, rank the restaurants from best to worst match.

Analysis:
{analysis}

Requirements:
{requirements}

Ranking (best first):""",

    "select_best": """Based on this ranking, output the index number of the best restaurant.

Ranking:
{ranking}

Output ONLY the number of the best restaurant.
ANSWER:""",
}


class DecomposedPrompting(BaseMethod):
    """Decomposed Prompting method - LLM generates decomposition plan, then executes handlers."""

    name = "decomp"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Evaluate restaurant recommendation using decomposed prompting."""
        # Step 1: Generate decomposition plan
        plan = self._generate_plan(query, context)
        steps = self._parse_plan(plan)

        if not steps:
            # Fallback: direct evaluation if plan parsing fails
            return self._fallback_evaluate(query, context)

        # Step 2: Execute plan sequentially
        results = {
            "context": context,
            "reviews": query,
            "query": query,
        }

        for step in steps[:MAX_STEPS]:
            result = self._execute_handler(step, results, query, context)
            results[step.output] = result

        # Step 3: Parse final result from last step
        final_output = results.get(steps[-1].output, results.get("answer", "0"))
        return parse_final_answer(final_output)

    def _generate_plan(self, query: str, context: str) -> str:
        """Generate decomposition plan using LLM."""
        return call_llm(DECOMPOSER_PROMPT)

    def _parse_plan(self, plan: str) -> List[PlanStep]:
        """Parse LLM-generated plan into executable steps."""
        steps = []
        # Match: Step N: [handler_name] input1, input2 -> output_name
        pattern = r'Step\s*(\d+):\s*\[(\w+)\]\s*([^->\n]+)\s*->\s*(\w+)'

        for match in re.finditer(pattern, plan, re.IGNORECASE):
            step_num = int(match.group(1))
            handler = match.group(2).strip().lower()
            inputs_str = match.group(3).strip()
            output = match.group(4).strip()

            # Parse inputs (comma-separated)
            inputs = [inp.strip() for inp in inputs_str.split(',')]

            steps.append(PlanStep(
                step_num=step_num,
                handler=handler,
                inputs=inputs,
                output=output
            ))

        return steps

    def _execute_handler(self, step: PlanStep, results: Dict[str, str],
                         query: str, context: str) -> str:
        """Execute a single handler with its inputs."""
        handler_name = step.handler
        if handler_name not in HANDLER_PROMPTS:
            # Unknown handler - return empty
            return ""

        prompt_template = HANDLER_PROMPTS[handler_name]

        # Build substitution dict from results and inputs
        subs = {}
        for inp in step.inputs:
            inp_lower = inp.lower()
            if inp_lower in results:
                subs[inp_lower] = results[inp_lower]
            elif inp in results:
                subs[inp.lower()] = results[inp]

        # Add common substitutions
        subs["context"] = context
        subs["reviews"] = query
        subs["query"] = query

        # Map input names to template variables
        if handler_name == "analyze_reviews":
            subs["criteria"] = results.get("requirements", context)
        elif handler_name == "count_evidence":
            subs["analysis"] = results.get("analysis", "")
        elif handler_name == "make_decision":
            subs["evidence"] = results.get("evidence", results.get("analysis", ""))
            subs["requirements"] = results.get("requirements", context)

        # Substitute variables in prompt
        try:
            prompt = prompt_template.format(**subs)
        except KeyError:
            # Missing variable - use what we have
            prompt = prompt_template
            for key, value in subs.items():
                prompt = prompt.replace("{" + key + "}", str(value))

        return call_llm(prompt)

    def _fallback_evaluate(self, query: str, context: str) -> int:
        """Fallback to simple CoT-style evaluation if plan parsing fails."""
        prompt = f"""Evaluate if this restaurant matches the user's needs.

Restaurant reviews:
{query}

User requirements:
{context}

Analyze the reviews and output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)."""

        response = call_llm(prompt)
        return parse_final_answer(response)

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task using decomposed prompting."""
        # Step 1: Generate plan for ranking
        plan = self._generate_ranking_plan()
        steps = self._parse_plan(plan)

        if not steps:
            return self._fallback_ranking(query, context, k)

        # Step 2: Execute plan
        results = {
            "context": context,
            "restaurants": query,
            "query": query,
        }

        for step in steps[:MAX_STEPS]:
            result = self._execute_ranking_handler(step, results, query, context)
            results[step.output] = result

        # Return final result
        return results.get(steps[-1].output, results.get("answer", "1"))

    def _generate_ranking_plan(self) -> str:
        """Generate ranking decomposition plan."""
        return call_llm(DECOMPOSER_PROMPT_RANKING)

    def _execute_ranking_handler(self, step: PlanStep, results: Dict[str, str],
                                  query: str, context: str) -> str:
        """Execute a ranking handler."""
        handler_name = step.handler

        if handler_name not in HANDLER_PROMPTS_RANKING:
            return ""

        prompt_template = HANDLER_PROMPTS_RANKING[handler_name]

        # Build substitutions
        subs = {
            "context": context,
            "restaurants": query,
            "requirements": results.get("requirements", context),
            "analysis": results.get("analysis", ""),
            "ranking": results.get("ranking", ""),
        }

        try:
            prompt = prompt_template.format(**subs)
        except KeyError:
            prompt = prompt_template
            for key, value in subs.items():
                prompt = prompt.replace("{" + key + "}", str(value))

        return call_llm(prompt)

    def _fallback_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Fallback ranking if plan parsing fails."""
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request.\nOutput only the restaurant number."
        else:
            instruction = f"Select the TOP {k} restaurants.\nOutput {k} numbers separated by commas."

        prompt = f"""Select the best restaurant for this user.

Restaurants:
{query}

User requirements:
{context}

{instruction}

ANSWER:"""
        return call_llm(prompt)
