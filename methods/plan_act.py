#!/usr/bin/env python3
"""Plan-and-Act baseline method.

Reference: Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks
https://arxiv.org/abs/2503.09572

Key difference from Plan-and-Solve (PS):
- PS: Fixed 3-step plan, execute all steps, then decide
- Plan-and-Act: Dynamic replanning after each step based on execution results
"""

import re
from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT_PLAN = """You are a Planner that creates step-by-step plans to evaluate restaurants.

Given a user request and restaurant data, break down the evaluation into clear steps.
Each step should be a specific analysis task that builds toward the final recommendation.

Output format (2-4 steps):
## Step 1
Reasoning: [why this step is needed]
Step: [what to check/analyze]

## Step 2
Reasoning: [why this step is needed]
Step: [what to check/analyze]

..."""

SYSTEM_PROMPT_EXECUTE = """You are an Executor that carries out analysis steps.

Given a step instruction and restaurant data, perform the specific analysis requested.
Be concise and factual. Report what you find in the data.

Output your analysis result directly."""

SYSTEM_PROMPT_REPLAN = """You are a Planner updating the remaining steps based on execution results.

Review what has been accomplished and what still needs to be done.
If the goal can be achieved with the information gathered, output: DONE
Otherwise, output the remaining steps needed.

Output format:
## Step N
Reasoning: [why this step is needed given current findings]
Step: [what to check/analyze next]

Or if sufficient information gathered:
DONE"""

SYSTEM_PROMPT_DECIDE = """Based on the step-by-step analysis, make a final recommendation.

Review all findings and determine if the restaurant matches the user's request.

Output ANSWER: 1 (recommend), 0 (neutral/unclear), or -1 (not recommend)"""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurants for a user's request.

Analyze each restaurant step-by-step against the user's criteria.

Output format: ANSWER: <n1>, <n2>, <n3>, ... (indices of best matches, best first)"""


class PlanAndAct(BaseMethod):
    """Plan-and-Act: Planner generates plan, Executor executes, Planner replans dynamically."""

    name = "plan_act"

    def __init__(self, run_dir: str = None, defense: bool = False, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)
        self.max_steps = 6  # Safety limit

    def evaluate(self, query: str, context: str) -> int:
        """Plan-and-Act evaluation with dynamic replanning."""
        # Phase 1: Initial Planning
        plan = self._plan(query, context)
        if not plan:
            # Fallback to direct evaluation if planning fails
            return self._direct_evaluate(query, context)

        # Phase 2: Execute + Replan Loop
        history = []
        step_count = 0

        while plan and step_count < self.max_steps:
            # Execute current step
            current_step = plan[0]
            result = self._execute(current_step, query, context, history)
            history.append((current_step, result))
            step_count += 1

            # Replan based on results
            if len(plan) > 1:
                plan = self._replan(query, context, history)
            else:
                plan = []  # Last step done

        # Phase 3: Final Decision
        return self._decide(query, context, history)

    def _plan(self, query: str, context: str) -> list:
        """Generate initial plan."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT DATA]
{query}

Create a plan to evaluate if this restaurant matches the user's request."""

        response = call_llm(prompt, system=SYSTEM_PROMPT_PLAN)
        return self._parse_plan(response)

    def _execute(self, step: str, query: str, context: str, history: list) -> str:
        """Execute a single step of the plan."""
        # Build history context
        history_text = ""
        for i, (prev_step, prev_result) in enumerate(history, 1):
            history_text += f"\n[STEP {i}] {prev_step}\n[RESULT {i}] {prev_result}\n"

        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT DATA]
{query}
{history_text}
[CURRENT STEP]
{step}

Execute this step and report your findings."""

        return call_llm(prompt, system=SYSTEM_PROMPT_EXECUTE)

    def _replan(self, query: str, context: str, history: list) -> list:
        """Replan based on execution history."""
        # Build history context
        history_text = ""
        for i, (step, result) in enumerate(history, 1):
            history_text += f"\n## Completed Step {i}\nStep: {step}\nResult: {result}\n"

        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT DATA AVAILABLE]
Restaurant information has been provided.

[EXECUTION HISTORY]
{history_text}

Based on what has been accomplished, what steps remain?
If you have enough information to make a recommendation, output DONE."""

        response = call_llm(prompt, system=SYSTEM_PROMPT_REPLAN)

        # Check if done
        if "DONE" in response.upper():
            return []

        return self._parse_plan(response)

    def _decide(self, query: str, context: str, history: list) -> int:
        """Make final recommendation based on all findings."""
        # Build findings summary
        findings = ""
        for i, (step, result) in enumerate(history, 1):
            findings += f"\n[STEP {i}] {step}\n[FINDING {i}] {result}\n"

        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT DATA]
{query}

[ANALYSIS FINDINGS]
{findings}

Based on all findings above, should this restaurant be recommended?"""

        response = call_llm(prompt, system=SYSTEM_PROMPT_DECIDE)
        return parse_final_answer(response)

    def _direct_evaluate(self, query: str, context: str) -> int:
        """Fallback direct evaluation if planning fails."""
        prompt = f"""[USER REQUEST]
{context}

[RESTAURANT DATA]
{query}

Evaluate if this restaurant matches the user's request."""

        response = call_llm(prompt, system=SYSTEM_PROMPT_DECIDE)
        return parse_final_answer(response)

    def _parse_plan(self, response: str) -> list:
        """Parse structured plan from LLM response.

        Expected format:
        ## Step 1
        Reasoning: ...
        Step: ...

        ## Step 2
        ...
        """
        steps = []
        # Match "## Step N" sections
        pattern = r'##\s*Step\s*\d+.*?(?=##\s*Step|\Z)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

        for match in matches:
            # Extract the "Step:" line content
            step_match = re.search(r'Step:\s*(.+?)(?:\n|$)', match, re.IGNORECASE)
            if step_match:
                step_text = step_match.group(1).strip()
                if step_text:
                    steps.append(step_text)
            else:
                # Fallback: use the whole section minus header
                lines = match.strip().split('\n')
                if len(lines) > 1:
                    # Join non-header lines
                    content = ' '.join(line.strip() for line in lines[1:] if line.strip())
                    if content:
                        steps.append(content)

        # Fallback: try numbered list parsing like PS
        if not steps:
            steps = self._parse_numbered_steps(response)

        return steps

    def _parse_numbered_steps(self, response: str) -> list:
        """Fallback parser for numbered lists."""
        lines = response.strip().split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                for i, ch in enumerate(line):
                    if ch in '.):' and i < 3:
                        line = line[i+1:].strip()
                        break
            if line and len(line) > 5:
                steps.append(line)
        return steps[:4]

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking with Plan-and-Act approach."""
        # Generate plan for ranking
        plan_prompt = f"""[USER REQUEST]
{context}

Multiple restaurants are available for comparison.
Create a plan to compare and rank them against the user's criteria."""

        plan_response = call_llm(plan_prompt, system=SYSTEM_PROMPT_PLAN)

        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request."
        else:
            instruction = f"Select the TOP {k} restaurants that best match, ranked from best to worst."

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
