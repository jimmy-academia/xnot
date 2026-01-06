#!/usr/bin/env python3
"""Program of Thoughts (PoT) method for restaurant recommendation.

Reference: Program of Thoughts Prompting: Disentangling Computation from Reasoning
Chen et al., TMLR 2023
https://arxiv.org/abs/2211.12588

Approach:
Generate Python code with semantic variable bindings that
disentangle reasoning from computation.
"""

import re
import json
from typing import Any

from .base import BaseMethod
from utils.llm import call_llm


SYSTEM_PROMPT = """You are a code generator for restaurant recommendation.
Generate Python code that analyzes a restaurant and determines if it matches a user's request.

IMPORTANT RULES:
1. Use semantic variable names that describe what each value represents
2. Follow the 4-step pattern: Extract -> Analyze -> Combine -> Decide
3. The final variable MUST be named `recommendation` with value:
   - 1 (recommend)
   - 0 (neutral/unclear)
   - -1 (not recommend)

Available variables:
- `restaurant`: dict with keys like 'attributes', 'hours', 'item_data' (reviews)
- `user_request`: string describing what the user wants

Output ONLY the Python code block, no explanations."""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurant for a user's request.
Analyze each restaurant's attributes and reviews against the user's criteria.
Output ONLY the restaurant number (1-N), nothing else."""

# Safe builtins for code execution
SAFE_BUILTINS = {
    'len': len,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'sorted': sorted,
    'reversed': reversed,
    'any': any,
    'all': all,
    'filter': filter,
    'map': map,
    'isinstance': isinstance,
    'True': True,
    'False': False,
    'None': None,
}


class ProgramOfThoughts(BaseMethod):
    """Program of Thoughts prompting method.

    Generates Python code with semantic variable bindings to analyze
    restaurant data and produce recommendations.
    """

    name = "pot"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: Any) -> int:
        """Evaluate restaurant recommendation using code generation.

        Args:
            query: User request text
            context: Restaurant data (dict or string)
        """
        # Build prompt for code generation
        prompt = self._build_prompt(query, context)

        # Get generated code from LLM
        response = call_llm(prompt, system=SYSTEM_PROMPT)

        # Extract code block
        code = self._extract_code(response)

        # Execute code and get result
        result = self._execute_code(code, query, context)

        return result

    def _build_prompt(self, query: str, context: Any) -> str:
        """Build prompt for code generation.

        Args:
            query: User request text
            context: Restaurant data (dict or string)
        """
        # Convert context (restaurant data) to string representation if dict
        if isinstance(context, dict):
            context_str = json.dumps(context, indent=2, default=str)
        else:
            context_str = str(context)

        return f"""Generate Python code to analyze this restaurant for the user's request.

[RESTAURANT DATA]
```python
restaurant = {context_str}
```

[USER REQUEST]
user_request = "{query}"

Generate code following the 4-step pattern:
# Step 1: Extract relevant attributes (semantic binding)
# Step 2: Analyze reviews for user preferences
# Step 3: Combine evidence
# Step 4: Final decision

The code must set `recommendation` to 1, 0, or -1."""

    def _extract_code(self, response: str) -> str:
        """Extract Python code block from LLM response."""
        # Try to find code block with ```python
        pattern = r'```python\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic code block
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Assume entire response is code
        return response.strip()

    def _execute_code(self, code: str, query: str, context: Any) -> int:
        """Execute generated code safely and extract result.

        Args:
            query: User request text
            context: Restaurant data
        """
        try:
            # Set up execution environment
            local_vars = {
                'restaurant': context,
                'user_request': query,
            }
            global_vars = {'__builtins__': SAFE_BUILTINS}

            # Execute the code
            exec(code, global_vars, local_vars)

            # Extract recommendation
            recommendation = local_vars.get('recommendation', 0)

            # Normalize to -1, 0, or 1
            if isinstance(recommendation, (int, float)):
                if recommendation > 0:
                    return 1
                elif recommendation < 0:
                    return -1
                return 0

            return 0

        except Exception as e:
            # Log error if debug mode
            import os
            if os.environ.get("KNOT_DEBUG", "0") == "1":
                print(f"[PoT] Code execution error: {e}")
                print(f"[PoT] Code:\n{code}")
            return 0

    # --- Ranking Methods ---

    def _build_ranking_prompt(self, query: str, context: str, k: int = 1) -> str:
        """Build prompt for ranking multiple restaurants.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        if k == 1:
            instruction = "Which restaurant (by number) BEST matches the user's request?\nOutput ONLY the restaurant number, nothing else."
        else:
            instruction = f"Which {k} restaurants (by number) best match the user's request?\nOutput ONLY the {k} numbers separated by commas, best match first."

        return f"""Analyze these restaurants using Program of Thoughts reasoning.

Think step by step:
1. Extract key attributes from each restaurant
2. Compare against the user's specific requirements
3. Score each restaurant on how well it matches

[RESTAURANTS]
{context}

[USER REQUEST]
{query}

{instruction}"""

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        prompt = self._build_ranking_prompt(query, context, k)
        return call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
