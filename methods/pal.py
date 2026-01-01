#!/usr/bin/env python3
"""PAL (Program-Aided Language) method for restaurant recommendation.

Reference: Gao et al. "PAL: Program-Aided Language Models" ICML 2023
https://arxiv.org/abs/2211.10435

Key idea: LLM generates Python code with reasoning steps;
computation offloaded to Python interpreter.
"""

import json
import re
from typing import Any

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT = """You are a Python code generator for restaurant recommendation.

Given restaurant data (as a Python dict) and a user request, generate a Python function
that analyzes the data and returns a recommendation.

IMPORTANT RULES:
1. Output ONLY valid Python code - no markdown, no explanation
2. Define a function: def evaluate(data: dict, request: str) -> int
3. Return exactly: 1 (recommend), 0 (neutral), or -1 (not recommend)
4. Use descriptive variable names that explain your reasoning
5. You can use: json, re, math, collections, statistics (already imported)
6. Access data using: data['attributes'], data['hours'], data['item_data'], etc.
7. Reviews are in data['item_data'] - each has 'review', 'stars', 'date' keys

Example structure:
def evaluate(data: dict, request: str) -> int:
    # Step 1: Extract relevant attributes
    noise_level = data['attributes'].get('NoiseLevel', 'unknown')

    # Step 2: Analyze reviews
    reviews = data.get('item_data', [])
    positive_count = sum(1 for r in reviews if r.get('stars', 0) >= 4)

    # Step 3: Check specific criteria from request
    # ... (analyze based on request keywords)

    # Step 4: Make decision
    if meets_criteria:
        return 1
    elif clearly_fails:
        return -1
    return 0"""


CODE_EXTRACTION_PATTERN = re.compile(
    r'```python\s*(.*?)\s*```|'  # Markdown code block
    r'```\s*(.*?)\s*```|'        # Generic code block
    r'(def evaluate\(.*)',       # Raw function start
    re.DOTALL
)


class ProgramAidedLanguage(BaseMethod):
    """PAL: Program-Aided Language Models method.

    Generates Python code to evaluate restaurant recommendations,
    then executes the code to get the final answer.
    """

    name = "pal"

    def __init__(self, run_dir: str = None, defense: bool = False, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)

    def evaluate(self, query: Any, context: str) -> int:
        """Evaluate restaurant recommendation using generated Python code.

        Args:
            query: Restaurant data (dict preferred, string also supported)
            context: User request text

        Returns:
            int: 1 (recommend), 0 (neutral), or -1 (not recommend)
        """
        # Ensure query is dict for code generation
        if isinstance(query, str):
            # Try to parse as JSON, otherwise wrap in dict
            try:
                data = json.loads(query)
            except json.JSONDecodeError:
                data = {"raw_text": query}
        else:
            data = query

        # Step 1: Generate Python code
        code = self._generate_code(data, context)

        # Step 2: Execute the code
        result = self._execute_code(code, data, context)

        return result

    def _generate_code(self, data: dict, context: str) -> str:
        """Generate Python code to evaluate the restaurant."""
        # Truncate data for prompt (avoid token limits)
        data_preview = self._truncate_data(data)

        prompt = f"""Generate Python code to evaluate this restaurant for the user's request.

[RESTAURANT DATA]
data = {json.dumps(data_preview, indent=2)}

[USER REQUEST]
{context}

Generate a Python function `def evaluate(data: dict, request: str) -> int` that:
1. Analyzes the restaurant data against the user's specific needs
2. Returns 1 (recommend), 0 (neutral), or -1 (not recommend)

Output ONLY the Python code:"""

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return self._extract_code(response)

    def _truncate_data(self, data: dict, max_reviews: int = 5) -> dict:
        """Truncate data to avoid token limits while preserving structure."""
        truncated = dict(data)
        if 'item_data' in truncated and len(truncated['item_data']) > max_reviews:
            truncated['item_data'] = truncated['item_data'][:max_reviews]
            truncated['_truncated'] = f"Showing {max_reviews} of {len(data['item_data'])} reviews"
        return truncated

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code in markdown blocks first
        match = CODE_EXTRACTION_PATTERN.search(response)
        if match:
            code = match.group(1) or match.group(2) or match.group(3)
            if code:
                return code.strip()

        # If no markdown, assume entire response is code
        # Remove any leading/trailing explanation
        lines = response.strip().split('\n')
        code_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith('def evaluate'):
                in_function = True
            if in_function:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # Last resort: return entire response
        return response.strip()

    def _execute_code(self, code: str, data: dict, context: str) -> int:
        """Safely execute generated Python code.

        Args:
            code: Python code containing evaluate() function
            data: Restaurant data dict
            context: User request text

        Returns:
            int: Result from evaluate(), defaults to 0 on error
        """
        # Create restricted namespace with allowed imports
        import math
        import statistics
        import os
        from collections import Counter, defaultdict

        namespace = {
            'json': json,
            're': re,
            'math': math,
            'statistics': statistics,
            'Counter': Counter,
            'defaultdict': defaultdict,
            'sum': sum,
            'len': len,
            'max': max,
            'min': min,
            'abs': abs,
            'any': any,
            'all': all,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'True': True,
            'False': False,
            'None': None,
        }

        debug = os.environ.get("KNOT_DEBUG", "0") == "1"

        try:
            # Execute code to define function
            exec(code, namespace)

            # Get the evaluate function
            evaluate_fn = namespace.get('evaluate')
            if evaluate_fn is None:
                # Try alternate function names
                for name in ['eval', 'analyze', 'recommend']:
                    if name in namespace and callable(namespace[name]):
                        evaluate_fn = namespace[name]
                        break

            if evaluate_fn is None:
                if debug:
                    print(f"[PAL] No evaluate function found in generated code")
                return 0  # No function found

            # Execute the function
            result = evaluate_fn(data, context)

            # Normalize result
            if isinstance(result, (int, float)):
                if result > 0:
                    return 1
                elif result < 0:
                    return -1
                return 0

            # Try parsing as string
            return parse_final_answer(str(result))

        except Exception as e:
            # Log error if debug mode
            if debug:
                print(f"[PAL] Code execution error: {e}")
                print(f"[PAL] Code:\n{code}")
            return 0  # Default to neutral on error

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task - select best from multiple items."""
        if k == 1:
            instruction = "Which restaurant (by number) BEST matches the user's request?\nOutput ONLY the restaurant number, nothing else."
        else:
            instruction = f"Which {k} restaurants (by number) best match the user's request?\nOutput ONLY the {k} numbers separated by commas, best match first."

        prompt = f"""Analyze these restaurants using program-aided reasoning.

Think step by step like you're writing code:
1. Parse each restaurant's key attributes
2. Check which attributes match user requirements
3. Compare and select the best match

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

{instruction}"""

        system = """You are selecting the best restaurant for a user's request.
Analyze each restaurant's attributes and reviews against the user's criteria.
Output ONLY the restaurant number (1-N), nothing else."""

        return call_llm(prompt, system=system)
