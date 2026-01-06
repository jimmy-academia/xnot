#!/usr/bin/env python3
"""Weaver-style method for restaurant recommendation.

Reference: "Weaver: Interweaving SQL and LLM for Table Reasoning" EMNLP 2025

Key idea: Dynamically combine SQL-like operations (pandas) with LLM semantic
processing to handle both structured and unstructured data in tables.
"""

import json
import re
from typing import Any, List, Dict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


PLANNER_PROMPT = """You are a table reasoning planner. Given a table and question,
create a step-by-step execution plan.

AVAILABLE OPERATIONS:
1. SQL: Filter, aggregate, sort, select columns (structured operations)
   Format: SQL: <description of what to query/filter>

2. LLM: Semantic analysis, classification, reasoning (unstructured operations)
   Format: LLM: <what to analyze> -> <new_column_name>

RULES:
- Use SQL for: filtering by values, counting, aggregations, sorting
- Use LLM for: sentiment analysis, semantic matching, text classification
- Keep plans focused (2-5 steps max)

OUTPUT FORMAT:
Step 1: <SQL or LLM>: <description>
Step 2: <SQL or LLM>: <description>
...
Final: Answer the question based on the processed table
"""

EXECUTOR_SQL_PROMPT = """Given the table and instruction, write a pandas query.

TABLE COLUMNS: {columns}
SAMPLE DATA:
{sample}

INSTRUCTION: {instruction}

Output ONLY the pandas code (no explanation). Use variable 'df' for the dataframe.
Example: df[df['stars'] >= 4]
"""

EXECUTOR_LLM_PROMPT = """Analyze each row and output a value for the new column.

COLUMN TO ANALYZE: {column}
NEW COLUMN PURPOSE: {purpose}

For each value below, output the result on a new line:
{values}

Output one result per line, in the same order as the input values.
"""

ANSWER_PROMPT = """Based on the processed table below, answer the question.

TABLE:
{table}

QUESTION: {question}

For recommendation: Output ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)
Be concise."""


class Weaver(BaseMethod):
    """Weaver: SQL + LLM hybrid table reasoning."""

    name = "weaver"

    def __init__(self, run_dir: str = None, defense: bool = False, max_steps: int = 5, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)
        self.max_steps = max_steps
        if not HAS_PANDAS:
            raise ImportError("Weaver requires pandas. Install with: pip install pandas")

    def evaluate(self, query: str, context: Any) -> int:
        """Evaluate using Weaver-style SQL+LLM hybrid reasoning.

        Args:
            query: User request text
            context: Restaurant data (dict or string)
        """
        # Convert context (restaurant data) to DataFrame
        if isinstance(context, str):
            try:
                data = json.loads(context)
            except json.JSONDecodeError:
                data = {"raw_text": context}
        else:
            data = context

        df = self._create_dataframe(data)

        if self.verbose:
            print(f"[Weaver] Initial table shape: {df.shape}")
            print(f"[Weaver] Columns: {list(df.columns)}")

        # Step 1: Generate execution plan
        plan = self._generate_plan(df, query)

        if self.verbose:
            print(f"[Weaver] Plan:\n{plan}")

        # Step 2: Execute plan steps
        df = self._execute_plan(df, plan, query, self.verbose)

        # Step 3: Extract answer
        answer = self._extract_answer(df, query)

        return parse_final_answer(answer)

    def _create_dataframe(self, data: dict):
        """Convert restaurant dict to pandas DataFrame."""
        rows = []

        attrs = data.get('attributes', {})
        name = data.get('item_name', 'Unknown')
        categories = data.get('categories', [])

        reviews = data.get('item_data', [])
        if not reviews:
            return pd.DataFrame([{
                'restaurant': name,
                'categories': ', '.join(categories) if categories else '',
                'noise_level': attrs.get('NoiseLevel', 'unknown'),
                'wifi': attrs.get('WiFi', 'unknown'),
                'price_range': attrs.get('RestaurantsPriceRange2', 'unknown'),
                'review': '',
                'stars': 0
            }])

        for i, review in enumerate(reviews[:10]):
            rows.append({
                'review_id': i + 1,
                'restaurant': name,
                'noise_level': attrs.get('NoiseLevel', 'unknown'),
                'wifi': attrs.get('WiFi', 'unknown'),
                'price_range': attrs.get('RestaurantsPriceRange2', 'unknown'),
                'review': review.get('review', '')[:300],
                'stars': review.get('stars', 0),
                'date': review.get('date', '')
            })

        return pd.DataFrame(rows)

    def _generate_plan(self, df, query: str) -> str:
        """Generate execution plan.

        Args:
            query: User request text
        """
        columns = list(df.columns)
        sample = df.head(3).to_string(index=False)

        prompt = f"""TABLE COLUMNS: {columns}
SAMPLE DATA:
{sample}

QUESTION: Does this restaurant match the user's request? User wants: {query}

Create a step-by-step plan to analyze this table and answer the question."""

        return call_llm(prompt, system=PLANNER_PROMPT)

    def _execute_plan(self, df, plan: str, query: str, debug: bool = False):
        """Execute the generated plan."""
        steps = self._parse_plan(plan)

        for i, (step_type, instruction) in enumerate(steps[:self.max_steps]):
            if debug:
                print(f"[Weaver] Executing Step {i+1}: {step_type} - {instruction[:50]}...")

            if step_type == 'SQL':
                df = self._execute_sql_step(df, instruction, debug)
            elif step_type == 'LLM':
                df = self._execute_llm_step(df, instruction, debug)

        return df

    def _parse_plan(self, plan: str) -> List[tuple]:
        """Parse plan into (step_type, instruction) tuples."""
        steps = []
        for line in plan.split('\n'):
            line = line.strip()
            if not line or line.lower().startswith('final'):
                continue

            match = re.search(r'(?:Step\s*\d+[:\s]*)?(?:(SQL|LLM)[:\s]+)(.+)', line, re.IGNORECASE)
            if match:
                step_type = match.group(1).upper()
                instruction = match.group(2).strip()
                steps.append((step_type, instruction))

        return steps

    def _execute_sql_step(self, df, instruction: str, debug: bool = False):
        """Execute SQL-like step using pandas."""
        columns = list(df.columns)
        sample = df.head(2).to_string(index=False)

        prompt = EXECUTOR_SQL_PROMPT.format(
            columns=columns,
            sample=sample,
            instruction=instruction
        )

        code = call_llm(prompt, system="Output ONLY valid pandas code. Use 'df' as the dataframe variable.")

        # Clean code
        code = code.strip()
        if code.startswith('```'):
            code = re.sub(r'^```\w*\n?', '', code)
            code = re.sub(r'\n?```$', '', code)
        code = code.strip()

        try:
            # Validate code - reject dangerous patterns
            DANGEROUS_PATTERNS = ['import ', '__', 'exec(', 'eval(', 'open(', 'os.', 'sys.', 'subprocess']
            code_lower = code.lower()
            for pattern in DANGEROUS_PATTERNS:
                if pattern in code_lower:
                    raise ValueError(f"Unsafe code pattern detected: {pattern}")

            # Execute with restricted namespace (no builtins)
            namespace = {'df': df, 'pd': pd, '__builtins__': {}}
            result = eval(code, namespace)

            if isinstance(result, pd.DataFrame):
                if len(result) > 0:
                    return result
                if debug:
                    print(f"[Weaver] SQL returned empty, keeping original")
                return df
            elif isinstance(result, pd.Series):
                df['_result'] = result.values[0] if len(result) == 1 else str(result.to_dict())
                return df
            else:
                df['_result'] = result
                return df

        except Exception as e:
            if debug:
                print(f"[Weaver] SQL execution error: {e}")
                print(f"[Weaver] Code: {code}")
            return df

    def _execute_llm_step(self, df, instruction: str, debug: bool = False):
        """Execute LLM step - semantic processing on column."""
        match = re.search(r'(?:analyze|classify|check|evaluate)\s+["\']?(\w+)["\']?\s*(?:->|for|to create)\s*["\']?(\w+)["\']?', instruction, re.IGNORECASE)

        if match:
            source_col = match.group(1)
            new_col = match.group(2)
        else:
            source_col = 'review'
            new_col = 'llm_result'

        if source_col not in df.columns:
            source_col = 'review'

        values = df[source_col].tolist()
        values_str = '\n'.join([f"{i+1}. {str(v)[:200]}" for i, v in enumerate(values)])

        prompt = EXECUTOR_LLM_PROMPT.format(
            column=source_col,
            purpose=instruction,
            values=values_str
        )

        response = call_llm(prompt, system="Output one result per line. Be concise.")

        results = response.strip().split('\n')
        results = [r.strip().lstrip('0123456789.-) ') for r in results if r.strip()]

        while len(results) < len(df):
            results.append('unknown')

        df[new_col] = results[:len(df)]

        return df

    def _extract_answer(self, df, query: str) -> str:
        """Extract final answer from processed table.

        Args:
            query: User request text
        """
        if len(df) > 5:
            table_str = df.head(5).to_string(index=False) + f"\n... ({len(df)-5} more rows)"
        else:
            table_str = df.to_string(index=False)

        prompt = ANSWER_PROMPT.format(
            table=table_str,
            question=f"Should this restaurant be recommended? User wants: {query}"
        )

        return call_llm(prompt, system="Provide a recommendation: ANSWER: 1, 0, or -1")

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        prompt = f"""Analyze these restaurants and select the best match.

[RESTAURANTS]
{context}

[USER REQUEST]
{query}

Which restaurant (by index) best matches? Output: ANSWER: <index>"""

        return call_llm(prompt, system=PLANNER_PROMPT)
