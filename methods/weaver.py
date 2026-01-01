#!/usr/bin/env python3
"""Weaver-style method for restaurant recommendation.

Reference: "Weaver: Interweaving SQL and LLM for Table Reasoning" EMNLP 2025
https://arxiv.org/abs/2505.18961

Key idea: Dynamically combine SQL-like operations (pandas) with LLM semantic
processing to handle both structured and unstructured data in tables.

This is a simplified implementation using pandas instead of MySQL.
"""

import json
import re
import pandas as pd
from typing import Any, List, Dict, Optional

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
- LLM steps create new columns that can be used in subsequent SQL steps
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
Example: df.groupby('noise_level').size()
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

    def evaluate(self, query: Any, context: str) -> int:
        """Evaluate using Weaver-style SQL+LLM hybrid reasoning."""
        import os
        debug = os.environ.get("KNOT_DEBUG", "0") == "1"

        # Convert query to DataFrame
        if isinstance(query, str):
            try:
                data = json.loads(query)
            except json.JSONDecodeError:
                data = {"raw_text": query}
        else:
            data = query

        df = self._create_dataframe(data)

        if debug:
            print(f"[Weaver] Initial table shape: {df.shape}")
            print(f"[Weaver] Columns: {list(df.columns)}")

        # Step 1: Generate execution plan
        plan = self._generate_plan(df, context)

        if debug:
            print(f"[Weaver] Plan:\n{plan}")

        # Step 2: Execute plan steps
        df = self._execute_plan(df, plan, context, debug)

        # Step 3: Extract answer
        answer = self._extract_answer(df, context)

        return parse_final_answer(answer)

    def _create_dataframe(self, data: dict) -> pd.DataFrame:
        """Convert restaurant dict to pandas DataFrame."""
        rows = []

        attrs = data.get('attributes', {})
        name = data.get('item_name', 'Unknown')
        categories = data.get('categories', [])

        reviews = data.get('item_data', [])
        if not reviews:
            # No reviews - single row
            return pd.DataFrame([{
                'restaurant': name,
                'categories': ', '.join(categories) if categories else '',
                'noise_level': attrs.get('NoiseLevel', 'unknown'),
                'wifi': attrs.get('WiFi', 'unknown'),
                'price_range': attrs.get('RestaurantsPriceRange2', 'unknown'),
                'review': '',
                'stars': 0
            }])

        for i, review in enumerate(reviews[:10]):  # Limit reviews
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

    def _generate_plan(self, df: pd.DataFrame, context: str) -> str:
        """Generate execution plan."""
        # Create table summary for planner
        columns = list(df.columns)
        sample = df.head(3).to_string(index=False)

        prompt = f"""TABLE COLUMNS: {columns}
SAMPLE DATA:
{sample}

QUESTION: Does this restaurant match the user's request? User wants: {context}

Create a step-by-step plan to analyze this table and answer the question."""

        return call_llm(prompt, system=PLANNER_PROMPT)

    def _execute_plan(self, df: pd.DataFrame, plan: str, context: str, debug: bool = False) -> pd.DataFrame:
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

            # Match "Step N: SQL: ..." or "Step N: LLM: ..."
            match = re.search(r'(?:Step\s*\d+[:\s]*)?(?:(SQL|LLM)[:\s]+)(.+)', line, re.IGNORECASE)
            if match:
                step_type = match.group(1).upper()
                instruction = match.group(2).strip()
                steps.append((step_type, instruction))

        return steps

    def _execute_sql_step(self, df: pd.DataFrame, instruction: str, debug: bool = False) -> pd.DataFrame:
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
            # Safe execution with limited namespace
            namespace = {'df': df, 'pd': pd}
            result = eval(code, namespace)

            if isinstance(result, pd.DataFrame):
                if len(result) > 0:
                    return result
                # Don't return empty DataFrame
                if debug:
                    print(f"[Weaver] SQL returned empty, keeping original")
                return df
            elif isinstance(result, pd.Series):
                # Series result - might be aggregation, add as column
                df['_result'] = result.values[0] if len(result) == 1 else str(result.to_dict())
                return df
            else:
                # Scalar result
                df['_result'] = result
                return df

        except Exception as e:
            if debug:
                print(f"[Weaver] SQL execution error: {e}")
                print(f"[Weaver] Code: {code}")
            return df

    def _execute_llm_step(self, df: pd.DataFrame, instruction: str, debug: bool = False) -> pd.DataFrame:
        """Execute LLM step - semantic processing on column."""
        # Parse instruction to find column and new column name
        # Format: "analyze X -> new_col" or "classify X for Y"
        match = re.search(r'(?:analyze|classify|check|evaluate)\s+["\']?(\w+)["\']?\s*(?:->|for|to create)\s*["\']?(\w+)["\']?', instruction, re.IGNORECASE)

        if match:
            source_col = match.group(1)
            new_col = match.group(2)
        else:
            # Default to review column
            source_col = 'review'
            new_col = 'llm_result'

        if source_col not in df.columns:
            source_col = 'review'  # Fallback

        # Process in batches
        values = df[source_col].tolist()
        values_str = '\n'.join([f"{i+1}. {str(v)[:200]}" for i, v in enumerate(values)])

        prompt = EXECUTOR_LLM_PROMPT.format(
            column=source_col,
            purpose=instruction,
            values=values_str
        )

        response = call_llm(prompt, system="Output one result per line. Be concise.")

        # Parse results
        results = response.strip().split('\n')
        # Clean results
        results = [r.strip().lstrip('0123456789.-) ') for r in results if r.strip()]

        # Pad if needed
        while len(results) < len(df):
            results.append('unknown')

        df[new_col] = results[:len(df)]

        return df

    def _extract_answer(self, df: pd.DataFrame, context: str) -> str:
        """Extract final answer from processed table."""
        # Summarize table for answer extraction
        if len(df) > 5:
            table_str = df.head(5).to_string(index=False) + f"\n... ({len(df)-5} more rows)"
        else:
            table_str = df.to_string(index=False)

        prompt = ANSWER_PROMPT.format(
            table=table_str,
            question=f"Should this restaurant be recommended? User wants: {context}"
        )

        return call_llm(prompt, system="Provide a recommendation: ANSWER: 1, 0, or -1")

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task."""
        prompt = f"""Analyze these restaurants and select the best match.

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

Which restaurant (by index) best matches? Output: ANSWER: <index>"""

        return call_llm(prompt, system=PLANNER_PROMPT)
