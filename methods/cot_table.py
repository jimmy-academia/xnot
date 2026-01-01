#!/usr/bin/env python3
"""Chain-of-Table method for restaurant recommendation.

Reference: Wang et al. "Chain-of-Table: Evolving Tables in the Reasoning Chain" ICLR 2024
https://arxiv.org/abs/2401.04398

Key idea: Uses tables as intermediate "thoughts" - LLM iteratively applies
table operations to reason about data.
"""

import json
import re
from typing import Any, List, Dict

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


# Available table operations
OPERATIONS = """Available operations:
1. f_add_column(name, expression) - Add new column based on expression
   Example: f_add_column("is_positive", "stars >= 4")

2. f_select_row(condition) - Keep only rows matching condition
   Example: f_select_row("stars >= 4")

3. f_select_column(columns) - Keep only specified columns
   Example: f_select_column(["review", "stars"])

4. f_group_by(column) - Group rows by column value
   Example: f_group_by("is_positive")

5. f_sort_by(column, desc=False) - Sort table by column
   Example: f_sort_by("stars", desc=True)

6. f_count() - Count rows in current table
   Example: f_count()

7. f_aggregate(column, func) - Aggregate column (sum, avg, min, max)
   Example: f_aggregate("stars", "avg")"""


SYSTEM_PROMPT = f"""You are analyzing restaurant data using table operations.

You will see a table of restaurant reviews and attributes.
Your task: iteratively transform the table to answer whether this restaurant
matches the user's request.

{OPERATIONS}

At each step:
1. Examine the current table state
2. Decide which operation to apply next
3. Output the operation in the format: OPERATION: <operation_call>
4. After sufficient analysis, output: ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)

Keep reasoning focused and use 2-5 operations before answering."""


STEP_PROMPT = """Current table state:
{table}

User request: {context}

Previous operations: {history}

What operation should be applied next? Or if you have enough information, provide the final answer.
Output either:
- OPERATION: <operation_call>
- ANSWER: 1, 0, or -1"""


class ChainOfTable(BaseMethod):
    """Chain-of-Table method using iterative table transformations."""

    name = "cot_table"

    def __init__(self, run_dir: str = None, defense: bool = False, max_steps: int = 5, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)
        self.max_steps = max_steps

    def evaluate(self, query: Any, context: str) -> int:
        """Evaluate using iterative table operations."""
        import os
        debug = os.environ.get("KNOT_DEBUG", "0") == "1"

        # Convert query to table format
        if isinstance(query, str):
            try:
                data = json.loads(query)
            except json.JSONDecodeError:
                data = {"raw_text": query}
        else:
            data = query

        # Initialize table from restaurant data
        table = self._create_table(data)
        history = []

        # Iterative reasoning loop
        for step in range(self.max_steps):
            # Format current table for prompt
            table_str = self._format_table(table)
            history_str = " -> ".join(history) if history else "None"

            prompt = STEP_PROMPT.format(
                table=table_str,
                context=context,
                history=history_str
            )

            response = call_llm(prompt, system=SYSTEM_PROMPT)

            if debug:
                print(f"[CoT-Table] Step {step + 1}: {response[:200]}")

            # Check for final answer
            if "ANSWER:" in response.upper():
                return parse_final_answer(response)

            # Parse and execute operation
            op_match = re.search(r'OPERATION:\s*(.+)', response, re.IGNORECASE)
            if op_match:
                operation = op_match.group(1).strip()
                history.append(operation)
                table = self._execute_operation(table, operation, debug)
            else:
                # No valid operation, try to get answer
                break

        # If we exhausted steps without answer, make final call
        final_prompt = f"""Based on the table analysis:
{self._format_table(table)}

Operations performed: {' -> '.join(history)}

User request: {context}

Provide final recommendation: ANSWER: 1 (recommend), 0 (neutral), or -1 (not recommend)"""

        response = call_llm(final_prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    def _create_table(self, data: dict) -> List[Dict]:
        """Convert restaurant dict to table (list of row dicts)."""
        table = []

        # Get restaurant-level attributes
        attrs = data.get('attributes', {})
        name = data.get('item_name', 'Unknown')
        categories = data.get('categories', [])

        # Create one row per review
        reviews = data.get('item_data', [])
        if not reviews:
            # No reviews - create single row with attributes only
            return [{
                'restaurant': name,
                'categories': ', '.join(categories) if categories else '',
                'noise_level': attrs.get('NoiseLevel', 'unknown'),
                'wifi': attrs.get('WiFi', 'unknown'),
                'price_range': attrs.get('RestaurantsPriceRange2', 'unknown'),
                'review': '',
                'stars': 0,
                'date': ''
            }]

        for i, review in enumerate(reviews[:10]):  # Limit to 10 reviews
            row = {
                'review_id': i + 1,
                'restaurant': name,
                'noise_level': attrs.get('NoiseLevel', 'unknown'),
                'wifi': attrs.get('WiFi', 'unknown'),
                'price_range': attrs.get('RestaurantsPriceRange2', 'unknown'),
                'review': review.get('review', '')[:200],  # Truncate long reviews
                'stars': review.get('stars', 0),
                'date': review.get('date', '')
            }
            table.append(row)

        return table

    def _format_table(self, table: List[Dict], max_rows: int = 5) -> str:
        """Format table as readable string for LLM."""
        if not table:
            return "(empty table)"

        # Get columns
        columns = list(table[0].keys())

        # Build header
        header = " | ".join(columns)
        separator = "-" * len(header)

        # Build rows (limit for token efficiency)
        rows = []
        for row in table[:max_rows]:
            row_str = " | ".join(str(row.get(c, ''))[:30] for c in columns)
            rows.append(row_str)

        if len(table) > max_rows:
            rows.append(f"... ({len(table) - max_rows} more rows)")

        return f"{header}\n{separator}\n" + "\n".join(rows)

    def _execute_operation(self, table: List[Dict], operation: str, debug: bool = False) -> List[Dict]:
        """Execute a table operation and return modified table."""
        try:
            # Parse operation
            op_lower = operation.lower()

            # f_add_column
            if 'f_add_column' in op_lower:
                match = re.search(r'f_add_column\s*\(\s*["\'](\w+)["\']\s*,\s*["\'](.+?)["\']\s*\)', operation, re.IGNORECASE)
                if match:
                    col_name, expr = match.groups()
                    return self._add_column(table, col_name, expr)

            # f_select_row
            if 'f_select_row' in op_lower:
                match = re.search(r'f_select_row\s*\(\s*["\'](.+?)["\']\s*\)', operation, re.IGNORECASE)
                if match:
                    condition = match.group(1)
                    return self._select_row(table, condition)

            # f_select_column
            if 'f_select_column' in op_lower:
                match = re.search(r'f_select_column\s*\(\s*\[(.+?)\]\s*\)', operation, re.IGNORECASE)
                if match:
                    cols_str = match.group(1)
                    columns = [c.strip().strip('"\'') for c in cols_str.split(',')]
                    return self._select_column(table, columns)

            # f_sort_by
            if 'f_sort_by' in op_lower:
                match = re.search(r'f_sort_by\s*\(\s*["\'](\w+)["\']\s*(?:,\s*(True|False))?\s*\)', operation, re.IGNORECASE)
                if match:
                    col = match.group(1)
                    desc = match.group(2) and match.group(2).lower() == 'true'
                    return self._sort_by(table, col, desc)

            if debug:
                print(f"[CoT-Table] Could not parse operation: {operation}")

        except Exception as e:
            if debug:
                print(f"[CoT-Table] Operation error: {e}")

        return table  # Return unchanged if operation fails

    def _add_column(self, table: List[Dict], col_name: str, expr: str) -> List[Dict]:
        """Add a computed column to table."""
        for row in table:
            try:
                # Simple expression evaluation with row context
                # Handle common patterns
                if '>=' in expr:
                    parts = expr.split('>=')
                    col, val = parts[0].strip(), float(parts[1].strip())
                    row[col_name] = row.get(col, 0) >= val
                elif '<=' in expr:
                    parts = expr.split('<=')
                    col, val = parts[0].strip(), float(parts[1].strip())
                    row[col_name] = row.get(col, 0) <= val
                elif '==' in expr:
                    parts = expr.split('==')
                    col, val = parts[0].strip(), parts[1].strip().strip('"\'')
                    row[col_name] = str(row.get(col, '')) == val
                elif 'contains' in expr.lower():
                    match = re.search(r'contains\s*\(\s*(\w+)\s*,\s*["\'](.+?)["\']\s*\)', expr, re.IGNORECASE)
                    if match:
                        col, substr = match.groups()
                        row[col_name] = substr.lower() in str(row.get(col, '')).lower()
                else:
                    row[col_name] = expr  # Just set literal value
            except Exception:
                row[col_name] = None
        return table

    def _select_row(self, table: List[Dict], condition: str) -> List[Dict]:
        """Filter rows based on condition."""
        result = []
        for row in table:
            try:
                # Evaluate condition
                if '>=' in condition:
                    parts = condition.split('>=')
                    col, val = parts[0].strip(), float(parts[1].strip())
                    if row.get(col, 0) >= val:
                        result.append(row)
                elif '<=' in condition:
                    parts = condition.split('<=')
                    col, val = parts[0].strip(), float(parts[1].strip())
                    if row.get(col, 0) <= val:
                        result.append(row)
                elif '==' in condition:
                    parts = condition.split('==')
                    col, val = parts[0].strip(), parts[1].strip().strip('"\'')
                    if str(row.get(col, '')) == val:
                        result.append(row)
                elif 'True' in condition or 'true' in condition:
                    col = condition.replace('==', '').replace('True', '').replace('true', '').strip()
                    if row.get(col):
                        result.append(row)
                else:
                    result.append(row)  # If can't parse, keep row
            except Exception:
                result.append(row)
        return result if result else table  # Don't return empty table

    def _select_column(self, table: List[Dict], columns: List[str]) -> List[Dict]:
        """Keep only specified columns."""
        return [{k: v for k, v in row.items() if k in columns} for row in table]

    def _sort_by(self, table: List[Dict], column: str, desc: bool = False) -> List[Dict]:
        """Sort table by column."""
        try:
            return sorted(table, key=lambda x: x.get(column, 0), reverse=desc)
        except Exception:
            return table

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Evaluate ranking task."""
        if k == 1:
            instruction = "Which restaurant (by number) BEST matches the user's request?\nOutput ONLY the restaurant number, nothing else."
        else:
            instruction = f"Which {k} restaurants (by number) best match the user's request?\nOutput ONLY the {k} numbers separated by commas, best match first."

        prompt = f"""Analyze these restaurants using table-based reasoning.

Think of each restaurant as a row in a table with columns:
- attributes (NoiseLevel, WiFi, etc.)
- hours (operating hours)
- reviews (from item_data)

Compare each restaurant's table data against the user's criteria.

[RESTAURANTS]
{query}

[USER REQUEST]
{context}

{instruction}"""

        system = """You are selecting the best restaurant for a user's request.
Analyze each restaurant's attributes and reviews against the user's criteria.
Output ONLY the restaurant number (1-N), nothing else."""

        return call_llm(prompt, system=system)
