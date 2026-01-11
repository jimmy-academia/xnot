"""
Phase 2 v2: General-Purpose Formula Seed Interpreter

Executes the Formula Seed produced by Phase 1 against restaurant data.
This interpreter has NO task-specific logic - it only understands:
1. Filtering reviews by keywords
2. Extracting semantic signals via LLM
3. Aggregating extractions (count, sum, max, min)
4. Computing formulas (arithmetic, conditionals, lookups)

The interpreter doesn't know what "incident" or "allergy" or "trust" means.
It just executes what the Formula Seed specifies.
"""

import json
import re
import math
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm_async


@dataclass
class Extraction:
    """Result of extracting signals from one review."""
    review_id: str
    fields: Dict[str, Any]  # extracted semantic fields
    meta: Dict[str, Any]    # review metadata (stars, year, useful, etc.)


@dataclass
class ExecutionContext:
    """Context for formula execution."""
    extractions: List[Extraction] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)  # computed values
    filters: Dict[str, Dict] = field(default_factory=dict)  # defined filters
    restaurant_context: Dict[str, Any] = field(default_factory=dict)  # restaurant info


class FormulaSeedInterpreter:
    """
    General-purpose interpreter for Formula Seeds.

    Understands the structure produced by Phase 1 LLM:
    - filtering: how to find relevant reviews
    - per_review_extraction_schema: what to extract from each review
    - aggregation_definitions: how to count/aggregate
    - external_data_and_lookup: lookup tables
    - calculation_steps: formulas to compute
    - output_specification: what to return
    """

    def __init__(self, seed: Dict[str, Any], verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.ctx = ExecutionContext()

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        restaurant: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the Formula Seed against restaurant data.

        Args:
            reviews: List of review dicts with text, stars, date, etc.
            restaurant: Restaurant dict with categories, name, etc.

        Returns:
            Dict of computed output values
        """
        self.ctx = ExecutionContext()
        self.ctx.restaurant_context = restaurant

        # Step 1: Filter relevant reviews
        relevant_reviews = self._filter_reviews(reviews)
        if self.verbose:
            print(f"  Filtered: {len(relevant_reviews)}/{len(reviews)} reviews relevant")

        # Step 2: Extract signals from each relevant review
        if relevant_reviews:
            self.ctx.extractions = await self._extract_signals(relevant_reviews)
            if self.verbose:
                print(f"  Extracted signals from {len(self.ctx.extractions)} reviews")

        # Step 3: Compute aggregations
        self._compute_aggregations()

        # Step 4: Execute calculation steps in order
        self._execute_calculations()

        # Step 5: Return output values
        return self._get_output()

    def _filter_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter reviews based on relevance criteria from seed."""
        filtering = self.seed.get("filtering", {})
        detectors = filtering.get("relevance_detectors", [])

        if not detectors:
            # No filtering specified, return all
            return reviews

        # Collect all patterns
        patterns = []
        for detector in detectors:
            detector_patterns = detector.get("patterns", [])
            case_insensitive = detector.get("case_insensitive", True)
            patterns.extend([
                (p.replace("*", ".*"), case_insensitive)
                for p in detector_patterns
            ])

        relevant = []
        for review in reviews:
            text = review.get("text", "")
            for pattern, case_insensitive in patterns:
                flags = re.IGNORECASE if case_insensitive else 0
                if re.search(pattern, text, flags):
                    relevant.append(review)
                    break

        return relevant

    async def _extract_signals(self, reviews: List[Dict[str, Any]]) -> List[Extraction]:
        """Extract semantic signals from reviews using LLM."""
        schema = self.seed.get("per_review_extraction_schema", {})
        target_fields = schema.get("target_review_fields", {})

        # Build extraction prompt from schema
        extraction_fields = []
        for field_name, field_spec in target_fields.items():
            if field_spec.get("type") == "enum":
                extraction_fields.append({
                    "name": field_name,
                    "type": "enum",
                    "values": field_spec.get("values", []),
                    "default": field_spec.get("default"),
                    "rules": field_spec.get("extraction_rules", {})
                })

        if not extraction_fields:
            # No extraction needed, just wrap reviews
            return [
                Extraction(
                    review_id=r.get("review_id", str(i)),
                    fields={},
                    meta=self._extract_meta(r)
                )
                for i, r in enumerate(reviews)
            ]

        # Extract in parallel
        tasks = [
            self._extract_single(review, extraction_fields)
            for review in reviews
        ]
        return await asyncio.gather(*tasks)

    async def _extract_single(
        self,
        review: Dict[str, Any],
        fields: List[Dict]
    ) -> Extraction:
        """Extract signals from a single review."""
        # Build prompt
        field_descriptions = []
        for f in fields:
            values_desc = ", ".join(f["values"]) if isinstance(f["values"], list) else str(f["values"])
            field_descriptions.append(f"- {f['name']}: one of [{values_desc}]")
            if f.get("rules"):
                for rule_name, patterns in f["rules"].items():
                    if patterns:
                        field_descriptions.append(f"  {rule_name}: {patterns[:3]}...")

        prompt = f"""Extract the following fields from this review:

{chr(10).join(field_descriptions)}

Review text:
"{review.get('text', '')}"

Return a JSON object with just the field values. Example:
{{"field1": "value1", "field2": "value2"}}

```json
"""

        try:
            response = await call_llm_async(prompt, role="worker")
            extracted = self._parse_json(response)
        except Exception as e:
            if self.verbose:
                print(f"  Warning: extraction failed for review: {e}")
            extracted = {}

        # Apply defaults
        for f in fields:
            if f["name"] not in extracted and f.get("default"):
                extracted[f["name"]] = f["default"]

        return Extraction(
            review_id=review.get("review_id", ""),
            fields=extracted,
            meta=self._extract_meta(review)
        )

    def _extract_meta(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from review."""
        meta = {
            "stars": review.get("stars", 3),
            "useful": review.get("useful_votes", review.get("useful", 0)),
            "year": self._extract_year(review),
        }
        return meta

    def _extract_year(self, review: Dict[str, Any]) -> int:
        """Extract year from review."""
        if "review_year" in review:
            return review["review_year"]
        date = review.get("review_date", review.get("date", ""))
        if date:
            match = re.search(r"(\d{4})", str(date))
            if match:
                return int(match.group(1))
        return 2020  # default

    def _compute_aggregations(self):
        """Compute aggregation values from seed."""
        agg_defs = self.seed.get("aggregation_definitions", {})

        # First pass: compute count-based aggregations
        for name, spec in agg_defs.items():
            if "source" in spec and spec["source"] == "reviews":
                # Count matching extractions
                condition = spec.get("condition", {})
                count = self._count_matching(condition)
                self.ctx.values[name] = count

        # Second pass: compute formula-based aggregations
        for name, spec in agg_defs.items():
            if "formula" in spec:
                formula = spec["formula"]
                try:
                    # Handle special count formulas
                    if "COUNT of reviews" in formula or "count of reviews" in formula.lower():
                        self.ctx.values[name] = len(self.ctx.extractions)
                    # Skip per-item templates (contain review-level variables)
                    elif any(v in formula for v in ["stars", "useful_votes", "review_year"]):
                        # This is a template, not a global formula - skip
                        continue
                    else:
                        # Arithmetic formula referencing other aggregations
                        value = self._evaluate_arithmetic(formula)
                        self.ctx.values[name] = value
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: failed to compute aggregation {name}: {e}")
                    self.ctx.values[name] = 0

    def _count_matching(self, condition: Dict[str, Any]) -> int:
        """Count extractions matching condition."""
        count = 0
        for ext in self.ctx.extractions:
            if self._matches_condition(ext, condition):
                count += 1
        return count

    def _matches_condition(self, ext: Extraction, condition: Dict[str, Any]) -> bool:
        """Check if extraction matches condition."""
        for field, expected in condition.items():
            # Check extraction fields
            actual = ext.fields.get(field)
            if actual is None:
                # Check with different case
                field_lower = field.lower()
                for k, v in ext.fields.items():
                    if k.lower() == field_lower:
                        actual = v
                        break

            if actual is None:
                return False

            # Handle different match types
            if isinstance(expected, dict):
                if "in" in expected:
                    if actual not in expected["in"]:
                        return False
                elif ">=" in expected:
                    if not (actual >= expected[">="]):
                        return False
                elif "<" in expected:
                    if not (actual < expected["<"]):
                        return False
            else:
                if str(actual).lower() != str(expected).lower():
                    return False

        return True

    def _execute_calculations(self):
        """Execute calculation steps in order."""
        calc_steps = self.seed.get("calculation_steps", {})
        order = calc_steps.get("order_of_operations", [])
        definitions = calc_steps.get("definitions", {})

        for step_name in order:
            if step_name in definitions:
                formula = definitions[step_name]
                try:
                    value = self._evaluate_formula(formula, step_name)
                    self.ctx.values[step_name] = value
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: failed to compute {step_name}: {e}")
                    self.ctx.values[step_name] = 0

    def _evaluate_formula(self, formula: str, name: str = "") -> Any:
        """
        Evaluate a formula string.

        Handles:
        - Arithmetic: 1.0 + (N_POSITIVE * 0.1)
        - Conditionals: if X > 0 then Y else Z
        - Functions: clamp(x, min, max), max(a, b), ln(x)
        - References: previously computed values
        """
        # Handle special formulas
        if formula.startswith("count of"):
            return self._evaluate_count_formula(formula)

        if formula.startswith("sum over"):
            return self._evaluate_sum_formula(formula)

        if formula.startswith("max(") and "among" in formula:
            return self._evaluate_max_formula(formula)

        if formula.startswith("derived from"):
            return self._evaluate_lookup(formula, name)

        # Handle semicolon-separated multi-conditional: "X if cond1; Y if cond2; else Z"
        if ";" in formula:
            return self._evaluate_multi_conditional(formula)

        # Handle "if X then Y else Z" format (starts with "if")
        if formula.strip().startswith("if "):
            return self._evaluate_if_then_else(formula)

        # Handle "X if condition else Y" format (conditional in middle)
        if " if " in formula.lower() and " else " in formula.lower():
            return self._evaluate_conditional(formula)

        # Handle clamp function
        if formula.startswith("clamp("):
            return self._evaluate_clamp(formula)

        # Standard arithmetic evaluation
        return self._evaluate_arithmetic(formula)

    def _evaluate_count_formula(self, formula: str) -> int:
        """Evaluate 'count of X where Y' formula."""
        # Parse: "count of incidents with review_year >= 2023"
        count = 0

        # Extract conditions from formula
        conditions = {}
        if "review_year >=" in formula:
            match = re.search(r"review_year >= (\d+)", formula)
            if match:
                year_threshold = int(match.group(1))
                for ext in self.ctx.extractions:
                    if ext.meta.get("year", 0) >= year_threshold:
                        if self._is_incident(ext):
                            count += 1
                return count

        if "review_year <" in formula:
            match = re.search(r"review_year < (\d+)", formula)
            if match:
                year_threshold = int(match.group(1))
                for ext in self.ctx.extractions:
                    if ext.meta.get("year", 0) < year_threshold:
                        if self._is_incident(ext):
                            count += 1
                return count

        if "MENTION_ALLERGY" in formula or "mentions" in formula.lower():
            return len(self.ctx.extractions)

        return count

    def _is_incident(self, ext: Extraction) -> bool:
        """Check if extraction represents an incident (firsthand + severity)."""
        account = ext.fields.get("ACCOUNT_TYPE", ext.fields.get("account_type", "none"))
        severity = ext.fields.get("INCIDENT_SEVERITY", ext.fields.get("incident_severity", "none"))
        return (
            str(account).lower() == "firsthand" and
            str(severity).lower() in ["mild", "moderate", "severe"]
        )

    def _evaluate_sum_formula(self, formula: str) -> float:
        """Evaluate 'sum over X of expr' formula."""
        # Parse: "sum over all incident reviews of ((5 - stars) + ln(useful_votes + 1))"
        total = 0.0

        for ext in self.ctx.extractions:
            if self._is_incident(ext):
                stars = ext.meta.get("stars", 3)
                useful = ext.meta.get("useful", 0)
                weight = (5 - stars) + math.log(useful + 1)
                total += weight

        return total

    def _evaluate_max_formula(self, formula: str) -> Any:
        """Evaluate 'max(field) among X' formula."""
        # Parse: "max(review_year) among incident reviews if N_TOTAL_INCIDENTS > 0 else 2020"

        # Check for conditional
        if " else " in formula:
            parts = formula.split(" else ")
            default = int(parts[1].strip())
        else:
            default = 2020

        values = []
        for ext in self.ctx.extractions:
            if self._is_incident(ext):
                year = ext.meta.get("year", 2020)
                values.append(year)

        if values:
            return max(values)
        return default

    def _evaluate_lookup(self, formula: str, name: str) -> float:
        """Evaluate lookup from external data."""
        # Get lookup table from seed
        external = self.seed.get("external_data_and_lookup", {})

        # Find the right table
        if "CUISINE" in name.upper():
            table = external.get("CUISINE_MODIFIERS", {})
            detection = external.get("CUISINE_MODIFIER_DETECTION", {})

            # Get restaurant categories
            categories = self.ctx.restaurant_context.get(
                "categories",
                self.ctx.restaurant_context.get("restaurant_categories", [])
            )
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",")]

            # Find highest matching modifier
            max_modifier = table.get("default", 1.0)
            method = detection.get("method", "highest_value_from_restaurant_categories")

            if "highest" in method or "max" in method:
                for cat in categories:
                    for cuisine, modifier in table.items():
                        if cuisine.lower() in cat.lower() or cat.lower() in cuisine.lower():
                            if isinstance(modifier, (int, float)) and modifier > max_modifier:
                                max_modifier = modifier

            return max_modifier

        return 1.0

    def _evaluate_multi_conditional(self, formula: str) -> Any:
        """Evaluate semicolon-separated conditionals: 'X if cond1; Y if cond2; else Z'"""
        parts = [p.strip() for p in formula.split(";")]

        for part in parts:
            if part.startswith("else "):
                # Final else clause
                value_str = part[5:].strip()
                return self._evaluate_arithmetic(value_str)

            if " if " in part:
                # "value if condition" format
                match = re.match(r"(.+?)\s+if\s+(.+)", part)
                if match:
                    value_str = match.group(1).strip()
                    condition = match.group(2).strip()
                    if self._evaluate_condition(condition):
                        return self._evaluate_arithmetic(value_str)

        return 0

    def _evaluate_conditional(self, formula: str) -> Any:
        """Evaluate 'X if condition else Y' formula."""
        # Parse: "N_RECENT / N_TOTAL_INCIDENTS if N_TOTAL_INCIDENTS > 0 else 0"
        match = re.match(r"(.+?)\s+if\s+(.+?)\s+else\s+(.+)", formula, re.IGNORECASE)
        if not match:
            return 0

        then_expr = match.group(1).strip()
        condition = match.group(2).strip()
        else_expr = match.group(3).strip()

        # Evaluate condition
        if self._evaluate_condition(condition):
            return self._evaluate_arithmetic(then_expr)
        else:
            return self._evaluate_arithmetic(else_expr)

    def _evaluate_if_then_else(self, formula: str) -> Any:
        """Evaluate 'if X then Y else Z' formula."""
        # Parse: "if FINAL_RISK_SCORE < 4.0 then 'Low Risk' else if FINAL_RISK_SCORE < 8.0 then 'High Risk' else 'Critical Risk'"

        # Use regex to find all condition-result pairs
        # Pattern: if CONDITION then 'RESULT'
        pattern = r"if\s+(.+?)\s+then\s+['\"]([^'\"]+)['\"]"
        matches = list(re.finditer(pattern, formula))

        for match in matches:
            condition = match.group(1).strip()
            result = match.group(2).strip()
            try:
                if self._evaluate_condition(condition):
                    return result
            except Exception:
                continue

        # Find final else value
        else_match = re.search(r"else\s+['\"]([^'\"]+)['\"](?:\s*$|(?!\s*if))", formula)
        if else_match:
            return else_match.group(1).strip()

        return "Unknown"

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a boolean condition."""
        # Handle compound conditions
        if " and " in condition.lower():
            parts = re.split(r"\s+and\s+", condition, flags=re.IGNORECASE)
            return all(self._evaluate_condition(p) for p in parts)

        if " or " in condition.lower():
            parts = re.split(r"\s+or\s+", condition, flags=re.IGNORECASE)
            return any(self._evaluate_condition(p) for p in parts)

        # Parse comparison
        for op in [">=", "<=", ">", "<", "==", "!="]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    left = self._evaluate_arithmetic(parts[0].strip())
                    right = self._evaluate_arithmetic(parts[1].strip())
                    if op == ">=":
                        return left >= right
                    elif op == "<=":
                        return left <= right
                    elif op == ">":
                        return left > right
                    elif op == "<":
                        return left < right
                    elif op == "==":
                        return left == right
                    elif op == "!=":
                        return left != right

        return False

    def _evaluate_clamp(self, formula: str) -> float:
        """Evaluate clamp(value, min, max)."""
        match = re.match(r"clamp\((.+?),\s*(.+?),\s*(.+?)\)", formula)
        if match:
            value = self._evaluate_arithmetic(match.group(1))
            min_val = float(match.group(2))
            max_val = float(match.group(3))
            return max(min_val, min(max_val, value))
        return 0

    def _evaluate_arithmetic(self, expr: str) -> float:
        """Evaluate arithmetic expression with variable substitution."""
        expr = expr.strip()

        # Handle string literals
        if expr.startswith("'") or expr.startswith('"'):
            return expr.strip("'\"")

        # Try to parse as number
        try:
            return float(expr)
        except ValueError:
            pass

        # Substitute variables
        substituted = expr
        for name, value in self.ctx.values.items():
            if isinstance(value, (int, float)):
                # Use word boundary to avoid partial matches
                substituted = re.sub(rf"\b{re.escape(name)}\b", str(value), substituted)

        # Handle functions (only if not already math.*)
        if "math.log" not in substituted:
            substituted = re.sub(r"\bln\(", "math.log(", substituted)
            substituted = re.sub(r"\blog\(", "math.log(", substituted)
        substituted = re.sub(r"\bsqrt\(", "math.sqrt(", substituted)
        substituted = re.sub(r"\babs\(", "abs(", substituted)

        # Evaluate
        try:
            # Safe eval with limited namespace
            result = eval(substituted, {"__builtins__": {}, "math": math, "max": max, "min": min, "abs": abs})
            return float(result)
        except Exception as e:
            if self.verbose:
                print(f"  Warning: failed to evaluate '{expr}' -> '{substituted}': {e}")
            return 0.0

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

        return {}

    def _get_output(self) -> Dict[str, Any]:
        """Return specified output values."""
        output_spec = self.seed.get("output_specification", {})
        reported = output_spec.get("reported_values", [])

        if not reported:
            # Return all computed values
            return dict(self.ctx.values)

        result = {}
        for name in reported:
            if name in self.ctx.values:
                result[name] = self.ctx.values[name]
            else:
                result[name] = None

        return result


async def test_interpreter():
    """Test the interpreter with saved formula seed."""
    # Load formula seed
    seed_path = Path(__file__).parent.parent / "results" / "phase1_v2" / "formula_seed.json"
    with open(seed_path) as f:
        seed = json.load(f)

    print("=" * 70)
    print("PHASE 2 V2: Formula Seed Interpreter Test")
    print("=" * 70)

    # Create test data
    test_reviews = [
        {
            "review_id": "r1",
            "text": "I have a severe peanut allergy and the staff was very accommodating. They asked about my allergies and made sure there was no cross-contamination.",
            "stars": 5,
            "useful_votes": 10,
            "review_date": "2024-06-15"
        },
        {
            "review_id": "r2",
            "text": "My child had an allergic reaction here. They claimed the dish was nut-free but we ended up in the ER. Had to use the epipen.",
            "stars": 1,
            "useful_votes": 25,
            "review_date": "2024-03-20"
        },
        {
            "review_id": "r3",
            "text": "Great Thai food! The pad thai was amazing.",
            "stars": 4,
            "useful_votes": 5,
            "review_date": "2023-11-10"
        },
        {
            "review_id": "r4",
            "text": "I'm worried about nut allergies at this place. The kitchen seems chaotic.",
            "stars": 3,
            "useful_votes": 2,
            "review_date": "2024-01-05"
        }
    ]

    test_restaurant = {
        "name": "Thai Kitchen",
        "categories": ["Thai", "Asian Fusion", "Restaurants"]
    }

    # Create interpreter
    interpreter = FormulaSeedInterpreter(seed, verbose=True)

    # Execute
    print(f"\nExecuting against {len(test_reviews)} reviews...")
    result = await interpreter.execute(test_reviews, test_restaurant)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    for name, value in result.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    return result


if __name__ == "__main__":
    asyncio.run(test_interpreter())
