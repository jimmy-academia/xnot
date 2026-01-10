"""
Phase 1: Formula Parser

Parses a task formula (like TASK_G1_PROMPT) into a structured execution schema.

Input: Task prompt (human-readable formula specification)
Output: ExecutionSchema (structured seed for Phase 2)

The schema captures:
1. Extraction spec - what signals to extract per review
2. Filter spec - how to find relevant reviews (keywords)
3. Aggregation rules - how to count/aggregate extractions
4. Computation graph - formulas in dependency order
5. Lookup tables - static mappings
6. Output spec - final output fields and verdict rules
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm, call_llm_async


@dataclass
class ExtractionField:
    """A field to extract from each review."""
    name: str  # e.g., "incident_severity"
    field_type: str  # "enum", "boolean", "number", "string"
    values: List[str] = field(default_factory=list)  # For enum type
    description: str = ""


@dataclass
class AggregationRule:
    """Rule for aggregating extracted values."""
    name: str  # e.g., "n_mild"
    formula: str  # e.g., "COUNT(incident_severity='mild' AND account_type='firsthand')"
    description: str = ""


@dataclass
class ComputationStep:
    """A derived value computation."""
    name: str  # e.g., "incident_score"
    formula: str  # e.g., "n_mild * 2 + n_moderate * 5 + n_severe * 15"
    depends_on: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class LookupTable:
    """Static lookup mapping."""
    name: str  # e.g., "cuisine_modifier"
    mapping: Dict[str, float] = field(default_factory=dict)
    default: float = 1.0
    source_field: str = ""  # e.g., "categories"


@dataclass
class VerdictRule:
    """Rule for determining final verdict."""
    source_field: str  # e.g., "final_risk_score"
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"max": 4.0, "verdict": "Low Risk"}, {"max": 8.0, "verdict": "High Risk"}, ...]


@dataclass
class ExecutionSchema:
    """
    Complete execution schema produced by Phase 1.

    This is the "seed" that Phase 2 expands into LWT script.
    """
    task_name: str

    # What to extract from each review
    extraction_fields: List[ExtractionField] = field(default_factory=list)

    # Review metadata to include (always available)
    review_metadata: List[str] = field(default_factory=lambda: ["stars", "date", "useful"])

    # Keywords to filter relevant reviews (optional)
    filter_keywords: List[str] = field(default_factory=list)

    # Aggregation rules (counting/summing extractions)
    aggregations: List[AggregationRule] = field(default_factory=list)

    # Derived computations (in dependency order)
    computations: List[ComputationStep] = field(default_factory=list)

    # Static lookup tables
    lookups: List[LookupTable] = field(default_factory=list)

    # Final output fields
    output_fields: List[str] = field(default_factory=list)

    # Verdict determination
    verdict_rule: Optional[VerdictRule] = None

    # Raw extraction prompt template (for Phase 2)
    extraction_prompt_template: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionSchema':
        """Create from dictionary."""
        # Reconstruct nested dataclasses
        extraction_fields = [ExtractionField(**f) for f in data.get('extraction_fields', [])]
        aggregations = [AggregationRule(**a) for a in data.get('aggregations', [])]
        computations = [ComputationStep(**c) for c in data.get('computations', [])]
        lookups = [LookupTable(**l) for l in data.get('lookups', [])]
        verdict_rule = VerdictRule(**data['verdict_rule']) if data.get('verdict_rule') else None

        return cls(
            task_name=data.get('task_name', ''),
            extraction_fields=extraction_fields,
            review_metadata=data.get('review_metadata', ["stars", "date", "useful"]),
            filter_keywords=data.get('filter_keywords', []),
            aggregations=aggregations,
            computations=computations,
            lookups=lookups,
            output_fields=data.get('output_fields', []),
            verdict_rule=verdict_rule,
            extraction_prompt_template=data.get('extraction_prompt_template', ''),
        )


# Prompt for Phase 1 LLM call
PHASE1_PROMPT = '''You are a formula parser. Analyze the following task specification and extract a structured execution schema.

## TASK SPECIFICATION

{task_prompt}

## YOUR JOB

Extract the following components:

### 1. EXTRACTION_FIELDS
What signals need to be extracted FROM EACH REVIEW? For each field:
- name: lowercase_snake_case identifier
- field_type: "enum", "boolean", "number", or "string"
- values: for enum types, list all valid values
- description: what this field captures

### 2. FILTER_KEYWORDS
What keywords indicate a review is relevant? (e.g., for allergy tasks: "allergy", "allergic", "peanut", "nut", etc.)

### 3. AGGREGATIONS
How to count/aggregate extracted values? Each rule:
- name: the aggregate variable name (e.g., "n_mild")
- formula: pseudo-code like "COUNT(field='value' AND other_field='value')"
- description: what this counts

### 4. COMPUTATIONS
Derived value formulas IN DEPENDENCY ORDER (earlier steps first). Each:
- name: variable name
- formula: Python expression using previous variables
- depends_on: list of variables this depends on
- description: what this computes

### 5. LOOKUPS
Static lookup tables (like cuisine modifiers). Each:
- name: variable name
- mapping: dict of key->value
- default: default value if no match
- source_field: what field to look up (e.g., "categories")

### 6. OUTPUT_FIELDS
List of final output field names in order.

### 7. VERDICT_RULE
How to determine final verdict:
- source_field: which field determines verdict
- thresholds: list of objects with "max" or "min" and "verdict" keys

### 8. EXTRACTION_PROMPT_TEMPLATE
Write a prompt template for extracting the fields from a single review.
Use REVIEW_TEXT, REVIEW_STARS, REVIEW_DATE as placeholders (will be substituted).
The prompt should ask the LLM to output JSON with the extraction fields.

## OUTPUT FORMAT

Output valid JSON matching this structure:
```json
{{
  "task_name": "...",
  "extraction_fields": [...],
  "filter_keywords": [...],
  "aggregations": [...],
  "computations": [...],
  "lookups": [...],
  "output_fields": [...],
  "verdict_rule": {{...}},
  "extraction_prompt_template": "..."
}}
```

Be precise. Use exact variable names from the formula.'''


class FormulaParser:
    """
    Phase 1: Parse task formula into ExecutionSchema.

    Usage:
        parser = FormulaParser()
        schema = await parser.parse(task_prompt)
        print(schema.to_json())
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    async def parse(self, task_prompt: str, task_name: str = "unknown") -> ExecutionSchema:
        """
        Parse a task formula into ExecutionSchema.

        Args:
            task_prompt: The formula specification (e.g., TASK_G1_PROMPT)
            task_name: Name identifier for the task

        Returns:
            ExecutionSchema ready for Phase 2
        """
        if self.verbose:
            print(f"[Phase 1] Parsing formula for task: {task_name}")

        # Build prompt
        prompt = PHASE1_PROMPT.format(task_prompt=task_prompt)

        # Call LLM
        response = await call_llm_async(prompt, role="planner")

        if self.verbose:
            print(f"[Phase 1] Got response ({len(response)} chars)")

        # Parse JSON from response
        schema_dict = self._extract_json(response)

        if not schema_dict:
            raise ValueError("Failed to parse schema from LLM response")

        # Ensure task_name is set
        schema_dict['task_name'] = task_name

        # Build ExecutionSchema
        schema = ExecutionSchema.from_dict(schema_dict)

        if self.verbose:
            print(f"[Phase 1] Extracted schema:")
            print(f"  - {len(schema.extraction_fields)} extraction fields")
            print(f"  - {len(schema.filter_keywords)} filter keywords")
            print(f"  - {len(schema.aggregations)} aggregation rules")
            print(f"  - {len(schema.computations)} computation steps")
            print(f"  - {len(schema.lookups)} lookup tables")
            print(f"  - {len(schema.output_fields)} output fields")

        return schema

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        # Find first { and last }
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

        return None


# Convenience function for testing
async def parse_formula(task_prompt: str, task_name: str = "unknown") -> ExecutionSchema:
    """Parse a formula into ExecutionSchema."""
    parser = FormulaParser()
    return await parser.parse(task_prompt, task_name)


if __name__ == "__main__":
    import asyncio

    # Test with G1a formula
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from g1_allergy import TASK_G1_PROMPT

    async def test():
        schema = await parse_formula(TASK_G1_PROMPT, "G1a")
        print("\n" + "="*60)
        print("EXECUTION SCHEMA")
        print("="*60)
        print(schema.to_json())

    asyncio.run(test())
