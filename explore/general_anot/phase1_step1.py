"""
Phase 1, Step 1.1: Extract Conditions

Parses a task formula to identify:
1. Per-review extraction fields (what signals to extract from each review)
2. Filter conditions (keywords/patterns to find relevant reviews)
3. Aggregation conditions (when to count/sum something)

Input: Task formula prompt (human-readable specification)
Output: ExtractionConditions dataclass
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm_async


@dataclass
class ExtractionField:
    """A signal to extract from each review."""
    name: str                           # e.g., "incident_severity"
    field_type: str                     # "enum", "boolean", "number", "string"
    values: List[str] = field(default_factory=list)  # For enum: valid values
    description: str = ""               # What this field captures
    extraction_hint: str = ""           # How to identify this in text


@dataclass
class FilterCondition:
    """Condition for filtering relevant reviews."""
    keywords: List[str] = field(default_factory=list)      # Keywords to match
    patterns: List[str] = field(default_factory=list)      # Regex patterns
    description: str = ""               # What makes a review relevant


@dataclass
class AggregationCondition:
    """Condition for counting/aggregating extractions."""
    name: str                           # e.g., "n_mild"
    agg_type: str                       # "count", "sum", "max", "min"
    conditions: Dict[str, Any] = field(default_factory=dict)  # e.g., {"incident_severity": "mild", "account_type": "firsthand"}
    description: str = ""


@dataclass
class ExtractionConditions:
    """
    Output of Step 1.1: All conditions needed for extraction.
    """
    # Per-review fields to extract
    extraction_fields: List[ExtractionField] = field(default_factory=list)

    # How to filter relevant reviews
    filter_condition: Optional[FilterCondition] = None

    # How to aggregate extracted values
    aggregation_conditions: List[AggregationCondition] = field(default_factory=list)

    # Metadata fields from reviews needed for computation
    required_metadata: List[str] = field(default_factory=lambda: ["stars", "date", "useful"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConditions':
        extraction_fields = [ExtractionField(**f) for f in data.get('extraction_fields', [])]
        filter_condition = FilterCondition(**data['filter_condition']) if data.get('filter_condition') else None
        aggregation_conditions = [AggregationCondition(**a) for a in data.get('aggregation_conditions', [])]

        return cls(
            extraction_fields=extraction_fields,
            filter_condition=filter_condition,
            aggregation_conditions=aggregation_conditions,
            required_metadata=data.get('required_metadata', ["stars", "date", "useful"]),
        )


STEP1_PROMPT = '''You are analyzing a task formula to extract structured conditions.

## TASK FORMULA

{task_prompt}

## YOUR JOB

Extract THREE things from this formula:

### 1. EXTRACTION_FIELDS
What signals must be extracted FROM EACH REVIEW? For each field:
- name: lowercase_snake_case (e.g., "incident_severity")
- field_type: "enum", "boolean", "number", or "string"
- values: for enum, list ALL valid values including "none" if applicable
- description: what this captures
- extraction_hint: specific phrases/patterns that indicate each value

Example:
```json
{{
  "name": "incident_severity",
  "field_type": "enum",
  "values": ["none", "mild", "moderate", "severe"],
  "description": "Severity of allergic reaction",
  "extraction_hint": "mild=stomach upset; moderate=hives,swelling; severe=anaphylaxis,EpiPen,ER"
}}
```

### 2. FILTER_CONDITION
How to identify RELEVANT reviews (not all reviews matter). Include:
- keywords: list of words that indicate relevance (case-insensitive)
- patterns: regex patterns if needed (optional)
- description: what makes a review relevant

### 3. AGGREGATION_CONDITIONS
How extracted values get COUNTED or SUMMED. For each aggregate:
- name: the variable name (e.g., "n_mild")
- agg_type: "count", "sum", "max", or "min"
- conditions: dict of field=value conditions that must ALL be true
- description: what this counts/sums

Example:
```json
{{
  "name": "n_mild",
  "agg_type": "count",
  "conditions": {{"incident_severity": "mild", "account_type": "firsthand"}},
  "description": "Count of firsthand mild incidents"
}}
```

### 4. REQUIRED_METADATA
What review metadata fields are needed for computation (e.g., stars, date, useful votes).

## OUTPUT FORMAT

Output valid JSON:
```json
{{
  "extraction_fields": [...],
  "filter_condition": {{
    "keywords": [...],
    "patterns": [...],
    "description": "..."
  }},
  "aggregation_conditions": [...],
  "required_metadata": ["stars", "date", "useful"]
}}
```

Be precise and complete. Include ALL fields and conditions from the formula.'''


class Step1ExtractConditions:
    """
    Step 1.1: Extract conditions from task formula.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    async def run(self, task_prompt: str) -> ExtractionConditions:
        """
        Extract conditions from task formula.

        Args:
            task_prompt: The formula specification

        Returns:
            ExtractionConditions with all extracted info
        """
        if self.verbose:
            print("[Step 1.1] Extracting conditions from formula...")

        # Build prompt
        prompt = STEP1_PROMPT.format(task_prompt=task_prompt)

        # Call LLM
        response = await call_llm_async(prompt, role="planner")

        if self.verbose:
            print(f"[Step 1.1] Got response ({len(response)} chars)")

        # Parse JSON
        data = self._extract_json(response)

        if not data:
            raise ValueError("Failed to parse conditions from LLM response")

        # Build result
        result = ExtractionConditions.from_dict(data)

        if self.verbose:
            print(f"[Step 1.1] Extracted:")
            print(f"  - {len(result.extraction_fields)} extraction fields")
            if result.filter_condition:
                print(f"  - {len(result.filter_condition.keywords)} filter keywords")
            print(f"  - {len(result.aggregation_conditions)} aggregation conditions")
            print(f"  - {len(result.required_metadata)} required metadata fields")

        return result

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Try JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

        return None


async def extract_conditions(task_prompt: str, verbose: bool = True) -> ExtractionConditions:
    """Convenience function to run Step 1.1."""
    step = Step1ExtractConditions(verbose=verbose)
    return await step.run(task_prompt)


# Test
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from g1_allergy import TASK_G1_PROMPT
    from utils.llm import configure

    async def test():
        configure(temperature=0.0)

        print("="*70)
        print("STEP 1.1: EXTRACT CONDITIONS")
        print("="*70)

        conditions = await extract_conditions(TASK_G1_PROMPT)

        # Save output
        output_dir = Path(__file__).parent.parent / "results" / "phase1_steps"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "step1_1_conditions.json"
        with open(output_file, 'w') as f:
            f.write(conditions.to_json())

        print(f"\nSaved to: {output_file}")

        # Pretty print
        print("\n" + "="*70)
        print("EXTRACTION FIELDS:")
        print("="*70)
        for ef in conditions.extraction_fields:
            print(f"\n  {ef.name} ({ef.field_type})")
            if ef.values:
                print(f"    values: {ef.values}")
            print(f"    desc: {ef.description}")
            if ef.extraction_hint:
                print(f"    hint: {ef.extraction_hint}")

        print("\n" + "="*70)
        print("FILTER CONDITION:")
        print("="*70)
        if conditions.filter_condition:
            print(f"  keywords: {conditions.filter_condition.keywords}")
            if conditions.filter_condition.patterns:
                print(f"  patterns: {conditions.filter_condition.patterns}")
            print(f"  desc: {conditions.filter_condition.description}")

        print("\n" + "="*70)
        print("AGGREGATION CONDITIONS:")
        print("="*70)
        for ac in conditions.aggregation_conditions:
            print(f"\n  {ac.name} = {ac.agg_type}()")
            print(f"    conditions: {ac.conditions}")
            print(f"    desc: {ac.description}")

        print("\n" + "="*70)
        print("REQUIRED METADATA:")
        print("="*70)
        print(f"  {conditions.required_metadata}")

        return conditions

    asyncio.run(test())
