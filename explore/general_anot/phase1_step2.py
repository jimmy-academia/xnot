"""
Phase 1, Step 1.2: Build Computation Graph

Parses a task formula to build the computation graph:
1. Lookup tables (static mappings)
2. Derived computations (formulas in dependency order)
3. Output fields and verdict rules

Input: Task formula prompt + ExtractionConditions from Step 1.1
Output: ComputationGraph dataclass

The computation graph contains EXECUTABLE Python formulas.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm_async

try:
    from .phase1_step1 import ExtractionConditions
except ImportError:
    from general_anot.phase1_step1 import ExtractionConditions


@dataclass
class LookupTable:
    """Static lookup mapping."""
    name: str                           # Variable name (e.g., "cuisine_modifier")
    source_field: str                   # What to look up (e.g., "categories")
    mapping: Dict[str, float] = field(default_factory=dict)
    default: float = 1.0
    description: str = ""


@dataclass
class ComputationStep:
    """A single computation in the graph."""
    name: str                           # Variable name (e.g., "incident_score")
    formula: str                        # Executable Python expression
    depends_on: List[str] = field(default_factory=list)  # Variable dependencies
    description: str = ""
    is_aggregate: bool = False          # True if this aggregates extractions
    is_output: bool = False             # True if this is a final output


@dataclass
class VerdictRule:
    """Rule for determining final verdict."""
    source_field: str                   # e.g., "final_risk_score"
    rules: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"condition": "< 4.0", "verdict": "Low Risk"}, ...]


@dataclass
class ComputationGraph:
    """
    Output of Step 1.2: Complete computation graph.
    """
    # Lookup tables (evaluated first)
    lookups: List[LookupTable] = field(default_factory=list)

    # Computation steps in dependency order
    computations: List[ComputationStep] = field(default_factory=list)

    # Final output field names
    output_fields: List[str] = field(default_factory=list)

    # Verdict determination
    verdict_rule: Optional[VerdictRule] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationGraph':
        lookups = [LookupTable(**l) for l in data.get('lookups', [])]
        computations = [ComputationStep(**c) for c in data.get('computations', [])]
        verdict_rule = VerdictRule(**data['verdict_rule']) if data.get('verdict_rule') else None

        return cls(
            lookups=lookups,
            computations=computations,
            output_fields=data.get('output_fields', []),
            verdict_rule=verdict_rule,
        )

    def get_execution_order(self) -> List[str]:
        """Return variable names in execution order."""
        order = []
        for lookup in self.lookups:
            order.append(lookup.name)
        for comp in self.computations:
            order.append(comp.name)
        return order


STEP2_PROMPT = '''You are building a computation graph from a task formula.

## TASK FORMULA

{task_prompt}

## EXTRACTION CONDITIONS (from Step 1.1)

The following will be extracted from each review:
{extraction_summary}

The following aggregations will be computed:
{aggregation_summary}

## YOUR JOB

Build the COMPUTATION GRAPH with executable Python formulas. Include:

### 1. LOOKUPS
Static lookup tables. Each lookup:
- name: variable name
- source_field: what field to look up (e.g., "categories" from restaurant metadata)
- mapping: dict of string -> float
- default: default value if no match
- description: what this represents

### 2. COMPUTATIONS
ALL derived formulas in DEPENDENCY ORDER (earlier dependencies first).

IMPORTANT: Write EXECUTABLE Python expressions. Use:
- Variable references: just use the variable name (e.g., `n_mild`, `incident_score`)
- Math: standard Python (`+`, `-`, `*`, `/`, `**`)
- Functions: `max()`, `min()`, `abs()`, `log()` (math.log), `sum()`, `len()`
- Conditionals: ternary `x if condition else y`

For each computation:
- name: variable name (lowercase_snake_case)
- formula: executable Python expression
- depends_on: list of variables this formula uses
- description: what this computes
- is_aggregate: true if this aggregates from extractions (like n_mild, n_total_incidents)
- is_output: true if this should be in final output

### 3. OUTPUT_FIELDS
List of variable names that should be reported (in order).

### 4. VERDICT_RULE
How to determine the final verdict:
- source_field: which variable to check
- rules: list of conditions and verdicts, checked in order
  Each rule: {{"condition": "< 4.0", "verdict": "Low Risk"}}
  Conditions: "< X", "<= X", "> X", ">= X", "== X"

## OUTPUT FORMAT

Output valid JSON:
```json
{{
  "lookups": [
    {{
      "name": "cuisine_modifier",
      "source_field": "categories",
      "mapping": {{"Thai": 2.0, "Chinese": 1.5, ...}},
      "default": 1.0,
      "description": "Risk modifier based on cuisine type"
    }}
  ],
  "computations": [
    {{
      "name": "n_total_incidents",
      "formula": "n_mild + n_moderate + n_severe",
      "depends_on": ["n_mild", "n_moderate", "n_severe"],
      "description": "Total firsthand incidents",
      "is_aggregate": true,
      "is_output": true
    }},
    {{
      "name": "incident_score",
      "formula": "n_mild * 2 + n_moderate * 5 + n_severe * 15",
      "depends_on": ["n_mild", "n_moderate", "n_severe"],
      "description": "Weighted incident score",
      "is_aggregate": false,
      "is_output": true
    }},
    ...
  ],
  "output_fields": ["n_total_incidents", "incident_score", "recency_decay", "credibility_factor", "final_risk_score", "verdict"],
  "verdict_rule": {{
    "source_field": "final_risk_score",
    "rules": [
      {{"condition": "< 4.0", "verdict": "Low Risk"}},
      {{"condition": "< 8.0", "verdict": "High Risk"}},
      {{"condition": ">= 8.0", "verdict": "Critical Risk"}}
    ]
  }}
}}
```

IMPORTANT:
- Include ALL computations from the formula
- Formulas must be EXECUTABLE Python (not pseudo-code)
- Order computations so dependencies come first
- Mark aggregates and outputs correctly'''


class Step2ComputationGraph:
    """
    Step 1.2: Build computation graph from task formula.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    async def run(
        self,
        task_prompt: str,
        extraction_conditions: ExtractionConditions
    ) -> ComputationGraph:
        """
        Build computation graph.

        Args:
            task_prompt: The formula specification
            extraction_conditions: Output from Step 1.1

        Returns:
            ComputationGraph with all computations
        """
        if self.verbose:
            print("[Step 1.2] Building computation graph...")

        # Build extraction summary for prompt
        extraction_summary = self._summarize_extractions(extraction_conditions)
        aggregation_summary = self._summarize_aggregations(extraction_conditions)

        # Build prompt
        prompt = STEP2_PROMPT.format(
            task_prompt=task_prompt,
            extraction_summary=extraction_summary,
            aggregation_summary=aggregation_summary,
        )

        # Call LLM
        response = await call_llm_async(prompt, role="planner")

        if self.verbose:
            print(f"[Step 1.2] Got response ({len(response)} chars)")

        # Parse JSON
        data = self._extract_json(response)

        if not data:
            raise ValueError("Failed to parse computation graph from LLM response")

        # Build result
        result = ComputationGraph.from_dict(data)

        if self.verbose:
            print(f"[Step 1.2] Built graph:")
            print(f"  - {len(result.lookups)} lookup tables")
            print(f"  - {len(result.computations)} computation steps")
            print(f"  - {len(result.output_fields)} output fields")

        return result

    def _summarize_extractions(self, conditions: ExtractionConditions) -> str:
        """Summarize extraction fields for prompt."""
        lines = []
        for ef in conditions.extraction_fields:
            if ef.field_type == "enum":
                lines.append(f"- {ef.name}: {ef.values}")
            else:
                lines.append(f"- {ef.name}: {ef.field_type}")
        return "\n".join(lines)

    def _summarize_aggregations(self, conditions: ExtractionConditions) -> str:
        """Summarize aggregation conditions for prompt."""
        lines = []
        for ac in conditions.aggregation_conditions:
            cond_str = ", ".join(f"{k}={v}" for k, v in ac.conditions.items())
            lines.append(f"- {ac.name} = {ac.agg_type}({cond_str})")
        return "\n".join(lines)

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


async def build_computation_graph(
    task_prompt: str,
    extraction_conditions: ExtractionConditions,
    verbose: bool = True
) -> ComputationGraph:
    """Convenience function to run Step 1.2."""
    step = Step2ComputationGraph(verbose=verbose)
    return await step.run(task_prompt, extraction_conditions)


# Test
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from g1_allergy import TASK_G1_PROMPT
    from utils.llm import configure
    from general_anot.phase1_step1 import extract_conditions

    async def test():
        configure(temperature=0.0)

        print("="*70)
        print("STEP 1.1: EXTRACT CONDITIONS")
        print("="*70)

        conditions = await extract_conditions(TASK_G1_PROMPT, verbose=True)

        print("\n" + "="*70)
        print("STEP 1.2: BUILD COMPUTATION GRAPH")
        print("="*70)

        graph = await build_computation_graph(TASK_G1_PROMPT, conditions, verbose=True)

        # Save output
        output_dir = Path(__file__).parent.parent / "results" / "phase1_steps"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "step1_2_computation_graph.json"
        with open(output_file, 'w') as f:
            f.write(graph.to_json())

        print(f"\nSaved to: {output_file}")

        # Pretty print
        print("\n" + "="*70)
        print("LOOKUPS:")
        print("="*70)
        for lookup in graph.lookups:
            print(f"\n  {lookup.name} (from {lookup.source_field})")
            print(f"    mapping: {lookup.mapping}")
            print(f"    default: {lookup.default}")

        print("\n" + "="*70)
        print("COMPUTATIONS (in order):")
        print("="*70)
        for comp in graph.computations:
            flags = []
            if comp.is_aggregate:
                flags.append("AGG")
            if comp.is_output:
                flags.append("OUT")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            print(f"\n  {comp.name}{flag_str}")
            print(f"    formula: {comp.formula}")
            print(f"    depends: {comp.depends_on}")
            print(f"    desc: {comp.description}")

        print("\n" + "="*70)
        print("OUTPUT FIELDS:")
        print("="*70)
        print(f"  {graph.output_fields}")

        print("\n" + "="*70)
        print("VERDICT RULE:")
        print("="*70)
        if graph.verdict_rule:
            print(f"  source: {graph.verdict_rule.source_field}")
            for rule in graph.verdict_rule.rules:
                print(f"    {rule}")

        return graph

    asyncio.run(test())
