"""
Phase 1 v2: Generate Formula Seed from Query

Single LLM call that reasons about the query and produces an executable specification.
The LLM discovers the structure needed - we don't prescribe a rigid schema.

Key principle: The LLM reads the query and figures out:
1. What signals need to be extracted from each review
2. How those signals are aggregated/combined
3. What external data (restaurant info) is used
4. What the computation flow is

The output structure emerges from understanding the query, not from a template.
"""

import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm_async


PHASE1_PROMPT = '''You are translating a task formula into an executable specification.

## TASK FORMULA

{task_prompt}

## YOUR JOB

Read the formula carefully. Your goal is to produce a JSON specification that captures EVERYTHING needed to execute this task on any restaurant's review data.

Think through these questions:

1. **FILTERING**: How do we find relevant reviews? What keywords or patterns indicate a review is relevant to this task?

2. **EXTRACTION**: What information must be extracted from each relevant review?
   - What are the semantic signals/fields?
   - For each field, what are the possible values and what does each value MEAN?
   - Include enough description that someone could correctly classify a review.

3. **AGGREGATION**: How are the extracted values combined?
   - What counts, sums, max/min operations are needed?
   - What conditions define which extractions to include?
   - Are there derived concepts (like "incident" = firsthand account with severity)?

4. **EXTERNAL DATA**: What restaurant-level information is used?
   - Any lookups based on restaurant attributes (categories, location, etc.)?
   - How should matching work (exact match, substring, etc.)?

5. **COMPUTATION**: What formulas derive the final results?
   - What is the dependency order?
   - What are the intermediate values?
   - What is the final output?

## OUTPUT

Produce a JSON specification that an interpreter could execute. The structure should:
- Capture the FULL semantics of the task (not just variable names, but what they mean)
- Be unambiguous about what comes from where (review text, review metadata, restaurant info)
- Preserve the computation order and dependencies
- Include default values where the formula specifies them

Think step by step, then output the JSON specification.

```json
'''


async def generate_formula_seed(task_prompt: str, task_name: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Generate Formula Seed from task prompt.

    The LLM reasons about the query and produces a structured specification.
    We don't prescribe the exact schema - it emerges from understanding the task.

    Args:
        task_prompt: The task formula (natural language specification)
        task_name: Name identifier for the task
        verbose: Print progress

    Returns:
        Formula Seed as a dictionary
    """
    if verbose:
        print(f"\n[Phase 1] Generating Formula Seed for: {task_name}")
        print(f"  Query length: {len(task_prompt)} chars")

    # Build prompt
    prompt = PHASE1_PROMPT.format(task_prompt=task_prompt)

    # Call LLM
    if verbose:
        print(f"  Calling LLM...")

    response = await call_llm_async(prompt, role="planner")

    if verbose:
        print(f"  Response length: {len(response)} chars")

    # Save raw response for debugging
    output_dir = Path(__file__).parent.parent / "results" / "phase1_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_file = output_dir / "raw_response.txt"
    with open(raw_file, 'w') as f:
        f.write(response)
    if verbose:
        print(f"  Raw response saved to: {raw_file}")

    # Extract JSON from response
    seed = _extract_json(response)

    if seed is None:
        raise ValueError(f"Failed to extract JSON from LLM response. See {raw_file}")

    # Add task name
    seed["task_name"] = task_name

    if verbose:
        print(f"  Generated seed with keys: {list(seed.keys())}")

    return seed


def _strip_json_comments(json_str: str) -> str:
    """Strip JavaScript-style comments from JSON string."""
    # Remove single-line comments (// ...)
    json_str = re.sub(r'//[^\n]*', '', json_str)
    # Remove multi-line comments (/* ... */)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str


def _extract_json(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = _strip_json_comments(json_match.group(1))
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON decode error in code block: {e}")

    # Try to find raw JSON (find matching braces)
    start = response.find('{')
    if start >= 0:
        depth = 0
        end = start
        for i, c in enumerate(response[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = _strip_json_comments(response[start:end])
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON decode error in raw JSON: {e}")

    return None


async def test_phase1():
    """Test Phase 1 with G1a-v2 query."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from g1_allergy import TASK_G1_PROMPT_V2
    from utils.llm import configure

    configure(temperature=0.0)

    print("=" * 70)
    print("PHASE 1 V2: Generate Formula Seed")
    print("=" * 70)

    seed = await generate_formula_seed(TASK_G1_PROMPT_V2, "G1a-v2", verbose=True)

    # Save output
    output_dir = Path(__file__).parent.parent / "results" / "phase1_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "formula_seed.json"
    with open(output_file, 'w') as f:
        json.dump(seed, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Pretty print structure
    print("\n" + "=" * 70)
    print("FORMULA SEED STRUCTURE:")
    print("=" * 70)

    def print_structure(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and len(str(v)) > 80:
                    print(f"{prefix}{k}:")
                    print_structure(v, indent + 1)
                else:
                    print(f"{prefix}{k}: {type(v).__name__} = {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}")
        elif isinstance(obj, list):
            print(f"{prefix}[{len(obj)} items]")
            if obj and isinstance(obj[0], dict):
                # Show first item structure
                print(f"{prefix}  [0]:")
                print_structure(obj[0], indent + 2)
                if len(obj) > 1:
                    print(f"{prefix}  ... and {len(obj)-1} more")

    print_structure(seed)

    return seed


if __name__ == "__main__":
    asyncio.run(test_phase1())
