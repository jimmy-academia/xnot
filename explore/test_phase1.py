#!/usr/bin/env python3
"""
Test Phase 1: Formula Parser

Run this to see what schema Phase 1 produces from G1a formula.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import configure

# Import test data
from g1_allergy import TASK_G1_PROMPT, TASK_G1_PROMPT_V2

# Import Phase 1
from general_anot.phase1 import FormulaParser, ExecutionSchema


async def test_phase1(task_prompt: str, task_name: str):
    """Test Phase 1 on a task prompt."""
    print(f"\n{'='*70}")
    print(f"TESTING PHASE 1: {task_name}")
    print(f"{'='*70}")

    parser = FormulaParser(verbose=True)
    schema = await parser.parse(task_prompt, task_name)

    # Save schema to file for inspection
    output_dir = Path(__file__).parent / "results" / "phase1_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{task_name}_schema.json"
    with open(output_file, 'w') as f:
        f.write(schema.to_json())

    print(f"\nSchema saved to: {output_file}")

    # Print key parts for review
    print(f"\n{'='*70}")
    print("EXTRACTION FIELDS:")
    print(f"{'='*70}")
    for field in schema.extraction_fields:
        print(f"  {field.name}: {field.field_type}")
        if field.values:
            print(f"    values: {field.values}")
        if field.description:
            print(f"    desc: {field.description}")

    print(f"\n{'='*70}")
    print("FILTER KEYWORDS:")
    print(f"{'='*70}")
    print(f"  {schema.filter_keywords}")

    print(f"\n{'='*70}")
    print("AGGREGATIONS:")
    print(f"{'='*70}")
    for agg in schema.aggregations:
        print(f"  {agg.name} = {agg.formula}")

    print(f"\n{'='*70}")
    print("COMPUTATIONS (in order):")
    print(f"{'='*70}")
    for comp in schema.computations:
        deps = f" [depends: {comp.depends_on}]" if comp.depends_on else ""
        print(f"  {comp.name} = {comp.formula}{deps}")

    print(f"\n{'='*70}")
    print("LOOKUPS:")
    print(f"{'='*70}")
    for lookup in schema.lookups:
        print(f"  {lookup.name} from {lookup.source_field}")
        print(f"    mapping: {lookup.mapping}")
        print(f"    default: {lookup.default}")

    print(f"\n{'='*70}")
    print("OUTPUT FIELDS:")
    print(f"{'='*70}")
    print(f"  {schema.output_fields}")

    print(f"\n{'='*70}")
    print("VERDICT RULE:")
    print(f"{'='*70}")
    if schema.verdict_rule:
        print(f"  source: {schema.verdict_rule.source_field}")
        for t in schema.verdict_rule.thresholds:
            print(f"    {t}")

    print(f"\n{'='*70}")
    print("EXTRACTION PROMPT TEMPLATE:")
    print(f"{'='*70}")
    print(schema.extraction_prompt_template[:500] + "..." if len(schema.extraction_prompt_template) > 500 else schema.extraction_prompt_template)

    return schema


async def main():
    configure(temperature=0.0)

    # Test on G1a (V1 formula)
    schema_v1 = await test_phase1(TASK_G1_PROMPT, "G1a")

    print("\n\n")

    # Optionally test on V2 as well
    # schema_v2 = await test_phase1(TASK_G1_PROMPT_V2, "G1a-v2")


if __name__ == "__main__":
    asyncio.run(main())
