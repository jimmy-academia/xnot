#!/usr/bin/env python3
"""
Test Step 1.3 using saved outputs from Step 1.1 and 1.2.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from general_anot.phase1_step1 import ExtractionConditions
from general_anot.phase1_step2 import ComputationGraph
from general_anot.phase1_step3 import generate_formula_seed

# Load saved outputs
output_dir = Path(__file__).parent / "results" / "phase1_steps"

# Load Step 1.1 output
with open(output_dir / "step1_1_conditions.json") as f:
    conditions_data = json.load(f)
conditions = ExtractionConditions.from_dict(conditions_data)

# Load Step 1.2 output
with open(output_dir / "step1_2_computation_graph.json") as f:
    graph_data = json.load(f)
graph = ComputationGraph.from_dict(graph_data)

print("="*70)
print("Loaded Step 1.1 and 1.2 outputs")
print("="*70)
print(f"Extraction fields: {len(conditions.extraction_fields)}")
print(f"Aggregation conditions: {len(conditions.aggregation_conditions)}")
print(f"Computation steps: {len(graph.computations)}")

print("\n" + "="*70)
print("STEP 1.3: GENERATE FORMULA SEED")
print("="*70)

seed = generate_formula_seed("G1a", conditions, graph, verbose=True)

# Save outputs
with open(output_dir / "step1_3_formula_seed_v2.json", 'w') as f:
    f.write(seed.to_json())

with open(output_dir / "step1_3_formula_script_v2.txt", 'w') as f:
    f.write(seed.to_script())

print(f"\nSaved to: {output_dir}")

# Print key outputs
print("\n" + "="*70)
print("EXTRACTION FIELDS (filtered):")
print("="*70)
for ef in seed.extraction_fields:
    print(f"  {ef.name}: {ef.field_type} = {ef.values}")

print("\n" + "="*70)
print("ARRAY ASSEMBLIES:")
print("="*70)
for aa in seed.array_assemblies:
    filter_str = f" WHERE {aa.filter_condition}" if aa.filter_condition else ""
    print(f"  {aa.array_name} <- extraction['{aa.source_field}']{filter_str}")

print("\n" + "="*70)
print("AGGREGATION COMPUTATIONS (from Step 1.1):")
print("="*70)
for comp in seed.computations:
    if comp.is_aggregate:
        print(f"  {comp.name} = {comp.formula}")

print("\n" + "="*70)
print("DERIVED COMPUTATIONS:")
print("="*70)
for comp in seed.computations:
    if not comp.is_aggregate:
        print(f"  {comp.name} = {comp.formula[:80]}{'...' if len(comp.formula) > 80 else ''}")

print("\n" + "="*70)
print("FORMULA SCRIPT:")
print("="*70)
print(seed.to_script())
