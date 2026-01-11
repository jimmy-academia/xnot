"""
General ANoT: Task-agnostic Adaptive Network of Thought

Architecture:
- Phase 1: Compile (formula → Formula Seed)
  - Step 1.1: Extract conditions (fields, keywords, aggregations)
  - Step 1.2: Build computation graph (formulas, dependencies)
  - Step 1.3: Generate Formula Seed

- Phase 2: Execute (seed + restaurant → results)
  - Filter relevant reviews
  - Extract signals (parallel LLM)
  - Assemble arrays
  - Execute Compute DAG
  - Return results

Usage:
    from general_anot.phase1_step3 import FormulaSeed, generate_formula_seed
    from general_anot.phase2 import Phase2Executor

    # Phase 1: Generate seed from formula
    seed = await generate_formula_seed("G1a", extraction_conditions, computation_graph)

    # Phase 2: Execute on restaurant
    executor = Phase2Executor(seed)
    result = await executor.execute(restaurant)
"""

# Phase 1 components
from .phase1_step1 import ExtractionConditions, ExtractionField, extract_conditions
from .phase1_step2 import ComputationGraph, ComputationStep, build_computation_graph
from .phase1_step3 import FormulaSeed, generate_formula_seed

# Phase 2 components
from .phase2 import Phase2Executor, execute_seed

# Legacy unified Phase 1 (single LLM call)
from .phase1 import FormulaParser, ExecutionSchema
