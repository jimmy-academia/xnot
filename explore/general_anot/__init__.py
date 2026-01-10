"""
General ANoT: Task-agnostic Adaptive Network of Thought

3-Phase Architecture:
- Phase 1: Parse formula → generate execution schema (cached per task)
- Phase 2: Extract signals from reviews (LLM per review)
- Phase 3: Compute aggregates and derived values (Python only)
"""

from .phase1 import FormulaParser, ExecutionSchema
