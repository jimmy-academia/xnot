"""
General ANoT: Task-agnostic Adaptive Network of Thought

A two-phase framework for executing task formulas on structured data:

- Phase 1: Compile (formula -> Formula Seed)
  Takes a task specification in natural language and produces an executable
  Formula Seed specification. The LLM discovers what signals to extract,
  how to aggregate them, and what formulas to compute.

- Phase 2: Execute (seed + data -> results)
  Interprets the Formula Seed against actual data (e.g., restaurant reviews).
  The interpreter has no task-specific logic - it executes what the seed specifies.

Usage:
    from explore.general_anot.phase1 import generate_formula_seed
    from explore.general_anot.phase2 import FormulaSeedInterpreter

    # Phase 1: Generate seed from task formula
    seed = await generate_formula_seed(task_prompt, "task_name")

    # Phase 2: Execute on data
    interpreter = FormulaSeedInterpreter(seed)
    result = await interpreter.execute(reviews, restaurant_context)

Evaluation:
    from explore.general_anot.eval import main as run_eval
    asyncio.run(run_eval())

Key Performance (G1a-v2, 100 restaurants):
- Ordinal AUPRC: 0.95 avg
- Primitive Accuracy: 0.87 avg
- Adjusted AUPRC: 0.82 avg (range: 0.74-0.87)
- Verdict Accuracy: 98%

Comparison vs Baselines:
- General ANoT: 0.82 avg (stable)
- Direct LLM:   0.53 avg (variable)
- Chain of Thought: 0.49 avg (highly variable)
"""

from .phase1 import generate_formula_seed
from .phase2 import FormulaSeedInterpreter, Extraction, ExecutionContext
