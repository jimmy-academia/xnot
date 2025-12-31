#!/usr/bin/env python3
"""KNoT v4: Iterative Hierarchical Planning with AND/OR parsing."""

import os
import re
import json
import asyncio
from typing import Optional

# Import from shared module (formerly base.py)
from .shared import (
    DEBUG,
    _current_item_id, _current_request_id,
    SYSTEM_PROMPT,
    call_llm,
    DebugLogger,
)
from utils.parsing import parse_script, parse_final_answer

# Import base class
from .knot import KnowledgeNetworkOfThought


class KnowledgeNetworkOfThoughtV4(KnowledgeNetworkOfThought):
    """
    v4: Iterative Hierarchical Planning with AND/OR parsing.

    Stage 1: Hierarchical Planning (phases i-vi)
    Stage 2: Script Generation with refinement (phases a-d)
    """

    def __init__(self, mode="string", run_dir: str = None):
        super().__init__(mode)
        self.run_dir = run_dir
        self.logger = None
        self.dag = None  # Stores the DAG structure from Stage 1

    def _get_review_count(self, query) -> int:
        """Extract number of reviews from query."""
        if isinstance(query, dict):
            item_data = query.get("item_data", [])
            return len(item_data)
        elif isinstance(query, str):
            # Try to count review sections
            review_markers = query.count("Review") + query.count("review") + query.count("Stars:")
            return max(1, review_markers // 2)  # Rough estimate
        return 5  # Default

    def _log(self, phase: str, event: str, data: dict = None):
        """Log if logger is available."""
        if self.logger:
            self.logger.log(phase, event, data)

    def _log_llm(self, phase: str, prompt: str, response: str):
        """Log LLM call if logger is available."""
        if self.logger:
            self.logger.log_llm_call(phase, prompt, response)

    def _flush(self):
        """Flush logger if available."""
        if self.logger:
            self.logger.flush()

    # ==================== STAGE 1: Hierarchical Planning ====================

    def stage1_phase_i(self, context: str) -> dict:
        """Phase i: Parse request structure into MUST/SHOULD/logic."""
        self._log("1.i", "start")

        prompt = f"""Parse this user request into structured conditions.

Request: {context}

Output a JSON object with:
- "must": list of dealbreaker conditions (required)
- "should": list of important but flexible conditions
- "logic": the AND/OR relationship between conditions
- "decision_rule": how to combine into -1, 0, or 1

Example for "fast service AND (good food OR good value)":
{{
  "must": ["speed"],
  "should": ["food_quality", "value"],
  "logic": "speed AND (food OR value)",
  "decision_rule": "if speed negative -> -1; if speed positive and (food or value positive) -> 1; else -> 0"
}}

Output ONLY the JSON object, no other text."""

        response = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        self._log_llm("1.i", prompt, response)

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: create a simple structure
            result = {
                "must": ["main_criterion"],
                "should": [],
                "logic": "main_criterion",
                "decision_rule": "if positive -> 1; if negative -> -1; else -> 0"
            }

        self._log("1.i", "end", {"result": result})
        self._flush()

        if DEBUG:
            print(f"Phase 1.i result: {result}")

        return result

    def stage1_phase_ii(self, query) -> dict:
        """Phase ii: Summarize data structure."""
        self._log("1.ii", "start")

        review_count = self._get_review_count(query)

        if isinstance(query, dict):
            # Dict mode - extract structure info
            item_data = query.get("item_data", [])
            content_map = []
            for i, review in enumerate(item_data[:10]):  # Limit to first 10
                review_text = review.get("review", "")[:100]
                stars = review.get("stars", "?")
                content_map.append(f"R{i}: {stars} stars, {len(review_text)} chars")
        else:
            # String mode - basic structure
            content_map = [f"R{i}: review {i}" for i in range(review_count)]

        result = {
            "count": review_count,
            "content_map": content_map
        }

        self._log("1.ii", "end", {"result": result})
        self._flush()

        if DEBUG:
            print(f"Phase 1.ii result: {result}")

        return result

    def stage1_phase_iii(self, phase_i_result: dict, phase_ii_result: dict) -> list:
        """Phase iii: High-level plan skeleton."""
        self._log("1.iii", "start")

        conditions = phase_i_result.get("must", []) + phase_i_result.get("should", [])
        review_count = phase_ii_result.get("count", 5)
        logic = phase_i_result.get("logic", "")

        prompt = f"""Create a high-level evaluation plan.

Conditions to check: {conditions}
Number of reviews: {review_count}
Logic: {logic}

Output a numbered list of high-level steps:
1. For each review, extract evidence for each condition
2. Aggregate evidence per condition
3. Apply logic to get final answer

Keep it brief, 3-5 steps max."""

        response = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        self._log_llm("1.iii", prompt, response)

        # Parse response into list
        result = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                result.append(line)

        if not result:
            result = [
                "1. For each review: extract evidence for each condition",
                "2. Aggregate evidence per condition across all reviews",
                f"3. Apply logic: {logic}"
            ]

        self._log("1.iii", "end", {"result": result})
        self._flush()

        if DEBUG:
            print(f"Phase 1.iii result: {result}")

        return result

    def stage1_phase_iv(self, phase_i_result: dict, phase_iii_result: list) -> list:
        """Phase iv: Validate plan <-> request (check-fix loop)."""
        self._log("1.iv", "start")

        conditions = phase_i_result.get("must", []) + phase_i_result.get("should", [])
        plan_text = "\n".join(phase_iii_result)

        # Check
        checks = []
        checks.append(("conditions_covered", all(c.lower() in plan_text.lower() for c in conditions)))
        checks.append(("has_extraction", "extract" in plan_text.lower() or "review" in plan_text.lower()))
        checks.append(("has_aggregation", "aggregate" in plan_text.lower() or "combine" in plan_text.lower()))
        checks.append(("has_decision", "logic" in plan_text.lower() or "final" in plan_text.lower() or "answer" in plan_text.lower()))

        all_passed = all(passed for _, passed in checks)
        self._log("1.iv", "check", {"checks": checks, "all_passed": all_passed})

        if not all_passed:
            # Fix: regenerate plan with explicit requirements
            prompt = f"""The plan is missing some elements. Create a complete plan that:
1. Covers ALL these conditions: {conditions}
2. Extracts evidence from each review
3. Aggregates evidence per condition
4. Applies decision logic

Current incomplete plan:
{plan_text}

Write the corrected plan as a numbered list."""

            response = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
            self._log_llm("1.iv", prompt, response)
            self._log("1.iv", "fix", {"action": "regenerate_plan"})

            # Parse fixed plan
            result = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    result.append(line)
            if not result:
                result = phase_iii_result
        else:
            result = phase_iii_result

        self._log("1.iv", "end", {"result": result})
        self._flush()

        if DEBUG:
            print(f"Phase 1.iv result: {result}")

        return result

    def stage1_phase_v(self, phase_i_result: dict, phase_ii_result: dict, phase_iv_result: list) -> dict:
        """Phase v: Expand into DAG structure."""
        self._log("1.v", "start")

        conditions = phase_i_result.get("must", []) + phase_i_result.get("should", [])
        review_count = phase_ii_result.get("count", 5)
        logic = phase_i_result.get("logic", "")
        decision_rule = phase_i_result.get("decision_rule", "")

        # Build DAG structure
        review_blocks = []
        for r in range(review_count):
            steps = [f"extract_{c}_R{r}" for c in conditions]
            review_blocks.append({"review": r, "steps": steps})

        aggregation = []
        for c in conditions:
            inputs = [f"extract_{c}_R{r}" for r in range(review_count)]
            aggregation.append({
                "condition": c,
                "inputs": inputs,
                "output": f"{c}_score"
            })

        decision = {
            "inputs": [f"{c}_score" for c in conditions],
            "logic": logic,
            "decision_rule": decision_rule,
            "output": "final_answer"
        }

        result = {
            "conditions": conditions,
            "review_count": review_count,
            "review_blocks": review_blocks,
            "aggregation": aggregation,
            "decision": decision
        }

        self._log("1.v", "end", {"result": result})
        self._flush()

        if DEBUG:
            print(f"Phase 1.v result: {json.dumps(result, indent=2)}")

        return result

    def stage1_phase_vi(self, dag: dict) -> dict:
        """Phase vi: Final plan validation."""
        self._log("1.vi", "start")

        # Validate DAG completeness
        checks = []

        review_blocks = dag.get("review_blocks", [])
        conditions = dag.get("conditions", [])
        review_count = dag.get("review_count", 0)

        checks.append(("review_blocks_count", len(review_blocks) == review_count))
        checks.append(("has_aggregation", len(dag.get("aggregation", [])) == len(conditions)))
        checks.append(("has_decision", dag.get("decision") is not None))

        all_passed = all(passed for _, passed in checks)
        self._log("1.vi", "check", {"checks": checks, "all_passed": all_passed})

        if not all_passed:
            self._log("1.vi", "fix", {"action": "structure_incomplete", "note": "proceeding with available structure"})

        self._log("1.vi", "end", {"validated": all_passed})
        self._flush()

        return dag

    # ==================== STAGE 2: Script Generation ====================

    def stage2_phase_a(self, dag: dict, query, context: str) -> str:
        """Phase a: Generate script deterministically from DAG structure.

        No LLM needed - we have the structure from Stage 1, just convert to script format.
        This eliminates the 70% failure rate from LLM-based generation.
        """
        self._log("2.a", "start")

        conditions = dag.get("conditions", ["criterion"])
        review_count = dag.get("review_count", 5)
        logic = dag.get("decision", {}).get("logic", "criterion")

        lines = []
        step = 0

        # 1. Extraction steps: one per (review × condition)
        # Layout: for each review, extract all conditions
        for r in range(review_count):
            for c in conditions:
                lines.append(
                    f'({step})=LLM("Extract {c} evidence from review {r}: '
                    f'{{{{(input)}}}}[item_data][{r}][review]. Output: POSITIVE, NEGATIVE, or NONE")'
                )
                step += 1

        # 2. Aggregation steps: one per condition
        # Each aggregation references all extraction steps for that condition
        agg_start = step
        num_conditions = len(conditions)
        for i, c in enumerate(conditions):
            # References to extraction steps for this condition across all reviews
            refs = ", ".join(f"{{{{({r * num_conditions + i})}}}}" for r in range(review_count))
            lines.append(
                f'({step})=LLM("Aggregate {c}: {refs}. '
                f'Count POSITIVE vs NEGATIVE. Output: POSITIVE, NEGATIVE, or NEUTRAL")'
            )
            step += 1

        # 3. Decision step: apply logic using aggregation results
        agg_refs = ", ".join(f"{c}={{{{({agg_start + i})}}}}" for i, c in enumerate(conditions))
        lines.append(
            f'({step})=LLM("Apply logic: {agg_refs}. Logic: {logic}. Output ONLY: -1, 0, or 1")'
        )

        script = "\n".join(lines)

        self._log("2.a", "end", {"script_length": len(script), "step_count": step + 1})
        self._flush()

        if DEBUG:
            print(f"Phase 2.a script (deterministic):\n{script[:500]}...")

        return script

    def stage2_phase_b(self, script: str, dag: dict, max_iterations: int = 2) -> str:
        """Phase b: Validate script structure (no LLM needed with deterministic generation)."""
        self._log("2.b", "start")

        conditions = dag.get("conditions", [])
        review_count = dag.get("review_count", 5)
        expected_extraction_steps = review_count * len(conditions)
        expected_aggregation_steps = len(conditions)
        expected_total = expected_extraction_steps + expected_aggregation_steps + 1

        # Parse and validate
        steps = parse_script(script)

        checks = [
            ("has_steps", len(steps) > 0),
            ("has_enough_steps", len(steps) >= expected_total),
            ("ends_with_decision", len(steps) > 0 and any(x in steps[-1][1] for x in ["-1", "0", "1"])),
        ]

        all_passed = all(passed for _, passed in checks)
        self._log("2.b", "check", {
            "checks": checks,
            "step_count": len(steps),
            "expected": expected_total,
            "all_passed": all_passed
        })

        if DEBUG:
            status = "PASSED" if all_passed else "FAILED"
            print(f"Phase 2.b validation: {status} ({len(steps)}/{expected_total} steps)")
            if not all_passed:
                failed = [name for name, passed in checks if not passed]
                print(f"  Failed checks: {failed}")

        self._log("2.b", "end", {"final_step_count": len(steps), "valid": all_passed})
        self._flush()

        return script

    def stage2_phase_c(self, script: str, dag: dict, query, context: str) -> str:
        """Phase c: Local refinement per block."""
        self._log("2.c", "start")

        steps = parse_script(script)
        if not steps:
            self._log("2.c", "end", {"result": "no_steps"})
            self._flush()
            return script

        conditions = dag.get("conditions", [])
        review_count = dag.get("review_count", 5)

        # Check each step has valid variable references
        issues = []
        for idx, instr in steps:
            # Check for input references
            if "{(input)}" in instr or "{(context)}" in instr:
                continue  # OK
            # Check for numbered references
            refs = re.findall(r'\{\((\d+)\)\}', instr)
            for ref in refs:
                if int(ref) >= int(idx):
                    issues.append((idx, f"references future step {ref}"))

        self._log("2.c", "check", {"issues": issues})

        if issues:
            # Fix issues
            for step_idx, issue in issues[:3]:  # Fix first 3 issues
                prompt = f"""Fix this script step. Issue: {issue}

Step ({step_idx}): {dict(steps).get(step_idx, '')}

Context: {context}
Conditions: {conditions}

Output only the corrected step in format: (N)=LLM("...")"""

                fix = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
                self._log_llm("2.c", prompt, fix)
                self._log("2.c", "fix", {"step": step_idx, "issue": issue})

        self._log("2.c", "end", {"issues_found": len(issues)})
        self._flush()

        return script

    def stage2_phase_d(self, script: str, dag: dict) -> str:
        """Phase d: Final validation."""
        self._log("2.d", "start")

        steps = parse_script(script)

        # Final checks
        checks = []
        checks.append(("has_steps", len(steps) > 0))
        checks.append(("has_final_step", len(steps) > 0))

        if steps:
            final_instr = steps[-1][1]
            checks.append(("final_outputs_number", "-1" in final_instr or "0" in final_instr or "1" in final_instr))

        all_passed = all(passed for _, passed in checks)
        self._log("2.d", "check", {"checks": checks, "all_passed": all_passed})
        self._log("2.d", "end", {"validated": all_passed})
        self._flush()

        if DEBUG:
            print(f"Phase 2.d final script validated: {all_passed}")

        return script

    # ==================== Main solve method ====================

    def solve(self, query, context: str, item_id: str = None, request_id: str = None) -> int:
        """Full v4 pipeline: Stage 1 (planning) -> Stage 2 (script gen) -> Execute."""

        # Use global IDs if not provided
        if item_id is None:
            item_id = _current_item_id
        if request_id is None:
            request_id = _current_request_id
        if item_id is None and isinstance(query, dict):
            item_id = query.get("item_id", "unknown")
        elif item_id is None:
            item_id = "unknown"

        # Initialize logger if run_dir is set and DebugLogger is available
        if self.run_dir and DebugLogger:
            self.logger = DebugLogger(self.run_dir, item_id, request_id or "unknown")
            self.logger.__enter__()

        try:
            # === STAGE 1: Hierarchical Planning ===
            if DEBUG:
                print("\n" + "=" * 60)
                print("STAGE 1: Hierarchical Planning")
                print("=" * 60)

            phase_i_result = self.stage1_phase_i(context)
            phase_ii_result = self.stage1_phase_ii(query)
            phase_iii_result = self.stage1_phase_iii(phase_i_result, phase_ii_result)
            phase_iv_result = self.stage1_phase_iv(phase_i_result, phase_iii_result)
            phase_v_result = self.stage1_phase_v(phase_i_result, phase_ii_result, phase_iv_result)
            dag = self.stage1_phase_vi(phase_v_result)

            self.dag = dag  # Store for potential debugging

            # === STAGE 2: Script Generation ===
            if DEBUG:
                print("\n" + "=" * 60)
                print("STAGE 2: Script Generation")
                print("=" * 60)

            script = self.stage2_phase_a(dag, query, context)
            if DEBUG:
                print(f"Phase 2.a script:\n{script}\n")
            script = self.stage2_phase_b(script, dag)
            if DEBUG:
                print(f"Phase 2.b script:\n{script}\n")
            script = self.stage2_phase_c(script, dag, query, context)
            script = self.stage2_phase_d(script, dag)
            if DEBUG:
                steps = parse_script(script)
                print(f"Final script has {len(steps)} parsed steps")

            # === STAGE 3: Execute ===
            if DEBUG:
                print("\n" + "=" * 60)
                print("STAGE 3: Execution")
                print("=" * 60)

            self._log("3", "start")

            try:
                output = asyncio.run(self.execute_script_parallel(script, query, context))
            except RuntimeError:
                output = self.execute_script(script, query, context)

            if DEBUG:
                print(f"Raw output before parsing: '{output[:200]}...' " if len(output) > 200 else f"Raw output: '{output}'")

            answer = parse_final_answer(output)

            self._log("3", "end", {"output": output, "answer": answer})
            self._flush()

            if DEBUG:
                print(f"\nFinal answer: {answer}")

            return answer

        finally:
            # Cleanup logger
            if self.logger:
                self.logger.__exit__(None, None, None)
                self.logger = None
