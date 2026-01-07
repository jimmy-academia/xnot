#!/usr/bin/env python3
"""ANoT - Adaptive Network of Thought (Enhanced).

Three-phase architecture:
1. PLANNING: Schema extraction + 3 LLM calls (conditions → pruning → skeleton)
2. EXPANSION: ReAct-like LWT expansion with tools
3. EXECUTION: Pure LWT execution with async DAG
"""

import os
import json
import re
import time
import asyncio
import threading
import traceback
from typing import Dict, List, Tuple

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.text import Text

from ..base import BaseMethod
from utils.llm import call_llm, call_llm_async
from utils.parsing import parse_script, substitute_variables
from utils.usage import get_usage_tracker

from .prompts import (
    SYSTEM_PROMPT, STEP1_EXTRACT_PROMPT, STEP2_PATH_PROMPT,
    STEP3_RULEOUT_PROMPT, STEP4_SKELETON_PROMPT, PHASE2_PROMPT,
    RANKING_TASK_COMPACT
)
from .helpers import (
    build_execution_layers, format_items_compact, format_schema_compact,
    filter_items_for_ranking, parse_conditions, parse_resolved_path,
    parse_candidates, parse_lwt_skeleton, format_items_for_ruleout, get_attr_value
)
from .tools import (
    tool_read, tool_lwt_list, tool_lwt_get,
    tool_lwt_set, tool_lwt_set_prompt, tool_lwt_delete, tool_lwt_insert,
    tool_lwt_insert_prompt, tool_review_length, tool_update_step, tool_insert_step,
    tool_get_review_lengths, tool_keyword_search, tool_get_review_snippet
)


class AdaptiveNetworkOfThought(BaseMethod):
    """Enhanced Adaptive Network of Thought - three-phase architecture."""

    name = "anot"

    def __init__(self, run_dir: str = None, defense: bool = False, verbose: bool = True, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, verbose=verbose, **kwargs)
        self._thread_local = threading.local()
        self._traces = {}
        self._traces_lock = threading.Lock()
        self._console = Console(force_terminal=True)
        self._live = None
        self._display_rows = {}
        self._display_lock = threading.RLock()
        self._display_title = ""
        self._display_stats = {"complete": 0, "total": 0, "tokens": 0, "cost": 0.0}
        self._last_display_update = 0
        self._errors = []  # Accumulated errors: (request_id, step_idx, error_msg)
        self._debug_log_file = None
        self._debug_log_path = None

        # Always open debug log file (append mode to avoid overwriting during scaling)
        if run_dir:
            self._debug_log_path = os.path.join(run_dir, "debug.log")
            try:
                self._debug_log_file = open(self._debug_log_path, "a", buffering=1)
                from datetime import datetime
                self._debug_log_file.write(f"\n=== ANoT Debug Log @ {datetime.now().isoformat()} ===\n")
                self._debug_log_file.flush()
            except Exception:
                pass  # Silent fail - debug log is optional

    def __del__(self):
        """Close debug log file on cleanup."""
        if hasattr(self, '_debug_log_file') and self._debug_log_file:
            try:
                self._debug_log_file.close()
            except Exception:
                pass

    # =========================================================================
    # Debug/Trace Methods
    # =========================================================================

    def _debug(self, level: int, phase: str, msg: str, content: str = None):
        """Write debug to file only (no terminal output)."""
        if not self._debug_log_file:
            return
        req_id = getattr(self._thread_local, 'request_id', 'R??')
        prefix = f"[{phase}:{req_id}]"
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {prefix} {msg}"

        self._debug_log_file.write(log_line + "\n")
        if content:
            self._debug_log_file.write(f">>> {content}\n")
        self._debug_log_file.flush()

    def _init_trace(self, request_id: str, context: str):
        """Initialize trace for request."""
        trace = {
            "request_id": request_id,
            "context": context,
            "phase1": {"strategy": "", "message": "", "latency_ms": 0},
            "phase2": {"expanded_lwt": [], "react_iterations": 0, "latency_ms": 0},
            "phase3": {"step_results": {}, "top_k": [], "final_output": "", "latency_ms": 0},
        }
        with self._traces_lock:
            self._traces[request_id] = trace

    def _get_trace(self, request_id: str = None) -> dict:
        """Get trace for request."""
        rid = request_id or getattr(self._thread_local, 'request_id', None)
        if not rid:
            return None
        with self._traces_lock:
            return self._traces.get(rid)

    def _update_trace_step(self, idx: str, data: dict):
        """Thread-safe update of trace step result."""
        with self._traces_lock:
            req_id = getattr(self._thread_local, 'request_id', None)
            if req_id and req_id in self._traces:
                self._traces[req_id]["phase3"]["step_results"][idx] = data

    def _save_trace_incremental(self, request_id: str = None):
        """Save current trace to JSONL file incrementally."""
        trace = self._get_trace(request_id)
        if not trace or not self.run_dir:
            return

        trace_path = os.path.join(self.run_dir, "anot_trace.jsonl")
        try:
            with open(trace_path, "a") as f:
                f.write(json.dumps(trace) + "\n")
                f.flush()
        except Exception as e:
            self._debug(1, "TRACE", f"Failed to save trace: {e}")

    def _log_llm_call(self, phase: str, step: str, prompt: str, response: str):
        """Log LLM call details to debug file."""
        if not self._debug_log_file:
            return

        req_id = getattr(self._thread_local, 'request_id', 'R??')
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        self._debug_log_file.write(f"\n{'='*60}\n")
        self._debug_log_file.write(f"[{timestamp}] [{phase}:{req_id}] LLM Call: {step}\n")
        self._debug_log_file.write(f"{'='*60}\n")
        self._debug_log_file.write(f"PROMPT:\n{prompt}\n")
        self._debug_log_file.write(f"{'-'*40}\n")
        self._debug_log_file.write(f"RESPONSE:\n{response}\n")
        self._debug_log_file.write(f"{'='*60}\n\n")
        self._debug_log_file.flush()

    # =========================================================================
    # Cache Methods
    # =========================================================================

    def _get_cache(self) -> dict:
        """Get thread-local step results cache."""
        return getattr(self._thread_local, 'cache', {})

    def _set_cache(self, value: dict):
        """Set thread-local step results cache."""
        self._thread_local.cache = value

    def _cache_set(self, key: str, value):
        """Set value in thread-local cache."""
        cache = self._get_cache()
        cache[key] = value
        self._thread_local.cache = cache

    # =========================================================================
    # Display Methods
    # =========================================================================

    def start_display(self, title: str = "", total: int = 0, requests: list = None):
        """Start rich Live display."""
        self._display_title = title
        self._display_stats = {"complete": 0, "total": total, "tokens": 0, "cost": 0.0}
        self._display_rows = {}
        self._last_display_update = 0

        if requests:
            for req in requests:
                rid = req.get("id", req.get("text", "")[:20])
                ctx = req.get("context") or req.get("text", "")
                self._display_rows[rid] = {"context": ctx, "phase": "---", "status": "pending"}

        self._live = Live(
            self._render_table(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop_display(self):
        """Stop rich Live display and print error summary if any."""
        if self._live:
            self._live.stop()
            self._live = None

        if self._errors:
            print(f"\n⚠️  {len(self._errors)} error(s) during execution:")
            for req_id, step_idx, msg in self._errors:
                print(f"  [{req_id}] Step {step_idx}: {msg}")
            self._errors.clear()

    def _update_display(self, request_id: str, phase: str, status: str, context: str = None):
        """Update display row for request."""
        with self._display_lock:
            was_complete = self._display_rows.get(request_id, {}).get("phase") == "✓"
            if request_id not in self._display_rows:
                self._display_rows[request_id] = {"context": context or "", "phase": phase, "status": status}
            else:
                self._display_rows[request_id]["phase"] = phase
                self._display_rows[request_id]["status"] = status
                if context:
                    self._display_rows[request_id]["context"] = context

            if phase == "✓" and not was_complete:
                self._display_stats["complete"] += 1
                summary = get_usage_tracker().get_summary()
                self._display_stats["tokens"] = summary.get("total_tokens", 0)
                self._display_stats["cost"] = summary.get("total_cost_usd", 0.0)

            if self._live:
                now = time.time()
                if now - self._last_display_update >= 0.1:
                    self._live.update(self._render_table())
                    self._last_display_update = now

    def _render_table(self) -> Table:
        """Build current display table with 4-column layout for compact view."""
        # Calculate dynamic widths based on console width
        console_width = self._console.width or 120
        # With padding=0:
        # Fixed per group: Req(4) + Ph(2) = 6, times 4 groups = 24
        # Plus 3 separators (width=1 each) = 3, total fixed = 27
        # Remaining split: Query gets 2x, St gets 1x (ratio 2:1)
        available = max(40, console_width - 27)  # minimum 40 for flexible
        # 4 groups * (2 + 1) = 12 ratio units
        unit = available // 12
        query_width = max(10, unit * 2)
        status_width = max(6, unit)

        table = Table(title=self._display_title, box=None, padding=0, collapse_padding=True)

        # 4 repeated column groups: Req, Query, Ph, St (with separator)
        for i in range(4):
            table.add_column("Req", style="cyan", width=4, no_wrap=True)
            table.add_column("Query", style="dim", width=query_width, no_wrap=True)
            table.add_column("Ph", style="bold", width=2, justify="center", no_wrap=True)
            table.add_column("St", width=status_width, no_wrap=True)
            if i < 3:
                table.add_column("|", width=1, style="dim")

        def phase_text(phase):
            if phase == "✓":
                return Text("✓", style="green bold")
            elif phase == "P1":
                return Text("1", style="yellow")
            elif phase in ("S1", "S2", "S3", "S4"):
                # Sub-steps within Phase 1
                return Text(phase[1], style="yellow dim")
            elif phase == "P2":
                return Text("2", style="blue")
            elif phase == "P3":
                return Text("3", style="magenta")
            return Text("-", style="dim")

        with self._display_lock:
            items = sorted(self._display_rows.items())
            # Group into rows of 4
            for i in range(0, len(items), 4):
                row_data = []
                for j in range(4):
                    if i + j < len(items):
                        req_id, row = items[i + j]
                        query = row["context"].replace("\n", " ")
                        # Show END of query (more distinctive), adapt to column width
                        q_chars = query_width - 2  # room for ".."
                        q_text = ".." + query[-q_chars:] if len(query) > query_width else query
                        # Compact status
                        status = row["status"][:status_width]
                        row_data.extend([req_id, q_text, phase_text(row["phase"]), status])
                    else:
                        row_data.extend(["", "", "", ""])
                    if j < 3:
                        row_data.append("|")
                table.add_row(*row_data)

        stats = self._display_stats
        footer = f"{stats['complete']}/{stats['total']} | {stats['tokens']:,}tok | ${stats['cost']:.4f}"
        table.caption = footer
        return table

    # =========================================================================
    # Phase 1: Multi-Step Planning (4 steps)
    # =========================================================================

    def _step1_extract_conditions(self, query: str) -> str:
        """Step 1: Extract conditions from user query."""
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "S1", "extract")
        prompt = STEP1_EXTRACT_PROMPT.format(query=query)
        response = call_llm(
            prompt,
            system=SYSTEM_PROMPT,
            role="planner",
            context={"method": "anot", "phase": 1, "step": "step1_extract"}
        )
        self._log_llm_call("P1", "step1_extract", prompt, response)
        return response

    def _step2_resolve_path(self, condition: dict, schema_compact: str, cond_num: int = 0, total_conds: int = 0) -> dict:
        """Step 2: Resolve path for a single condition."""
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "S2", f"path {cond_num}/{total_conds}")
        prompt = STEP2_PATH_PROMPT.format(
            condition_description=f"[{condition['type']}] {condition['description']}",
            schema_compact=schema_compact
        )
        response = call_llm(
            prompt,
            system=SYSTEM_PROMPT,
            role="planner",
            context={"method": "anot", "phase": 1, "step": "step2_resolve"}
        )
        self._log_llm_call("P1", f"step2_resolve_{condition['description'][:20]}", prompt, response)
        resolved = parse_resolved_path(response)
        resolved["description"] = condition["description"]
        resolved["original_type"] = condition["type"]
        # Carry sentiment through for review conditions
        if "sentiment" in condition:
            resolved["sentiment"] = condition["sentiment"]
        return resolved

    def _step3_quick_ruleout(self, hard_conditions: list, items_compact: str, n_items: int) -> list:
        """Step 3: Quick rule-out by checking hard conditions."""
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "S3", "ruleout")
        if not hard_conditions:
            # No hard conditions - all items are candidates
            return list(range(1, n_items + 1))

        # Format hard conditions for prompt
        hard_cond_str = "\n".join([
            f"{i+1}. {c['path']} = {c['expected']}"
            for i, c in enumerate(hard_conditions)
        ])

        prompt = STEP3_RULEOUT_PROMPT.format(
            hard_conditions=hard_cond_str,
            items_compact=items_compact
        )
        response = call_llm(
            prompt,
            system=SYSTEM_PROMPT,
            role="planner",
            context={"method": "anot", "phase": 1, "step": "step3_ruleout"}
        )
        self._log_llm_call("P1", "step3_ruleout", prompt, response)
        candidates = parse_candidates(response)

        # Fallback: if no candidates parsed, use all items
        if not candidates:
            self._debug(1, "P1", "No candidates parsed from ruleout, using all items")
            candidates = list(range(1, n_items + 1))

        return candidates

    def _step4_generate_skeleton(self, candidates: list, soft_conditions: list, k: int) -> list:
        """Step 4: Generate LWT skeleton for soft conditions on candidates."""
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "S4", "skeleton")
        if not candidates:
            # No candidates - return empty LWT with default ranking
            return [("final", "LLM('No candidates passed hard conditions. Output: []')")]

        candidates_str = ", ".join(str(c) for c in candidates)

        if not soft_conditions:
            # No soft conditions - just output candidates ranked
            top_k = candidates[:k]
            return [("final", f"LLM('Candidates: [{candidates_str}]. All passed hard conditions. Output: {top_k}')")]

        # Format soft conditions with sentiment if available
        soft_cond_str = "\n".join([
            f"{i+1}. [REVIEW:{c.get('sentiment', 'POSITIVE').upper()}] {c['description']}"
            if c.get('sentiment') else
            f"{i+1}. [{c.get('original_type', 'REVIEW')}] {c['description']} (search for: {c['expected']})"
            for i, c in enumerate(soft_conditions)
        ])

        # Build soft question for prompt template
        soft_question = soft_conditions[0]['description'] if soft_conditions else "matches criteria"

        prompt = STEP4_SKELETON_PROMPT.format(
            candidates=candidates_str,
            soft_conditions=soft_cond_str,
            soft_question=soft_question,
            k=k
        )
        response = call_llm(
            prompt,
            system=SYSTEM_PROMPT,
            role="planner",
            context={"method": "anot", "phase": 1, "step": "step4_skeleton"}
        )
        self._log_llm_call("P1", "step4_skeleton", prompt, response)

        # Parse LWT skeleton
        skeleton_steps = parse_lwt_skeleton(response)

        if not skeleton_steps:
            # Fallback: generate simple ranking step
            self._debug(1, "P1", "No skeleton parsed, generating fallback")
            top_k = candidates[:k]
            skeleton_steps = [("final", f"LLM('Rank candidates {candidates_str} by reviews. Output top-{k}: {top_k}')")]

        # Ensure all steps are tuples (var_name, step_content)
        normalized = []
        for step in skeleton_steps:
            if isinstance(step, tuple) and len(step) == 2:
                normalized.append(step)
            elif isinstance(step, str):
                # Parse string format "(var)=content"
                match = re.match(r'\((\w+)\)=(.+)', step)
                if match:
                    normalized.append((match.group(1), match.group(2)))
                else:
                    normalized.append(("final", step))
            else:
                self._debug(1, "P1", f"Unexpected step format: {step}")

        return normalized if normalized else skeleton_steps

    def phase1_plan(self, query: str, items: List[dict], k: int = 1) -> Tuple[list, list, list]:
        """Phase 1: Multi-step planning with condition extraction, path resolution,
        quick rule-out, and LWT skeleton generation.

        Args:
            query: User request text (e.g., "Looking for a cafe...")
            items: List of item dicts
            k: Number of top predictions

        Returns:
            Tuple of (candidates, lwt_skeleton, resolved_conditions) for Phase 2/3
        """
        self._debug(1, "P1", f"Multi-step planning for: {query[:60]}...")
        n_items = len(items)

        # Prepare schema for path resolution
        filtered_items = filter_items_for_ranking(items)
        schema_compact = format_schema_compact(filtered_items[:2], num_examples=2, truncate=50)

        # Step 1: Extract conditions
        self._debug(1, "P1", "Step 1: Extracting conditions...")
        conditions_raw = self._step1_extract_conditions(query)
        conditions = parse_conditions(conditions_raw)
        self._debug(1, "P1", f"Extracted {len(conditions)} conditions: {conditions}")

        if not conditions:
            # Fallback: no conditions found, rank all items
            self._debug(1, "P1", "No conditions found, using default ranking")
            all_items = list(range(1, n_items + 1))
            top_k = all_items[:k]
            return all_items, [("final", f"LLM('Rank items 1-{n_items} for query: {query[:100]}. Output top-{k}: {top_k}')")], []

        # Step 2: Resolve paths for each condition
        self._debug(1, "P1", "Step 2: Resolving paths...")
        resolved = []
        for i, cond in enumerate(conditions):
            path_info = self._step2_resolve_path(cond, schema_compact, i+1, len(conditions))
            resolved.append(path_info)
            self._debug(2, "P1", f"  {cond['description']}: {path_info}")

        hard = [r for r in resolved if r['type'] == 'HARD']
        soft = [r for r in resolved if r['type'] == 'SOFT']
        self._debug(1, "P1", f"Conditions: {len(hard)} HARD, {len(soft)} SOFT")

        # Step 3: Quick rule-out (check hard conditions, prune items)
        self._debug(1, "P1", "Step 3: Quick rule-out...")
        if hard:
            items_compact = format_items_for_ruleout(filtered_items, hard)
            candidates = self._step3_quick_ruleout(hard, items_compact, n_items)
        else:
            candidates = list(range(1, n_items + 1))
        self._debug(1, "P1", f"Candidates after rule-out: {candidates} ({len(candidates)}/{n_items})")

        # Step 4: Generate LWT skeleton (soft conditions on candidates only)
        self._debug(1, "P1", "Step 4: Generating LWT skeleton...")
        skeleton_steps = self._step4_generate_skeleton(candidates, soft, k)
        self._debug(1, "P1", f"Generated {len(skeleton_steps)} LWT steps")

        return candidates, skeleton_steps, resolved

    # =========================================================================
    # Phase 2: ReAct LWT Expansion
    # =========================================================================

    def phase2_expand(self, skeleton_steps: list, candidates: list, query: dict, conditions: list = None) -> List[str]:
        """Phase 2: Refine LWT skeleton using ReAct loop with slice syntax for long reviews.

        Phase 2 uses tools to check review lengths, search for keywords, and
        modify LWT steps to use slice notation for truncating long reviews.
        Also handles complex conditions (like hours range checks) by inserting steps.

        Args:
            skeleton_steps: LWT skeleton from Phase 1 [(var_name, step), ...]
            candidates: List of candidate item numbers
            query: Full query dict (for tools)
            conditions: List of resolved conditions from Phase 1

        Returns:
            List of LWT steps (formatted strings)
        """
        self._debug(1, "P2", f"Refining skeleton with {len(skeleton_steps)} steps...")
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "P2", "refining")

        # Convert skeleton to LWT format
        # skeleton_steps is list of (var_name, step_content) tuples
        lwt_steps = []
        for var_name, step_content in skeleton_steps:
            # Format as LWT step: (var_name)=step_content
            if step_content.startswith("LLM("):
                lwt_steps.append(f"({var_name})={step_content}")
            else:
                lwt_steps.append(f"({var_name})={step_content}")

        # Format skeleton for display
        lwt_skeleton_str = "\n".join(lwt_steps)
        self._debug(2, "P2", f"Initial LWT:\n{lwt_skeleton_str}")

        # Format conditions for display
        if conditions:
            conditions_str = "\n".join([
                f"- [{c.get('original_type', 'UNKNOWN')}] {c.get('description', '')} → {c.get('path', '')}"
                for c in conditions
            ])
        else:
            conditions_str = "None"
        self._debug(2, "P2", f"Conditions:\n{conditions_str}")

        # ReAct loop for refinement - check review lengths, handle complex conditions
        prompt = PHASE2_PROMPT.format(conditions=conditions_str, lwt_skeleton=lwt_skeleton_str)
        conversation = [prompt]

        max_iterations = 10
        iteration = 0
        for iteration in range(max_iterations):
            self._debug(2, "P2", f"ReAct iteration {iteration + 1}")
            full_prompt = "\n".join(conversation)

            response = call_llm(
                full_prompt,
                system=SYSTEM_PROMPT,
                role="planner",
                context={"method": "anot", "phase": 2, "step": f"react_{iteration}"}
            )
            self._log_llm_call("P2", f"react_{iteration}", full_prompt, response)

            if not response.strip():
                self._debug(1, "P2", "Empty response, breaking")
                break

            # Process tool calls
            action_results = []

            # Check for lwt_list()
            if "lwt_list()" in response:
                action_results.append(("lwt_list()", tool_lwt_list(lwt_steps)))

            # Process lwt_set() calls (raw step string)
            for match in re.finditer(r'lwt_set\((\d+),\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                step = match.group(2).replace('\\"', '"').replace('\\n', '\n')
                result = tool_lwt_set(int(match.group(1)), step, lwt_steps)
                action_results.append((f"lwt_set({match.group(1)})", result))

            # Process lwt_set_prompt() calls (auto-formatted LLM call)
            for match in re.finditer(r'lwt_set_prompt\((\d+),\s*"([^"]+)",\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                idx, step_id, prompt = int(match.group(1)), match.group(2), match.group(3).replace('\\"', '"').replace('\\n', '\n')
                result = tool_lwt_set_prompt(idx, step_id, prompt, lwt_steps)
                action_results.append((f"lwt_set_prompt({idx}, \"{step_id}\")", result))

            # Process update_step() calls (ID-based, no index - safer)
            for match in re.finditer(r'update_step\("([^"]+)",\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                step_id, prompt = match.group(1), match.group(2).replace('\\"', '"').replace('\\n', '\n')
                result = tool_update_step(step_id, prompt, lwt_steps)
                action_results.append((f"update_step(\"{step_id}\")", result))

            # Process insert_step() calls (ID-based, insert before final)
            for match in re.finditer(r'insert_step\("([^"]+)",\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                step_id, prompt = match.group(1), match.group(2).replace('\\"', '"').replace('\\n', '\n')
                result = tool_insert_step(step_id, prompt, lwt_steps)
                action_results.append((f"insert_step(\"{step_id}\")", result))

            # Process lwt_delete() calls
            for match in re.finditer(r'lwt_delete\((\d+)\)', response):
                result = tool_lwt_delete(int(match.group(1)), lwt_steps)
                action_results.append((f"lwt_delete({match.group(1)})", result))

            # Process lwt_insert() calls (raw step string)
            for match in re.finditer(r'lwt_insert\((\d+),\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                step = match.group(2).replace('\\"', '"').replace('\\n', '\n')
                result = tool_lwt_insert(int(match.group(1)), step, lwt_steps)
                action_results.append((f"lwt_insert({match.group(1)})", result))

            # Process lwt_insert_prompt() calls (auto-formatted LLM call)
            for match in re.finditer(r'lwt_insert_prompt\((\d+),\s*"([^"]+)",\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
                idx, step_id, prompt = int(match.group(1)), match.group(2), match.group(3).replace('\\"', '"').replace('\\n', '\n')
                result = tool_lwt_insert_prompt(idx, step_id, prompt, lwt_steps)
                action_results.append((f"lwt_insert_prompt({idx}, \"{step_id}\")", result))

            # Process review_length() calls (legacy)
            for match in re.finditer(r'review_length\((\d+)\)', response):
                result = tool_review_length(int(match.group(1)), query)
                action_results.append((f"review_length({match.group(1)})", result))

            # Process get_review_lengths() calls (new - per-review lengths)
            for match in re.finditer(r'get_review_lengths\((\d+)\)', response):
                result = tool_get_review_lengths(int(match.group(1)), query)
                action_results.append((f"get_review_lengths({match.group(1)})", result))

            # Process keyword_search() calls
            for match in re.finditer(r'keyword_search\((\d+),\s*"([^"]+)"\)', response):
                result = tool_keyword_search(int(match.group(1)), match.group(2), query)
                action_results.append((f"keyword_search({match.group(1)}, \"{match.group(2)}\")", result))

            # Process get_review_snippet() calls
            for match in re.finditer(r'get_review_snippet\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', response):
                result = tool_get_review_snippet(
                    int(match.group(1)), int(match.group(2)),
                    int(match.group(3)), int(match.group(4)), query
                )
                action_results.append((f"get_review_snippet({match.group(1)}, {match.group(2)}, {match.group(3)}, {match.group(4)})", result))

            # Process read() calls
            for match in re.finditer(r'read\("([^"]+)"\)', response):
                result = tool_read(match.group(1), query)
                if len(result) > 2000:
                    result = result[:2000] + "... (truncated)"
                action_results.append((f"read(\"{match.group(1)}\")", result))

            # Check for done()
            if "done()" in response.lower():
                self._debug(1, "P2", f"ReAct done after {iteration + 1} iterations")
                break

            if action_results:
                # ReAct format: append response and observation
                obs_text = action_results[0][1] if len(action_results) == 1 else "\n".join([f"{name}: {result}" for name, result in action_results])
                conversation.append(f"\n{response}\nObservation: {obs_text}\n")
            else:
                self._debug(1, "P2", "No action found, prompting for action")
                conversation.append(f"\n{response}\n\nPlease output Action: with a tool call (get_review_lengths, keyword_search, lwt_set, lwt_delete, or done):")

        self._debug(1, "P2", f"Refined LWT: {len(lwt_steps)} steps")

        trace = self._get_trace()
        if trace:
            trace["phase2"]["expanded_lwt"] = lwt_steps
            trace["phase2"]["react_iterations"] = iteration + 1

        return lwt_steps

    # =========================================================================
    # Phase 3: Pure LWT Execution
    # =========================================================================

    async def _execute_step_async(self, idx: str, instr: str, items: dict, user_query: str) -> Tuple[str, str]:
        """Execute a single LWT step asynchronously."""
        # Convert double braces to single (from LWT template escaping)
        instr = instr.replace('{{', '{').replace('}}', '}')
        filled = substitute_variables(instr, items, user_query, self._get_cache())
        self._debug(3, "P3", f"Step {idx} filled:", filled)

        start = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        try:
            result = await call_llm_async(
                filled,
                system=SYSTEM_PROMPT,
                role="worker",
                context={"method": "anot", "phase": 3, "step": idx},
                return_usage=True
            )
            output = result["text"]
            prompt_tokens = result["prompt_tokens"]
            completion_tokens = result["completion_tokens"]
        except Exception as e:
            output = "NO"
            error_msg = f"{type(e).__name__}: {str(e)}"
            req_id = getattr(self._thread_local, 'request_id', 'R??')
            self._errors.append((req_id, idx, error_msg))
            self._debug(1, "P3", f"ERROR in step {idx}: {e}", content=traceback.format_exc())

        latency = (time.time() - start) * 1000
        self._log_llm_call("P3", f"step_{idx}", filled, output)
        self._debug(2, "P3", f"Step {idx}: {output[:50]}... ({latency:.0f}ms)")

        step_data = {
            "output": output[:100] if len(output) > 100 else output,
            "latency_ms": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if output == "NO" and 'error_msg' in locals():
            step_data["error"] = error_msg
        self._update_trace_step(idx, step_data)

        return idx, output

    def _compute_partial_match_ranking(self, cache: dict, n_items: int, k: int = 5) -> str:
        """Compute partial match ranking from condition step outputs.

        For each item, count how many condition sets contain it.
        Return top-k items ranked by count descending.
        """
        # Extract condition results (keys starting with 'c')
        condition_sets = []
        for key, value in cache.items():
            if str(key).startswith('c') and key != 'final':
                # Parse indices from output like "[1, 2, 3]" or "1, 2, 3"
                indices = set()
                for match in re.finditer(r'\b(\d+)\b', str(value)):
                    idx = int(match.group(1))
                    if 1 <= idx <= n_items:
                        indices.add(idx)
                if indices:  # Only add non-empty condition sets
                    condition_sets.append(indices)

        if not condition_sets:
            self._debug(2, "P3", "No valid condition sets found, using default ranking")
            return ", ".join(str(i) for i in range(1, min(k, n_items) + 1))

        # Count matches per item
        scores = {}
        for item_idx in range(1, n_items + 1):
            score = sum(1 for cond_set in condition_sets if item_idx in cond_set)
            scores[item_idx] = score

        # Rank by score descending
        ranked = sorted(scores.keys(), key=lambda x: -scores[x])
        top_k = ranked[:k]

        self._debug(2, "P3", f"Partial match scores: {len(condition_sets)} conditions, top={top_k[:5]}")
        return ", ".join(str(idx) for idx in top_k)

    async def _execute_parallel(self, lwt: str, data_for_sub: dict, user_query: str, n_items: int = 50, k: int = 5) -> str:
        """Execute LWT with DAG parallel execution."""
        self._set_cache({})
        steps = parse_script(lwt)

        if not steps:
            self._debug(1, "P3", "ERROR: No valid steps in LWT")
            return ""

        layers = build_execution_layers(steps)
        self._debug(1, "P3", f"Executing {len(steps)} steps in {len(layers)} layers...")

        final = ""
        for layer in layers:
            tasks = [self._execute_step_async(idx, instr, data_for_sub, user_query) for idx, instr in layer]
            results = await asyncio.gather(*tasks)
            for idx, output in results:
                self._cache_set(idx, output)
                final = output

        return final

    def phase3_execute(self, lwt: str, items: dict, user_query: str, n_items: int = 50, k: int = 5) -> str:
        """Execute the LWT script.

        Args:
            lwt: The LWT script to execute
            items: Restaurant data dict (for {(input)} substitution)
            user_query: User's request text (for {(query)} substitution)
            n_items: Number of items being ranked (for partial match)
            k: Number of top items to return
        """
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "P3", "executing")

        # Use pipeline data directly - already {"1": {...}, "2": {...}} with reviews
        raw_items = items.get('items', items)
        if isinstance(raw_items, dict):
            query_dict = raw_items  # Use directly - already 1-indexed string keys with reviews
        else:
            # Fallback for list format
            query_dict = {str(i + 1): item for i, item in enumerate(raw_items)}

        self._debug(2, "P3", f"Query: {len(query_dict)} items")

        try:
            return asyncio.run(self._execute_parallel(lwt, query_dict, user_query, n_items, k))
        except RuntimeError:
            # Already in async context, run sequentially
            self._set_cache({})
            steps = parse_script(lwt)
            if not steps:
                return ""

            final = ""
            for idx, instr in steps:
                filled = substitute_variables(instr, query_dict, user_query, self._get_cache())
                output = call_llm(
                    filled,
                    system=SYSTEM_PROMPT,
                    role="worker",
                    context={"method": "anot", "phase": 3, "step": idx}
                )
                self._log_llm_call("P3", f"step_{idx}", filled, output)
                self._cache_set(idx, output)
                final = output

            return final

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def evaluate(self, query, context: str) -> int:
        """Single item evaluation (not used for ranking)."""
        return 0

    def evaluate_ranking(self, query, context, k: int = 1, request_id: str = "R01") -> str:
        """Ranking evaluation: Phase 1 → Phase 2 → Phase 3.

        Args:
            query: User request text (e.g., "Looking for a cafe...")
            context: Restaurant data dict {"items": {...}} or JSON string
            k: Number of top predictions
            request_id: Request identifier for tracing
        """
        self._init_trace(request_id, query)
        self._thread_local.request_id = request_id

        # Parse restaurant data from context (not query!)
        if isinstance(context, str):
            data = json.loads(context)
        else:
            data = context

        # Extract items - handle both list and dict formats
        raw_items = data.get('items', [data]) if isinstance(data, dict) else [data]
        if isinstance(raw_items, dict):
            # Dict format: {"1": item1, "2": item2, ...} - convert to list ordered by key
            items = [raw_items[k] for k in sorted(raw_items.keys(), key=lambda x: int(x))]
        else:
            items = raw_items
        n_items = len(items)

        self._debug(1, "INIT", f"Ranking {n_items} items for: {query[:60]}...")
        self._update_display(request_id, "---", "starting", query)
        trace = self._get_trace()

        # Phase 1: Multi-step planning (condition extraction → path resolution → rule-out → skeleton)
        self._update_display(request_id, "P1", "planning")
        p1_start = time.time()
        candidates, skeleton_steps, resolved_conditions = self.phase1_plan(query, items, k)
        p1_latency = (time.time() - p1_start) * 1000

        if trace:
            trace["phase1"]["candidates"] = candidates
            trace["phase1"]["skeleton_steps"] = len(skeleton_steps)
            trace["phase1"]["conditions"] = len(resolved_conditions)
            trace["phase1"]["latency_ms"] = p1_latency
            self._save_trace_incremental(request_id)

        # Phase 2: Refine skeleton with review length handling and complex condition handling
        # Pass conditions so LLM can add computation steps (e.g., hours range checks)
        self._update_display(request_id, "P2", "refining")
        p2_start = time.time()
        expanded_lwt_steps = self.phase2_expand(skeleton_steps, candidates, data, resolved_conditions)
        p2_latency = (time.time() - p2_start) * 1000
        expanded_lwt = "\n".join(expanded_lwt_steps)

        if trace:
            trace["phase2"]["latency_ms"] = p2_latency
            self._save_trace_incremental(request_id)

        # Phase 3: Execute with items data and user query text
        self._update_display(request_id, "P3", "executing")
        p3_start = time.time()
        output = self.phase3_execute(expanded_lwt, data, query, n_items=n_items, k=k)
        p3_latency = (time.time() - p3_start) * 1000

        # Parse final output (LLM outputs 1-indexed)
        indices = []
        for match in re.finditer(r'\b(\d+)\b', output):
            idx = int(match.group(1))
            if 1 <= idx <= n_items and idx not in indices:
                indices.append(idx)

        if not indices:
            indices = list(range(1, min(k, n_items) + 1))  # 1-indexed fallback

        top_k = [str(idx) for idx in indices[:k]]  # Already 1-indexed
        self._debug(1, "P3", f"Final ranking: {','.join(top_k)}")

        if trace:
            trace["phase3"]["top_k"] = [int(x) for x in top_k]
            trace["phase3"]["final_output"] = output[:500]
            trace["phase3"]["latency_ms"] = p3_latency
            self._save_trace_incremental(request_id)

        self._update_display(request_id, "✓", ",".join(top_k))
        self._thread_local.request_id = None

        return ", ".join(top_k)


def create_method(run_dir: str = None, defense: bool = False, debug: bool = False):
    """Factory function to create ANoT instance."""
    return AdaptiveNetworkOfThought(run_dir=run_dir, defense=defense, verbose=debug)
