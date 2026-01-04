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
from typing import Optional, Dict, List, Any, Tuple

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.text import Text

from .base import BaseMethod
from utils.llm import call_llm, call_llm_async
from utils.parsing import parse_script, substitute_variables
from utils.usage import get_usage_tracker

# =============================================================================
# Constants and Utilities (moved from shared.py - anot-specific)
# =============================================================================

SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested."


def extract_dependencies(instruction: str) -> set:
    """Extract step IDs referenced in instruction (e.g., {(0)}, {(5.agg)}, {(final)})."""
    matches = re.findall(r'\{\(([a-zA-Z0-9_.]+)\)\}', instruction)
    return set(matches)


def build_execution_layers(steps: list) -> list:
    """Group steps into layers that can run in parallel.

    Returns list of layers, where each layer is [(idx, instr), ...].
    Steps in the same layer have no dependencies on each other.
    """
    if not steps:
        return []

    # Build dependency graph
    step_deps = {}
    for idx, instr in steps:
        step_deps[idx] = extract_dependencies(instr)

    # Assign steps to layers using topological sort
    layers = []
    assigned = set()

    while len(assigned) < len(steps):
        current_layer = []
        for idx, instr in steps:
            if idx in assigned:
                continue
            deps = step_deps[idx]
            if deps <= assigned:
                current_layer.append((idx, instr))

        if not current_layer:
            remaining = [(idx, instr) for idx, instr in steps if idx not in assigned]
            if remaining:
                unresolved = [idx for idx, _ in remaining]
                raise ValueError(f"Cycle detected in LWT dependencies. Unresolved steps: {unresolved}")
            break

        layers.append(current_layer)
        for idx, _ in current_layer:
            assigned.add(idx)

    return layers


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_items_compact(items: list) -> str:
    """Format items as one line each with key=value pairs.

    Example output:
    Item 0: "Tria Cafe" - HasTV=False, GoodForKids=False, DriveThru=None, WiFi=free
    Item 1: "Front Street" - HasTV=False, GoodForKids=True, DriveThru=None, WiFi=free

    Rules:
    - One line per item
    - Include item name
    - Flatten attributes to key=value pairs
    - Use None for missing attributes
    - For complex nested values (dicts), just show key=<dict>
    """
    lines = []
    for i, item in enumerate(items):
        name = item.get("item_name", f"Item {i}")
        attrs = item.get("attributes", {})
        hours = item.get("hours", {})

        # Flatten attributes
        attr_parts = []
        for k, v in sorted(attrs.items()):
            # Simplify complex values
            if isinstance(v, dict):
                attr_parts.append(f"{k}=<dict>")
            elif isinstance(v, str) and len(v) > 20:
                attr_parts.append(f"{k}={v[:15]}...")
            else:
                attr_parts.append(f"{k}={v}")

        # Add hours summary if present
        if hours:
            days = list(hours.keys())
            if days:
                attr_parts.append(f"hours={','.join(days[:3])}...")

        attrs_str = ", ".join(attr_parts) if attr_parts else "(no attributes)"
        lines.append(f'Item {i}: "{name}" - {attrs_str}')

    return "\n".join(lines)


# =============================================================================
# Prompts
# =============================================================================

PHASE1_PROMPT = """Analyze the user request and rank items.

[USER REQUEST]
{context}

[ITEMS]
{items_compact}

[TASK]
1. Extract conditions (e.g., DriveThru=True, GoodForKids=True, HasTV=False)
2. Find which items match ALL conditions
3. Output the matching item indices

[OUTPUT FORMAT]
===LWT_SKELETON===
(final)=LLM("User wants: {context}. Item(s) that match: [LIST INDICES]. Output the best index.")

===MESSAGE===
CONDITIONS: <list>
REMAINING: <indices of matching items>
NEEDS_EXPANSION: no
"""

PHASE2_PROMPT = """Check if the LWT skeleton needs expansion, then call done().

[MESSAGE FROM PHASE 1]
{message}

[CURRENT LWT]
{lwt_skeleton}

[TOOLS]
- done() → finish (call this when skeleton is complete)
- lwt_insert(idx, "step") → add step (only if needed)
- read(path) → get data (only if needed)

[DECISION]
Look at NEEDS_EXPANSION in the message:
- If "no" → just call done() now
- If "yes" → use tools to add steps, then done()

What is your action?
"""


# =============================================================================
# Phase 2 Tools
# =============================================================================

def tool_read(path: str, data: dict) -> str:
    """Read full value at path."""
    def resolve_path(p: str, d):
        if not p:
            return d
        # Parse path like items[2].item_data[0].review
        parts = re.split(r'\.|\[|\]', p)
        parts = [x for x in parts if x]
        val = d
        for part in parts:
            try:
                if isinstance(val, list) and part.isdigit():
                    val = val[int(part)]
                elif isinstance(val, dict):
                    val = val.get(part)
                else:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
        return val

    result = resolve_path(path, data)
    if result is None:
        return f"Error: path '{path}' not found"
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def tool_lwt_list(lwt_steps: List[str]) -> str:
    """Show current LWT steps with indices."""
    if not lwt_steps:
        return "(empty)"
    lines = []
    for i, step in enumerate(lwt_steps):
        lines.append(f"{i}: {step}")
    return "\n".join(lines)


def tool_lwt_get(idx: int, lwt_steps: List[str]) -> str:
    """Get step at index."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    return lwt_steps[idx]


def tool_lwt_set(idx: int, step: str, lwt_steps: List[str]) -> str:
    """Replace step at index. Returns status."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    lwt_steps[idx] = step
    return f"Replaced step at index {idx}"


def tool_lwt_delete(idx: int, lwt_steps: List[str]) -> str:
    """Delete step at index. Returns status."""
    if idx < 0 or idx >= len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)-1})"
    deleted = lwt_steps.pop(idx)
    return f"Deleted step at index {idx}"


def tool_lwt_insert(idx: int, step: str, lwt_steps: List[str]) -> str:
    """Insert step at index (shifts others down). Returns status."""
    if idx < 0 or idx > len(lwt_steps):
        return f"Error: index {idx} out of range (0-{len(lwt_steps)})"
    lwt_steps.insert(idx, step)
    return f"Inserted at index {idx}"


# =============================================================================
# ANoT Implementation
# =============================================================================

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

        # Always open debug log file (overwrites each run)
        if run_dir:
            self._debug_log_path = os.path.join(run_dir, "debug.log")
            try:
                self._debug_log_file = open(self._debug_log_path, "w", buffering=1)  # overwrite, line buffered
                from datetime import datetime
                self._debug_log_file.write(f"=== ANoT Debug Log @ {datetime.now().isoformat()} ===\n")
                self._debug_log_file.flush()
            except Exception as e:
                pass  # Silent fail - debug log is optional

    def __del__(self):
        """Close debug log file on cleanup."""
        if hasattr(self, '_debug_log_file') and self._debug_log_file:
            try:
                self._debug_log_file.close()
            except Exception:
                pass

    def _debug(self, level: int, phase: str, msg: str, content: str = None):
        """Write debug to file only (no terminal output)."""
        if not self._debug_log_file:
            return
        req_id = getattr(self._thread_local, 'request_id', 'R??')
        prefix = f"[{phase}:{req_id}]"
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {prefix} {msg}"

        # Write to debug log file only (full detail)
        self._debug_log_file.write(log_line + "\n")
        if content:
            self._debug_log_file.write(f">>> {content}\n")
        self._debug_log_file.flush()

    def _init_trace(self, request_id: str, context: str):
        """Initialize trace for request."""
        trace = {
            "request_id": request_id,
            "context": context,
            "phase1": {"skeleton": [], "message": "", "latency_ms": 0},
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

    def _get_cache(self) -> dict:
        """Get thread-local step results cache."""
        return getattr(self._thread_local, 'cache', {})

    def _set_cache(self, value: dict):
        """Set thread-local step results cache."""
        self._thread_local.cache = value

    def _cache_get(self, key: str, default=None):
        """Get value from thread-local cache."""
        return self._get_cache().get(key, default)

    def _cache_set(self, key: str, value):
        """Set value in thread-local cache."""
        cache = self._get_cache()
        cache[key] = value
        self._thread_local.cache = cache

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

        # Print accumulated errors
        if self._errors:
            print(f"\n⚠️  {len(self._errors)} error(s) during execution:")
            for req_id, step_idx, msg in self._errors:
                print(f"  [{req_id}] Step {step_idx}: {msg}")
            self._errors.clear()  # Clear for next run

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
        """Build current display table."""
        table = Table(title=self._display_title, box=None, padding=(0, 1))
        table.add_column("Req", style="cyan", width=5)
        table.add_column("Context", style="dim", width=35, overflow="ellipsis")
        table.add_column("Phase", style="bold", width=6, justify="center")
        table.add_column("Status", width=15)

        with self._display_lock:
            for req_id, row in sorted(self._display_rows.items()):
                phase = row["phase"]
                if phase == "✓":
                    phase_text = Text("✓", style="green bold")
                elif phase == "P1":
                    phase_text = Text("P1", style="yellow")
                elif phase == "P2":
                    phase_text = Text("P2", style="blue")
                elif phase == "P3":
                    phase_text = Text("P3", style="magenta")
                else:
                    phase_text = Text("---", style="dim")

                ctx = row["context"][:32] + "..." if len(row["context"]) > 35 else row["context"]
                table.add_row(req_id, ctx, phase_text, row["status"])

        stats = self._display_stats
        footer = f"Progress: {stats['complete']}/{stats['total']} | Tokens: {stats['tokens']:,} | ${stats['cost']:.4f}"
        table.caption = footer
        return table

    # =========================================================================
    # Phase 1: Planning (LWT Skeleton + Message)
    # =========================================================================

    def phase1_plan(self, context: str, items: List[dict]) -> Tuple[List[str], str]:
        """Phase 1: Generate LWT skeleton and message for Phase 2.

        Args:
            context: User's natural language request
            items: List of item dicts with attributes

        Returns:
            (lwt_skeleton, message) tuple where:
            - lwt_skeleton: List of LWT step strings
            - message: Natural language explanation for Phase 2
        """
        self._debug(1, "P1", f"Planning for: {context[:60]}...")

        # Format items compactly - one line per item
        items_compact = format_items_compact(items)
        self._debug(2, "P1", f"Compact items:\n{items_compact[:500]}...")

        prompt = PHASE1_PROMPT.format(
            context=context,
            items_compact=items_compact
        )

        response = call_llm(
            prompt,
            system=SYSTEM_PROMPT,
            role="planner",
            context={"method": "anot", "phase": 1, "step": "plan"}
        )

        self._log_llm_call("P1", "plan", prompt, response)
        self._debug(3, "P1", "Plan response:", response)

        # Parse response by delimiters
        lwt_skeleton = []
        message = ""

        # Extract LWT_SKELETON section
        if "===LWT_SKELETON===" in response:
            skel_start = response.index("===LWT_SKELETON===") + len("===LWT_SKELETON===")
            skel_end = response.find("===MESSAGE===", skel_start) if "===MESSAGE===" in response else len(response)
            skeleton_str = response[skel_start:skel_end].strip()
            # Parse skeleton into list of steps
            for line in skeleton_str.split("\n"):
                line = line.strip()
                if line and line.startswith("("):
                    lwt_skeleton.append(line)

        # Extract MESSAGE section
        if "===MESSAGE===" in response:
            msg_start = response.index("===MESSAGE===") + len("===MESSAGE===")
            message = response[msg_start:].strip()

        self._debug(1, "P1", f"LWT skeleton: {len(lwt_skeleton)} steps")
        self._debug(2, "P1", f"Skeleton:\n" + "\n".join(lwt_skeleton[:5]) + ("..." if len(lwt_skeleton) > 5 else ""))
        self._debug(2, "P1", f"Message:\n{message[:300]}...")

        return lwt_skeleton, message

    # =========================================================================
    # Phase 2: ReAct LWT Expansion
    # =========================================================================

    def phase2_expand(self, lwt_skeleton: List[str], message: str, query: dict) -> List[str]:
        """Phase 2: Expand LWT skeleton using ReAct loop with LWT manipulation tools.

        Args:
            lwt_skeleton: List of LWT step strings from Phase 1
            message: Natural language message from Phase 1
            query: Full data dict for read() tool

        Returns:
            Expanded list of LWT step strings
        """
        self._debug(1, "P2", f"ReAct expansion: {len(lwt_skeleton)} initial steps...")
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "P2", "ReAct expand")

        # External LWT list (modified by tools)
        lwt_steps = list(lwt_skeleton)  # Copy to avoid mutation

        # Build initial prompt with skeleton
        skeleton_str = "\n".join(f"{i}: {step}" for i, step in enumerate(lwt_steps)) if lwt_steps else "(empty)"
        prompt = PHASE2_PROMPT.format(
            message=message,
            lwt_skeleton=skeleton_str
        )

        # Conversation history for ReAct
        conversation = [prompt]

        max_iterations = 50
        for iteration in range(max_iterations):
            self._debug(2, "P2", f"ReAct iteration {iteration + 1}")

            # Build full prompt
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

            # Find FIRST action in response (true ReAct: one action per turn)
            action_result = None
            action_type = None

            # Check for done() first
            if "done()" in response.lower():
                self._debug(1, "P2", f"ReAct done after {iteration + 1} iterations")
                break

            # Check for lwt_list()
            if "lwt_list()" in response:
                action_type = "lwt_list"
                action_result = tool_lwt_list(lwt_steps)
                self._debug(2, "P2", f"lwt_list() → {len(lwt_steps)} steps")

            # Check for lwt_get(idx)
            elif match := re.search(r'lwt_get\((\d+)\)', response):
                action_type = "lwt_get"
                idx = int(match.group(1))
                action_result = tool_lwt_get(idx, lwt_steps)
                self._debug(2, "P2", f"lwt_get({idx}) → {action_result[:50]}...")

            # Check for lwt_set(idx, step)
            elif match := re.search(r'lwt_set\((\d+),\s*"(.+?)"\)', response, re.DOTALL):
                action_type = "lwt_set"
                idx = int(match.group(1))
                step = match.group(2).replace('\\"', '"').replace('\\n', '\n')
                action_result = tool_lwt_set(idx, step, lwt_steps)
                self._debug(2, "P2", f"lwt_set({idx}) → {action_result}")

            # Check for lwt_delete(idx)
            elif match := re.search(r'lwt_delete\((\d+)\)', response):
                action_type = "lwt_delete"
                idx = int(match.group(1))
                action_result = tool_lwt_delete(idx, lwt_steps)
                self._debug(2, "P2", f"lwt_delete({idx}) → {action_result}")

            # Check for lwt_insert(idx, step)
            elif match := re.search(r'lwt_insert\((\d+),\s*"(.+?)"\)', response, re.DOTALL):
                action_type = "lwt_insert"
                idx = int(match.group(1))
                step = match.group(2).replace('\\"', '"').replace('\\n', '\n')
                action_result = tool_lwt_insert(idx, step, lwt_steps)
                self._debug(2, "P2", f"lwt_insert({idx}) → {action_result}")

            # Check for read(path)
            elif match := re.search(r'read\("([^"]+)"\)', response):
                action_type = "read"
                path = match.group(1)
                action_result = tool_read(path, query)
                # Truncate long results
                if len(action_result) > 2000:
                    action_result = action_result[:2000] + "... (truncated)"
                self._debug(2, "P2", f"read({path}) → {action_result[:100]}...")

            # If action found, add response + result to conversation
            if action_result is not None:
                conversation.append(f"\n{response}\n\nRESULT: {action_result}\n\nContinue:")
            else:
                # No recognizable action - ask LLM to try again or done
                self._debug(1, "P2", "No action found, prompting for action")
                conversation.append(f"\n{response}\n\nNo valid action found. Use lwt_list(), lwt_get(idx), lwt_set(idx, step), lwt_delete(idx), lwt_insert(idx, step), read(path), or done():")

        # Return expanded LWT
        self._debug(1, "P2", f"Expanded LWT: {len(lwt_steps)} steps after {iteration + 1} iterations")

        trace = self._get_trace()
        if trace:
            trace["phase2"]["expanded_lwt"] = lwt_steps
            trace["phase2"]["react_iterations"] = iteration + 1

        return lwt_steps

    # =========================================================================
    # Phase 3: Pure LWT Execution
    # =========================================================================

    async def _execute_step_async(self, idx: str, instr: str, query: dict, context: str) -> Tuple[str, str]:
        """Execute a single LWT step asynchronously."""
        filled = substitute_variables(instr, query, context, self._get_cache())
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
        self._debug(2, "P3", f"Step {idx}: {output[:50]}... ({latency:.0f}ms, {prompt_tokens}+{completion_tokens} tok)")

        # Thread-safe trace update
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

    async def _execute_parallel(self, lwt: str, query: dict, context: str) -> str:
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
            tasks = [self._execute_step_async(idx, instr, query, context) for idx, instr in layer]
            results = await asyncio.gather(*tasks)
            for idx, output in results:
                self._cache_set(idx, output)
                final = output

        return final

    def phase3_execute(self, lwt: str, query: dict, context: str) -> str:
        """Execute the LWT script."""
        req_id = getattr(self._thread_local, 'request_id', None)
        if req_id:
            self._update_display(req_id, "P3", "executing")

        try:
            return asyncio.run(self._execute_parallel(lwt, query, context))
        except RuntimeError:
            # Already in async context, run sequentially
            self._set_cache({})
            steps = parse_script(lwt)
            if not steps:
                return ""

            final = ""
            for idx, instr in steps:
                filled = substitute_variables(instr, query, context, self._get_cache())
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

    def evaluate_ranking(self, query, context: str, k: int = 1, request_id: str = "R00") -> str:
        """Ranking evaluation: Phase 1 → Phase 2 → Phase 3.

        Phase 1: Generate LWT skeleton + message (planning)
        Phase 2: Expand LWT using ReAct tools (expansion)
        Phase 3: Execute LWT with DAG parallelism (execution)
        """
        self._init_trace(request_id, context)
        self._thread_local.request_id = request_id

        # Parse data
        if isinstance(query, str):
            data = json.loads(query)
        else:
            data = query
        items = data.get('items', [data]) if isinstance(data, dict) else [data]
        n_items = len(items)

        self._debug(1, "INIT", f"Ranking {n_items} items for: {context[:60]}...")
        self._update_display(request_id, "---", "starting", context)
        trace = self._get_trace()

        # Phase 1: Plan (LWT skeleton + message)
        self._update_display(request_id, "P1", "planning")
        p1_start = time.time()
        lwt_skeleton, message = self.phase1_plan(context, items)
        p1_latency = (time.time() - p1_start) * 1000

        if trace:
            trace["phase1"]["skeleton"] = lwt_skeleton
            trace["phase1"]["message"] = message[:500] if message else ""
            trace["phase1"]["latency_ms"] = p1_latency
            self._save_trace_incremental(request_id)

        # Phase 2: Expand LWT using ReAct tools
        self._update_display(request_id, "P2", "expanding")
        p2_start = time.time()
        expanded_lwt_steps = self.phase2_expand(lwt_skeleton, message, data)
        p2_latency = (time.time() - p2_start) * 1000
        expanded_lwt = "\n".join(expanded_lwt_steps)

        if trace:
            trace["phase2"]["latency_ms"] = p2_latency
            self._save_trace_incremental(request_id)

        # Phase 3: Execute
        self._update_display(request_id, "P3", "executing")
        p3_start = time.time()
        output = self.phase3_execute(expanded_lwt, data, context)
        p3_latency = (time.time() - p3_start) * 1000

        # Parse final output to get ranking
        indices = []
        for match in re.finditer(r'\b(\d+)\b', output):
            idx = int(match.group(1))
            if 0 <= idx < n_items and idx not in indices:
                indices.append(idx)

        # If no valid indices, fallback to first k items
        if not indices:
            indices = list(range(min(k, n_items)))

        # Convert to 1-indexed for output
        top_k = [str(idx + 1) for idx in indices[:k]]

        self._debug(1, "P3", f"Final ranking: {','.join(top_k)}")

        if trace:
            trace["phase3"]["top_k"] = [int(x) for x in top_k]
            trace["phase3"]["final_output"] = output[:500]
            trace["phase3"]["latency_ms"] = p3_latency
            self._save_trace_incremental(request_id)

        self._update_display(request_id, "✓", ",".join(top_k))
        self._thread_local.request_id = None

        return ", ".join(top_k)


# =============================================================================
# Factory
# =============================================================================

def create_method(run_dir: str = None, defense: bool = False, debug: bool = False):
    """Factory function to create ANoT instance."""
    return AdaptiveNetworkOfThought(run_dir=run_dir, defense=defense, verbose=debug)
