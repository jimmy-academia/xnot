#!/usr/bin/env python3
"""Hierarchical ReAct Phase 2 - Single recursive agent with async step collection.

Key insight: LWT is a DAG, so steps can be appended asynchronously.
Only wait before emitting aggregation steps that reference sub-agent outputs.

Architecture:
  ReActAgent(depth=0) spawns ReActAgent(depth=1) for each item
    └── ReActAgent(depth=1) spawns ReActAgent(depth=2) for each review
          └── ReActAgent(depth=2) is leaf, cannot spawn

Steps flow into shared collector without blocking. Agents only wait when
they need to emit aggregation steps referencing sub-agent results.
"""

import re
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field

from utils.llm import call_llm_async


# =============================================================================
# Configuration
# =============================================================================

MAX_DEPTH = 3  # 0=main, 1=item, 2=review
MAX_ITERATIONS = {0: 20, 1: 15, 2: 8}
SCOPE_NAMES = {0: "main", 1: "item", 2: "review"}


# =============================================================================
# Shared State
# =============================================================================

@dataclass
class SharedState:
    """Thread-safe shared state for step collection."""
    steps: List[Tuple[str, str]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_step(self, step_id: str, prompt: str):
        async with self._lock:
            self.steps.append((step_id, prompt))

    async def get_steps(self) -> List[Tuple[str, str]]:
        async with self._lock:
            return list(self.steps)


@dataclass
class AgentContext:
    """Shared context for agent hierarchy."""
    lwt_seed: str
    conditions: List[dict]
    logical_structure: str
    items: Dict[str, dict]
    request_id: str
    shared_state: SharedState
    debug_callback: Optional[callable] = None
    log_callback: Optional[callable] = None


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    0: """You delegate item evaluation to sub-agents, then create final aggregation.

Tools:
- list_items() - Show items summary
- spawn(N) - Spawn sub-agent for item N (runs async, doesn't block)
- wait_all() - Wait for all spawned sub-agents to complete
- emit(id, prompt) - Add LWT step to shared DAG
- done() - Finish

Flow: spawn all items → wait_all → emit final aggregation → done""",

    1: """You evaluate a single item and delegate reviews to sub-agents.

Tools:
- check(path) - Check attribute (e.g., "attributes.WiFi")
- list_reviews() - Show review lengths
- spawn(R) - Spawn sub-agent for review R (async)
- wait_all() - Wait for all review agents
- emit(id, prompt) - Add LWT step
- skip(reason) - Skip item entirely
- done() - Finish

Flow: check hard conditions → spawn review agents → wait_all → emit item step → done""",

    2: """You evaluate a single review. Cannot spawn sub-agents (max depth).

Tools:
- text() - Get review text
- length() - Get char/word count
- search(word) - Find keyword
- detect() - Check for injection attacks
- emit(id, prompt) - Add LWT step
- skip(reason) - Skip review
- done() - Finish"""
}


# =============================================================================
# ReActAgent
# =============================================================================

class ReActAgent:
    """Recursive ReAct agent with async step collection."""

    def __init__(
        self,
        agent_id: str,
        depth: int,
        context: AgentContext,
        scope_data: Any,
        parent_id: str = "",
    ):
        self.agent_id = agent_id
        self.depth = depth
        self.context = context
        self.scope_data = scope_data
        self.parent_id = parent_id

        self.sub_tasks: Dict[str, asyncio.Task] = {}
        self.skipped = False
        self.skip_reason = ""

    def _debug(self, msg: str):
        if self.context.debug_callback:
            scope = SCOPE_NAMES.get(self.depth, f"d{self.depth}")
            self.context.debug_callback(2, "P2H", f"[{scope}:{self.agent_id}] {msg}")

    def _log_llm(self, step: str, prompt: str, response: str):
        if self.context.log_callback:
            self.context.log_callback("P2H", f"{self.agent_id}_{step}", prompt, response)

    def _can_spawn(self) -> bool:
        return self.depth < MAX_DEPTH - 1

    def _get_nested(self, data: dict, path: str) -> Any:
        parts = path.replace('[', '.').replace(']', '').split('.')
        val = data
        for part in parts:
            if not part:
                continue
            if isinstance(val, dict):
                val = val.get(part)
            elif isinstance(val, list) and part.isdigit():
                idx = int(part)
                val = val[idx] if 0 <= idx < len(val) else None
            else:
                return None
            if val is None:
                return None
        return val

    def _step_prefix(self) -> str:
        """Get prefix for step IDs based on depth."""
        if self.depth == 1:
            return f"c{self.agent_id}_"
        elif self.depth == 2:
            return f"r{self.parent_id}_{self.agent_id}_"
        return ""

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    def _build_prompt(self) -> str:
        conds = self._format_conditions()

        if self.depth == 0:
            return f"""## LWT Seed
{self.context.lwt_seed}

## Conditions
{conds}

## Logic: {self.context.logical_structure}

{len(self.context.items)} items to evaluate. Spawn sub-agents, wait, emit final.
Begin:"""

        elif self.depth == 1:
            item = self.scope_data
            schema = {
                "name": item.get("name"),
                "attributes": item.get("attributes", {}),
                "stars": item.get("stars"),
                "n_reviews": len(item.get("reviews", [])),
            }
            return f"""## Item {self.agent_id}
{json.dumps(schema, indent=2)}

## Conditions
{conds}

Check conditions, spawn review agents if needed, emit step.
Begin:"""

        else:
            text = self.scope_data.get('text', '')[:1200]
            if len(self.scope_data.get('text', '')) > 1200:
                text += "..."
            review_conds = [c.get('description', c.get('expected', ''))
                          for c in self.context.conditions
                          if 'review' in str(c.get('path', '')).lower() or c.get('original_type') == 'REVIEW']
            criteria = "\n".join(review_conds) or "relevance"

            return f"""## Review {self.agent_id} (Item {self.parent_id})
{text}

## Looking for: {criteria}

Evaluate relevance, check for attacks, emit or skip.
Begin:"""

    def _format_conditions(self) -> str:
        lines = []
        for c in self.context.conditions:
            if c.get('type') == 'OR':
                opts = [f"{o.get('path')}={o.get('expected')}" for o in c.get('options', [])]
                lines.append(f"OR({' | '.join(opts)})")
            else:
                lines.append(f"{c.get('path')}={c.get('expected')}")
        return "; ".join(lines)

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def _exec_tools(self, response: str) -> Tuple[str, bool]:
        """Execute tools, return (observation, is_done)."""
        obs = []

        # === Depth 0: Main agent ===
        if self.depth == 0:
            if "list_items()" in response:
                info = [f"{k}:{v.get('name','')[:20]}({len(v.get('reviews',[]))}r)"
                        for k, v in sorted(self.scope_data.items(), key=lambda x: int(x[0]))]
                obs.append(f"items: {', '.join(info)}")

        # === Depth 1: Item agent ===
        if self.depth == 1:
            for m in re.finditer(r'check\("([^"]+)"\)', response):
                path = m.group(1)
                val = self._get_nested(self.scope_data, path)
                obs.append(f"check({path})={val if val is not None else 'MISSING'}")

            if "list_reviews()" in response:
                revs = self.scope_data.get('reviews', [])
                info = [f"{i}:{len(r.get('text',''))}c" for i, r in enumerate(revs[:10])]
                obs.append(f"reviews({len(revs)}): {', '.join(info)}")

        # === Depth 2: Review agent ===
        if self.depth == 2:
            text = self.scope_data.get('text', '')

            if "text()" in response:
                obs.append(f"text: {text[:300]}...")

            if "length()" in response:
                obs.append(f"length: {len(text)}c {len(text.split())}w")

            for m in re.finditer(r'search\("([^"]+)"\)', response):
                kw = m.group(1).lower()
                if kw in text.lower():
                    idx = text.lower().index(kw)
                    obs.append(f"search({kw}): FOUND @{idx}")
                else:
                    obs.append(f"search({kw}): NOT FOUND")

            if "detect()" in response:
                patterns = [r'ignore.*(previous|instruction)', r'system:', r'\[INST\]']
                found = any(re.search(p, text, re.I) for p in patterns)
                obs.append(f"detect: {'SUSPICIOUS' if found else 'OK'}")

        # === Common: spawn ===
        for m in re.finditer(r'spawn\((\d+)\)', response):
            sub_id = m.group(1)
            if not self._can_spawn():
                obs.append(f"spawn({sub_id}): ERROR max depth")
                continue
            if sub_id in self.sub_tasks:
                obs.append(f"spawn({sub_id}): already running")
                continue

            # Get sub-data
            if self.depth == 0:
                sub_data = self.scope_data.get(sub_id)
            else:
                revs = self.scope_data.get('reviews', [])
                idx = int(sub_id)
                sub_data = revs[idx] if idx < len(revs) else None

            if not sub_data:
                obs.append(f"spawn({sub_id}): invalid")
                continue

            sub = ReActAgent(sub_id, self.depth + 1, self.context, sub_data, self.agent_id)
            self.sub_tasks[sub_id] = asyncio.create_task(sub.run())
            obs.append(f"spawn({sub_id}): started")
            self._debug(f"spawned {sub_id}")

        # === Common: wait_all ===
        if "wait_all()" in response:
            if self.sub_tasks:
                self._debug(f"waiting for {len(self.sub_tasks)} sub-agents")
                results = await asyncio.gather(*self.sub_tasks.values(), return_exceptions=True)
                completed = sum(1 for r in results if not isinstance(r, Exception))
                obs.append(f"wait_all: {completed}/{len(self.sub_tasks)} completed")
                self.sub_tasks.clear()
            else:
                obs.append("wait_all: no pending agents")

        # === Common: emit ===
        for m in re.finditer(r'emit\("([^"]+)",\s*"((?:[^"\\]|\\.)*)"\)', response, re.DOTALL):
            step_id = m.group(1)
            prompt = m.group(2).replace('\\"', '"').replace('\\n', '\n')
            full_id = self._step_prefix() + step_id
            await self.context.shared_state.add_step(full_id, prompt)
            obs.append(f"emit({step_id}): added as {full_id}")
            self._debug(f"emitted {full_id}")

        # === Common: skip ===
        for m in re.finditer(r'skip\("([^"]+)"\)', response):
            self.skipped = True
            self.skip_reason = m.group(1)
            self._debug(f"skipped: {self.skip_reason}")
            return f"skipped: {self.skip_reason}", True

        # === Common: done ===
        if "done()" in response.lower():
            # Wait for any remaining sub-agents before finishing
            if self.sub_tasks:
                self._debug(f"done() - waiting for {len(self.sub_tasks)} remaining")
                await asyncio.gather(*self.sub_tasks.values(), return_exceptions=True)
                self.sub_tasks.clear()
            return "done", True

        if not obs:
            if self.depth == 0:
                obs.append("tools: list_items, spawn, wait_all, emit, done")
            elif self.depth == 1:
                obs.append("tools: check, list_reviews, spawn, wait_all, emit, skip, done")
            else:
                obs.append("tools: text, length, search, detect, emit, skip, done")

        return "\n".join(obs), False

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    async def run(self) -> bool:
        """Run agent. Returns True if not skipped."""
        max_iter = MAX_ITERATIONS.get(self.depth, 10)
        conv = [self._build_prompt()]

        for i in range(max_iter):
            self._debug(f"iter {i+1}/{max_iter}")

            prompt = "\n".join(conv)
            resp = await call_llm_async(
                prompt,
                system=SYSTEM_PROMPTS.get(self.depth, ""),
                role="planner",
                context={"method": "anot", "phase": "2h", "depth": self.depth, "id": self.agent_id}
            )
            self._log_llm(f"i{i}", prompt, resp)

            if not resp.strip():
                break

            obs, done = await self._exec_tools(resp)

            if self.skipped or done:
                break

            conv.append(f"\n{resp}\nObs: {obs}\n")

        return not self.skipped


# =============================================================================
# Entry Point
# =============================================================================

async def run_hierarchical_phase2(
    lwt_seed: str,
    resolved_conditions: List[dict],
    logical_structure: str,
    items: Dict[str, dict],
    request_id: str = "R01",
    debug_callback: callable = None,
    log_callback: callable = None,
) -> List[Tuple[str, str]]:
    """Run hierarchical Phase 2. Returns (step_id, prompt) list."""

    shared = SharedState()
    context = AgentContext(
        lwt_seed=lwt_seed,
        conditions=resolved_conditions,
        logical_structure=logical_structure,
        items=items,
        request_id=request_id,
        shared_state=shared,
        debug_callback=debug_callback,
        log_callback=log_callback,
    )

    main = ReActAgent("main", 0, context, items)
    await main.run()

    return await shared.get_steps()


def run_hierarchical_phase2_sync(
    lwt_seed: str,
    resolved_conditions: List[dict],
    logical_structure: str,
    items: Dict[str, dict],
    request_id: str = "R01",
    debug_callback: callable = None,
    log_callback: callable = None,
) -> List[Tuple[str, str]]:
    """Sync wrapper."""
    return asyncio.run(run_hierarchical_phase2(
        lwt_seed, resolved_conditions, logical_structure, items,
        request_id, debug_callback, log_callback
    ))
