#!/usr/bin/env python3
"""ANoT - Adaptive Network of Thought.

Key innovation: Analyzes context to identify evidence types (metadata/hours/reviews)
and generates LWT scripts tailored to the specific evidence sources.
"""

import os
import json
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

from .shared import (
    DEBUG,
    SYSTEM_PROMPT,
    call_llm,
    substitute_variables,
    parse_script,
    parse_final_answer,
)

# =============================================================================
# Rich Console for Debug Output
# =============================================================================

ANOT_THEME = Theme({
    "phase": "bold magenta",
    "subphase": "bold cyan",
    "context": "yellow",
    "script": "green",
    "step": "bold blue",
    "output": "white",
    "time": "dim",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
})
console = Console(theme=ANOT_THEME, force_terminal=True)


# =============================================================================
# Prompts
# =============================================================================

CONTEXT_ANALYSIS_PROMPT = """Analyze this user request to understand what they need.

Request: {context}

Identify:
1. What conditions must be satisfied? (list each one)
2. For each condition, where would you find evidence?
   - METADATA: attributes like WiFi, NoiseLevel, OutdoorSeating, Alcohol, RestaurantsPriceRange2, DogsAllowed, BikeParking, Ambience, HasTV
   - HOURS: time-based constraints (open on specific day, available during specific hours)
   - REVIEWS: subjective qualities mentioned in customer reviews (good matcha, aesthetic, cozy, latte art, books)

Output format:
CONDITIONS:
- [condition 1]: [METADATA/HOURS/REVIEWS] - [what to check]
- [condition 2]: [METADATA/HOURS/REVIEWS] - [what to check]
...

LOGIC: AND (all must be satisfied)
"""

SCRIPT_GENERATION_PROMPT = """You need to generate an LWT script to evaluate a restaurant.

The user wants:
{context_analysis}

Available data keys:
- Attributes: {attribute_keys}
- Hours: {available_days}
- Reviews: {review_count} reviews

Script format - each line must be exactly like this:
(0)=LLM("instruction here")
(1)=LLM("another instruction")

Use these placeholders in your instructions:
- {{(input)}}[attributes][KeyName] - to access an attribute value
- {{(input)}}[hours][Monday] - to access hours for a day
- {{(input)}}[item_data][0][review] - to access review text (0-indexed)
- {{(N)}} - to reference result of step N

Example script for "quiet cafe with WiFi":
(0)=LLM("Check {{(input)}}[attributes][NoiseLevel]. Is it quiet? Output: 1=yes, -1=no, 0=unclear")
(1)=LLM("Check {{(input)}}[attributes][WiFi]. Is it free? Output: 1=yes, -1=no, 0=unclear")
(2)=LLM("Results: noise={{(0)}}, wifi={{(1)}}. If any -1 output -1. If all 1 output 1. Else 0")

Now write the script (just the numbered lines, nothing else):
"""

# =============================================================================
# Prompts for Ranking Mode
# =============================================================================

RANKING_CONTEXT_ANALYSIS_PROMPT = """Analyze this user request to understand what they want in a restaurant.

Request: {context}

Identify:
1. What are the key requirements? (list each one)
2. How should restaurants be compared?
   - METADATA: attributes to compare (WiFi, NoiseLevel, Price, etc.)
   - HOURS: time-based requirements
   - REVIEWS: qualities to look for in reviews

Output format:
REQUIREMENTS:
- [requirement 1]: [priority: HIGH/MEDIUM/LOW]
- [requirement 2]: [priority: HIGH/MEDIUM/LOW]
...

COMPARISON FOCUS: [what matters most for ranking]
"""

DIRECT_RANKING_PROMPT = """You are comparing multiple restaurants to find the BEST match for the user.

User wants: {context}

Key requirements:
{context_analysis}

Here are the restaurants to compare:
{query}

For each restaurant, evaluate how well it matches the requirements.
Consider:
- Which requirements are satisfied?
- Which are not satisfied or unclear?
- Overall fit for the user's needs

Output ONLY the index number (1, 2, 3, etc.) of the BEST matching restaurant.
If multiple are equally good, pick the first one.

Output:"""


# =============================================================================
# ANoT Implementation
# =============================================================================

class AdaptiveNetworkOfThought:
    """Adaptive Network of Thought - context-aware LWT script generation."""

    def __init__(self, run_dir: str = None, debug: bool = False):
        self.run_dir = run_dir
        self.debug = debug or DEBUG
        self.cache = {}
        self.context_cache = {}  # Cache for context analysis results

    def phase1_analyze_context(self, context: str) -> str:
        """Phase 1: Analyze context to identify conditions and evidence sources."""
        # Check cache first
        if context in self.context_cache:
            if self.debug:
                console.print(Panel("PHASE 1: Context Analysis (CACHED)", style="phase"))
                console.print(self.context_cache[context])
                console.rule()
            return self.context_cache[context]

        prompt = CONTEXT_ANALYSIS_PROMPT.format(context=context)

        if self.debug:
            console.print(Panel("PHASE 1: Context Analysis", style="phase"))
            console.print("[dim]Prompt:[/dim]")
            console.print(prompt, style="dim")

        start = time.time()
        analysis = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        # Cache the result
        self.context_cache[context] = analysis

        if self.debug:
            console.print(f"\n[time]Duration: {duration:.2f}s[/time]")
            console.print("[subphase]Analysis Result:[/subphase]")
            console.print(analysis)
            console.rule()

        return analysis

    def phase1b_analyze_query(self, query: dict) -> dict:
        """Phase 1b: Analyze query structure (deterministic, no LLM)."""
        attributes = query.get("attributes", {})
        hours = query.get("hours", {})
        reviews = query.get("item_data", [])

        # Calculate review stats
        review_lengths = [len(r.get("review", "")) for r in reviews]
        avg_length = sum(review_lengths) / max(len(review_lengths), 1)

        info = {
            "attribute_keys": list(attributes.keys()),
            "has_hours": bool(hours),
            "available_days": list(hours.keys()) if hours else [],
            "review_count": len(reviews),
            "avg_review_length": avg_length,
        }

        if self.debug:
            console.print(Panel("PHASE 1b: Query Structure", style="phase"))
            table = Table(show_header=True)
            table.add_column("Type", style="cyan")
            table.add_column("Available Data", style="white")
            table.add_row("Attributes", ", ".join(info['attribute_keys']) or "(none)")
            table.add_row("Hours", ", ".join(info['available_days']) or "(none)")
            table.add_row("Reviews", f"{info['review_count']} reviews (avg {info['avg_review_length']:.0f} chars)")
            console.print(table)
            console.rule()

        return info

    def phase2_direct_evaluate(self, context_analysis: str, query_info: dict,
                                query: dict, context: str) -> str:
        """Phase 2: Direct evaluation based on context analysis (bypasses script generation)."""
        # Build a direct evaluation prompt
        attrs = query.get("attributes", {})
        hours = query.get("hours", {})
        reviews = query.get("item_data", [])[:3]  # First 3 reviews

        review_texts = [r.get("review", "")[:500] for r in reviews]

        prompt = f"""Evaluate this restaurant for the user's request.

User wants: {context}

Conditions to check:
{context_analysis}

Restaurant data:
- Attributes: {json.dumps(attrs, indent=2)}
- Hours: {json.dumps(hours, indent=2)}
- Sample reviews: {review_texts}

SCORING RULES:
- Output 1 (RECOMMEND): Most important conditions are satisfied, no major dealbreakers
- Output 0 (UNCLEAR): Mixed signals, missing data, or partially satisfied conditions
- Output -1 (NOT RECOMMEND): A critical condition is clearly violated

IMPORTANT:
- If data is missing for a condition, treat it as UNCLEAR (lean toward 0)
- Only output -1 if you have clear evidence against a critical requirement
- Consider the overall fit, not just individual conditions

Output ONLY: -1, 0, or 1"""

        if self.debug:
            console.print(Panel("PHASE 2: Direct Evaluation", style="phase"))
            console.print(f"[dim]Prompt (first 500 chars):[/dim]")
            console.print(f"[dim]{prompt[:500]}...[/dim]")

        start = time.time()
        result = call_llm(prompt, system=SYSTEM_PROMPT, role="worker")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(f"[subphase]Result:[/subphase] {result}")
            console.rule()

        return result

    def _execute_step_sync(self, idx: str, instr: str, query: dict, context: str) -> str:
        """Execute a single step synchronously."""
        filled = substitute_variables(instr, query, context, self.cache)

        if self.debug:
            console.print(f"\n[step]Step ({idx})[/step]")
            console.print(f"  [dim]Instruction:[/dim] {instr}")
            console.print(f"  [dim]Filled:[/dim] {filled[:300]}{'...' if len(filled) > 300 else ''}")

        try:
            start = time.time()
            output = call_llm(filled, system=SYSTEM_PROMPT, role="worker")
            duration = time.time() - start
        except Exception as e:
            output = "0"
            duration = 0
            if self.debug:
                console.print(f"  [error]Error: {e}[/error]")

        if self.debug:
            console.print(f"  [output]→ {output}[/output] [time]({duration:.2f}s)[/time]")

        return output

    def execute_script(self, script: str, query: dict, context: str) -> str:
        """Phase 3: Execute LWT script step by step."""
        self.cache = {}
        steps = parse_script(script)

        if not steps:
            # Fallback to direct LLM call
            if self.debug:
                console.print("[warning]No steps parsed, using fallback[/warning]")
            return self._fallback_direct(query, context)

        if self.debug:
            console.print(Panel(f"PHASE 3: Execution ({len(steps)} steps)", style="phase"))

        final = ""
        for idx, instr in steps:
            output = self._execute_step_sync(idx, instr, query, context)
            self.cache[idx] = output
            final = output

        if self.debug:
            console.rule()

        return final

    def _fallback_direct(self, query: dict, context: str) -> str:
        """Fallback when script parsing fails."""
        prompt = f"""Based on restaurant data:
{json.dumps(query, indent=2)[:2000]}

User wants: {context}

Should this restaurant be recommended?
Output ONLY: -1 (no), 0 (unclear), or 1 (yes)"""

        return call_llm(prompt, system=SYSTEM_PROMPT, role="worker")

    def solve(self, query, context: str) -> int:
        """Full pipeline: analyze → evaluate directly."""
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT SOLVE[/bold white]", style="on blue"))
            console.print(f"[context]Context:[/context] {context}")
            console.print(f"[context]Item:[/context] {query.get('item_name', 'Unknown')}")
            console.rule()

        # Phase 1: Analyze context (what to check)
        context_analysis = self.phase1_analyze_context(context)

        # Phase 1b: Analyze query structure (what data is available)
        query_info = self.phase1b_analyze_query(query)

        # Phase 2: Direct evaluation (bypasses script generation)
        output = self.phase2_direct_evaluate(context_analysis, query_info, query, context)

        # Parse final answer
        answer = parse_final_answer(output)

        if self.debug:
            if answer == 1:
                console.print(Panel(f"[success]Final Answer: {answer} (RECOMMEND)[/success]", style="green"))
            elif answer == -1:
                console.print(Panel(f"[error]Final Answer: {answer} (NOT RECOMMEND)[/error]", style="red"))
            else:
                console.print(Panel(f"[warning]Final Answer: {answer} (UNCLEAR)[/warning]", style="yellow"))
            console.print()

        return answer

    # =========================================================================
    # Ranking Mode Methods
    # =========================================================================

    def phase1_analyze_context_ranking(self, context: str) -> str:
        """Phase 1 for ranking: Analyze context for comparison criteria."""
        # Check cache first
        cache_key = f"ranking:{context}"
        if cache_key in self.context_cache:
            if self.debug:
                console.print(Panel("PHASE 1: Ranking Context Analysis (CACHED)", style="phase"))
                console.print(self.context_cache[cache_key])
                console.rule()
            return self.context_cache[cache_key]

        prompt = RANKING_CONTEXT_ANALYSIS_PROMPT.format(context=context)

        if self.debug:
            console.print(Panel("PHASE 1: Ranking Context Analysis", style="phase"))

        start = time.time()
        analysis = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        # Cache the result
        self.context_cache[cache_key] = analysis

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(analysis)
            console.rule()

        return analysis

    def phase2_direct_ranking(self, context: str, context_analysis: str, query: str) -> str:
        """Phase 2 for ranking: Direct comparison of all restaurants."""
        # Truncate query if too long (keep first 8000 chars)
        query_truncated = query[:8000] if len(query) > 8000 else query

        prompt = DIRECT_RANKING_PROMPT.format(
            context=context,
            context_analysis=context_analysis,
            query=query_truncated,
        )

        if self.debug:
            console.print(Panel("PHASE 2: Direct Ranking", style="phase"))
            console.print(f"[dim]Query length: {len(query)} chars[/dim]")

        start = time.time()
        result = call_llm(prompt, system=SYSTEM_PROMPT, role="worker")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(f"[subphase]Result:[/subphase] {result}")
            console.rule()

        return result

    def _parse_ranking_indices(self, output: str, k: int = 1) -> list:
        """Parse ranking output to extract indices."""
        import re
        numbers = re.findall(r'\b(\d+)\b', output)
        indices = []
        seen = set()
        for n in numbers:
            idx = int(n)
            if 1 <= idx <= 50 and idx not in seen:
                seen.add(idx)
                indices.append(idx)
                if len(indices) >= k:
                    break
        return indices if indices else [1]

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking evaluation: returns string with top-k indices.

        Args:
            query: All restaurants formatted with indices (string)
            context: User request text
            k: Number of top predictions to return

        Returns:
            String with top-k indices (e.g., "3" or "3, 1, 5")
        """
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT RANKING[/bold white]", style="on magenta"))
            console.print(f"[context]Context:[/context] {context}")
            console.print(f"[context]k:[/context] {k}")
            console.rule()

        # Phase 1: Analyze context for ranking criteria
        context_analysis = self.phase1_analyze_context_ranking(context)

        # Phase 2: Direct ranking comparison
        output = self.phase2_direct_ranking(context, context_analysis, query)

        # Parse indices from output
        indices = self._parse_ranking_indices(output, k)
        result = ", ".join(str(i) for i in indices)

        if self.debug:
            console.print(Panel(f"[success]Ranking Result: {result}[/success]", style="green"))
            console.print()

        return result


# =============================================================================
# Factory and Method Interface
# =============================================================================

_instance = None


def create_method(run_dir: str = None, debug: bool = False):
    """Factory function to create ANoT instance.

    Returns the AdaptiveNetworkOfThought instance which has both:
    - solve(query, context) for per-item evaluation
    - evaluate_ranking(query, context, k) for ranking evaluation
    """
    global _instance
    if _instance is None:
        _instance = AdaptiveNetworkOfThought(run_dir=run_dir, debug=debug)
    return _instance


def method(query, context: str) -> int:
    """Default ANoT method (per-item evaluation)."""
    return create_method().solve(query, context)
