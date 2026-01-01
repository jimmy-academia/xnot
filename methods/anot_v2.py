#!/usr/bin/env python3
"""ANoT v2 - Adaptive Network of Thought with Ranking Support.

Key changes from v1:
- Added evaluate_ranking() for ranking mode
- Ranking-specific script generation prompts
- Returns string with indices for ranking tasks
"""

import os
import json
import re
import time
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

from .base import BaseMethod
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
# Prompts for Per-Item Mode
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

RANKING_SCRIPT_PROMPT = """Generate a script to RANK restaurants and find the BEST match.

User wants: {context}
Analysis: {context_analysis}

CRITICAL RULES:
1. The input {{(input)}} is TEXT listing restaurants with [N] markers - DO NOT use dict-access like {{(input)}}[attributes]
2. Each instruction MUST be on a SINGLE LINE - no line breaks inside LLM("...")
3. The FINAL step MUST output ONLY a single index number

Script format (each on ONE line):
(0)=LLM("single line instruction here with {{(input)}}")
(1)=LLM("single line instruction referencing {{(0)}}")

Example for "quiet cafe with good coffee":
(0)=LLM("For each restaurant [N] in this list, extract noise level and coffee quality mentions. List: {{(input)}}. Format: [N]: noise=X, coffee=Y")
(1)=LLM("From {{(0)}}, which restaurant best matches quiet and good coffee? Output ONLY the index number.")

Now write the script (each step on ONE line):
"""


# =============================================================================
# ANoT v2 Implementation
# =============================================================================

class AdaptiveNetworkOfThoughtV2(BaseMethod):
    """Adaptive Network of Thought v2 - with ranking support."""

    name = "anot_v2"

    def __init__(self, run_dir: str = None, defense: bool = False, debug: bool = False, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)
        self.debug = debug or DEBUG
        self.cache = {}

    # =========================================================================
    # Per-Item Mode (same as v1)
    # =========================================================================

    def phase1_analyze_context(self, context: str) -> str:
        """Phase 1: Analyze context to identify conditions and evidence sources."""
        prompt = CONTEXT_ANALYSIS_PROMPT.format(context=context)

        if self.debug:
            console.print(Panel("PHASE 1: Context Analysis", style="phase"))

        start = time.time()
        analysis = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(analysis)
            console.rule()

        return analysis

    def phase1b_analyze_query(self, query: dict) -> dict:
        """Phase 1b: Analyze query structure (deterministic, no LLM)."""
        attributes = query.get("attributes", {})
        hours = query.get("hours", {})
        reviews = query.get("item_data", [])

        info = {
            "attribute_keys": list(attributes.keys()),
            "has_hours": bool(hours),
            "available_days": list(hours.keys()) if hours else [],
            "review_count": len(reviews),
        }

        if self.debug:
            console.print(Panel("PHASE 1b: Query Structure", style="phase"))
            table = Table(show_header=True)
            table.add_column("Type", style="cyan")
            table.add_column("Available Data", style="white")
            table.add_row("Attributes", ", ".join(info['attribute_keys']) or "(none)")
            table.add_row("Hours", ", ".join(info['available_days']) or "(none)")
            table.add_row("Reviews", f"{info['review_count']} reviews")
            console.print(table)
            console.rule()

        return info

    def phase2_generate_script(self, context_analysis: str, query_info: dict) -> str:
        """Phase 2: Generate LWT script tailored to evidence types."""
        prompt = SCRIPT_GENERATION_PROMPT.format(
            context_analysis=context_analysis,
            attribute_keys=", ".join(query_info["attribute_keys"]) or "(none)",
            available_days=", ".join(query_info["available_days"]) or "(none)",
            review_count=query_info["review_count"],
        )

        if self.debug:
            console.print(Panel("PHASE 2: Script Generation", style="phase"))

        start = time.time()
        script = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(f"[subphase]Generated Script:[/subphase]")
            console.print(script)
            console.rule()

        return script

    def _execute_step(self, idx: str, instr: str, query, context: str) -> str:
        """Execute a single step."""
        filled = substitute_variables(instr, query, context, self.cache)

        if self.debug:
            console.print(f"\n[step]Step ({idx})[/step]")
            console.print(f"  [dim]Filled:[/dim] {filled[:300]}...")

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

    def execute_script(self, script: str, query, context: str) -> str:
        """Phase 3: Execute LWT script step by step."""
        self.cache = {}
        steps = parse_script(script)

        if not steps:
            if self.debug:
                console.print("[warning]No steps parsed, using fallback[/warning]")
            return self._fallback_direct(query, context)

        if self.debug:
            console.print(Panel(f"PHASE 3: Execution ({len(steps)} steps)", style="phase"))

        final = ""
        for idx, instr in steps:
            output = self._execute_step(idx, instr, query, context)
            self.cache[idx] = output
            final = output

        return final

    def _fallback_direct(self, query, context: str) -> str:
        """Fallback when script parsing fails."""
        query_str = json.dumps(query, indent=2)[:2000] if isinstance(query, dict) else str(query)[:2000]
        prompt = f"""Based on restaurant data:
{query_str}

User wants: {context}

Should this restaurant be recommended?
Output ONLY: -1 (no), 0 (unclear), or 1 (yes)"""

        return call_llm(prompt, system=SYSTEM_PROMPT, role="worker")

    def evaluate(self, query, context: str) -> int:
        """Per-item evaluation: returns -1, 0, or 1."""
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT v2 EVALUATE[/bold white]", style="on blue"))
            console.print(f"[context]Context:[/context] {context}")

        # Phase 1: Analyze context
        context_analysis = self.phase1_analyze_context(context)

        # Phase 1b: Analyze query structure
        if isinstance(query, dict):
            query_info = self.phase1b_analyze_query(query)
        else:
            query_info = {"attribute_keys": [], "available_days": [], "review_count": 0}

        # Phase 2: Generate script
        script = self.phase2_generate_script(context_analysis, query_info)

        # Phase 3: Execute script
        output = self.execute_script(script, query, context)

        # Parse final answer
        answer = parse_final_answer(output)

        if self.debug:
            console.print(Panel(f"Final Answer: {answer}", style="green" if answer == 1 else "red" if answer == -1 else "yellow"))

        return answer

    # =========================================================================
    # Ranking Mode (NEW in v2)
    # =========================================================================

    def phase1_analyze_context_ranking(self, context: str) -> str:
        """Phase 1 for ranking: Analyze context for comparison criteria."""
        prompt = RANKING_CONTEXT_ANALYSIS_PROMPT.format(context=context)

        if self.debug:
            console.print(Panel("PHASE 1: Context Analysis (Ranking)", style="phase"))

        start = time.time()
        analysis = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(analysis)
            console.rule()

        return analysis

    def phase1b_analyze_ranking_query(self, query: str) -> dict:
        """Phase 1b for ranking: Analyze the text-formatted restaurant list."""
        # Count restaurants by finding [N] markers
        indices = re.findall(r'\[(\d+)\]', query)
        restaurant_count = len(set(indices))

        # Detect what info is present in the text
        query_lower = query.lower()
        has_reviews = "review" in query_lower or "said" in query_lower
        has_hours = "hours" in query_lower or "open" in query_lower
        has_attributes = any(attr in query_lower for attr in
            ["wifi", "noise", "parking", "outdoor", "price", "alcohol"])

        # Detect potential issues
        avg_len_per_restaurant = len(query) / max(restaurant_count, 1)
        potential_heterogeneity = avg_len_per_restaurant > 2000

        info = {
            "restaurant_count": restaurant_count,
            "has_reviews": has_reviews,
            "has_hours": has_hours,
            "has_attributes": has_attributes,
            "potential_heterogeneity": potential_heterogeneity,
            "query_length": len(query),
        }

        if self.debug:
            console.print(Panel("PHASE 1b: Ranking Query Structure", style="phase"))
            table = Table(show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            for k, v in info.items():
                table.add_row(k, str(v))
            console.print(table)
            console.rule()

        return info

    def phase2_generate_ranking_script(self, context: str, context_analysis: str) -> str:
        """Phase 2 for ranking: Generate comparison script."""
        prompt = RANKING_SCRIPT_PROMPT.format(
            context=context,
            context_analysis=context_analysis,
        )

        if self.debug:
            console.print(Panel("PHASE 2: Ranking Script Generation", style="phase"))

        start = time.time()
        script = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print(script)
            console.rule()

        return script

    def _fallback_ranking(self, query: str, context: str) -> str:
        """Fallback for ranking when script fails."""
        prompt = f"""Given these restaurants:
{query[:3000]}

User wants: {context}

Which restaurant (by index number) is the BEST match?
Output ONLY the index number (e.g., 1, 2, 3, etc.)"""

        return call_llm(prompt, system=SYSTEM_PROMPT, role="worker")

    def _parse_ranking_indices(self, output: str, k: int = 1) -> List[int]:
        """Parse ranking output to extract indices."""
        # Try to find numbers in output
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
            query: All restaurants formatted with indices
            context: User request text
            k: Number of top predictions to return

        Returns:
            String with top-k indices (e.g., "3" or "3, 1, 5")
        """
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT v2 RANKING[/bold white]", style="on magenta"))
            console.print(f"[context]Context:[/context] {context}")
            console.print(f"[context]k:[/context] {k}")
            console.rule()

        # Phase 1: Analyze context for ranking criteria
        context_analysis = self.phase1_analyze_context_ranking(context)

        # Phase 1b: Analyze ranking query structure
        query_info = self.phase1b_analyze_ranking_query(query)

        # Phase 2: Generate ranking script
        script = self.phase2_generate_ranking_script(context, context_analysis)

        # Phase 3: Execute script
        steps = parse_script(script)

        if not steps:
            if self.debug:
                console.print("[warning]No steps parsed, using fallback[/warning]")
            output = self._fallback_ranking(query, context)
        else:
            if self.debug:
                console.print(Panel(f"PHASE 3: Ranking Execution ({len(steps)} steps)", style="phase"))

            self.cache = {}
            for idx, instr in steps:
                # For ranking, query is a string, not dict
                filled = instr.replace("{(input)}", query).replace("{(context)}", context)
                # Substitute previous step results
                for cache_idx, cache_val in self.cache.items():
                    filled = filled.replace(f"{{({cache_idx})}}", str(cache_val))

                if self.debug:
                    console.print(f"\n[step]Step ({idx})[/step]")
                    console.print(f"  [dim]Instruction:[/dim] {instr[:200]}...")

                try:
                    output = call_llm(filled, system=SYSTEM_PROMPT, role="worker")
                except Exception as e:
                    output = "1"
                    if self.debug:
                        console.print(f"  [error]Error: {e}[/error]")

                self.cache[idx] = output

                if self.debug:
                    console.print(f"  [output]→ {output[:200]}[/output]")

            output = self.cache.get(str(len(steps) - 1), "1")

        # Parse indices from output
        indices = self._parse_ranking_indices(output, k)
        result = ", ".join(str(i) for i in indices)

        if self.debug:
            console.print(Panel(f"Ranking Result: {result}", style="success"))
            console.print()

        return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_method(run_dir: str = None, defense: bool = False, debug: bool = False):
    """Factory function to create ANoT v2 instance."""
    return AdaptiveNetworkOfThoughtV2(run_dir=run_dir, defense=defense, debug=debug)
