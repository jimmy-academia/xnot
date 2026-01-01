#!/usr/bin/env python3
"""ANoT Origin - Adaptive Network of Thought following experiment_plan.md exactly.

3-Phase Implementation:
- Phase 1: Seed Workflow Generation (context + schema → template LWT)
- Phase 2: Workflow Adaptation (always runs: structure counting + module arrangement + local adaptation)
- Phase 3: Execution (run adapted LWT script)
"""

import os
import json
import re
import time
from typing import Optional, Dict, List

from rich.console import Console
from rich.panel import Panel
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
# Prompts
# =============================================================================

SEED_WORKFLOW_PROMPT = """Analyze this user request and generate a seed LWT script template.

Request: {context}

Available data schema:
- Attributes: {attribute_keys}
- Hours: {available_days}
- Reviews: {review_count} reviews with fields: review, stars, date, user
- User metadata: name, friends, review_count, elite

For each requirement in the request:
1. Identify the evidence source:
   - ATTR: {{(input)}}[attributes][KeyName] - for WiFi, NoiseLevel, OutdoorSeating, etc.
   - HOURS: {{(input)}}[hours][DayName] - for opening hours
   - REVIEW: {{(input)}}[item_data][N][review] - for review text analysis

2. Generate script steps to check each requirement

Value interpretation:
- NoiseLevel: "u'quiet'" or "'quiet'" means quiet, "u'loud'" or "'loud'" means loud
- WiFi: "'free'" means free WiFi, "'no'" or "None" means no WiFi
- Boolean attributes: "True" or "False" as strings
- Hours: "7:0-19:0" means 7AM-7PM, closed if None

Output format - numbered steps with LLM calls:
(0)=LLM("Check {{(input)}}[attributes][AttrName]. [interpretation guide]. Output: 1=satisfied, -1=not satisfied, 0=unclear")
(1)=LLM("From review {{(input)}}[item_data][0][review], find mentions of [topic]. Output: POSITIVE/NEGATIVE/NONE")
...
(N)=LLM("Aggregate {{(0)}}, {{(1)}}... If any requirement failed output -1. If all met output 1. Else 0")

Output ONLY the script lines, nothing else:
"""

ADAPTATION_PROMPT = """Adapt this seed LWT script for the actual data structure.

Seed script:
{seed_script}

Actual data structure:
- Mode: {mode}
- Restaurants: {num_restaurants}
- Reviews per restaurant: {reviews_info}

Issues detected:
{issues_description}

Generate the complete LWT script with:
1. Correct access paths for the data mode:
   - Single item: {{(input)}}[attributes], {{(input)}}[item_data][N]
   - Ranking: {{(input)}}[items][R][attributes], {{(input)}}[items][R][item_data][N]
2. Steps for each restaurant and each review as needed
3. If issues detected, add handling steps (filter suspicious content, verify results)
4. Final aggregation step that outputs -1, 0, or 1 (single) or restaurant index (ranking)

Output ONLY the numbered script lines:
"""


# =============================================================================
# ANoT Origin Implementation
# =============================================================================

class AdaptiveNetworkOfThoughtOrigin(BaseMethod):
    """Adaptive Network of Thought following experiment_plan.md exactly."""

    name = "anot_origin"

    def __init__(self, run_dir: str = None, defense: bool = False, debug: bool = False, **kwargs):
        super().__init__(run_dir=run_dir, defense=defense, **kwargs)
        self.debug = debug or DEBUG
        self.cache = {}  # Step execution cache
        self.seed_cache = {}  # Cache seed workflows per (context, schema_hash)

    # =========================================================================
    # Phase 1: Seed Workflow Generation
    # =========================================================================

    def extract_schema(self, query) -> dict:
        """Extract data schema from query (deterministic, no LLM).

        Args:
            query: dict (single item) or dict with "items" key (ranking mode)

        Returns:
            Schema dict describing available data
        """
        # Handle ranking mode (query has "items" key)
        if isinstance(query, dict) and "items" in query:
            items = query["items"]
            if not items:
                return self._empty_schema()
            # Use first item for schema
            sample = items[0]
            reviews = sample.get("item_data", [])
        elif isinstance(query, dict):
            # Single item mode
            sample = query
            reviews = query.get("item_data", [])
        else:
            return self._empty_schema()

        # Extract attribute keys
        attributes = sample.get("attributes", {})
        hours = sample.get("hours", {})

        # Extract review fields from first review
        review_fields = []
        user_fields = []
        if reviews:
            first_review = reviews[0]
            review_fields = list(first_review.keys())
            if "user" in first_review and first_review["user"]:
                user_fields = list(first_review["user"].keys())

        return {
            "attribute_keys": list(attributes.keys()),
            "has_hours": bool(hours),
            "available_days": list(hours.keys()) if hours else [],
            "review_count": len(reviews),
            "review_fields": review_fields,
            "user_fields": user_fields,
        }

    def _empty_schema(self) -> dict:
        return {
            "attribute_keys": [],
            "has_hours": False,
            "available_days": [],
            "review_count": 0,
            "review_fields": [],
            "user_fields": [],
        }

    def _schema_hash(self, schema: dict) -> str:
        """Create a hash key for schema caching."""
        return f"attrs:{len(schema['attribute_keys'])}_days:{len(schema['available_days'])}_reviews:{schema['review_count']}"

    def phase1_generate_seed_workflow(self, context: str, schema: dict) -> str:
        """Phase 1: Generate seed LWT from context + schema.

        The seed workflow is a template that will be adapted in Phase 2.
        """
        # Check cache
        cache_key = (context, self._schema_hash(schema))
        if cache_key in self.seed_cache:
            if self.debug:
                console.print("[success]Seed cache HIT[/success]")
            return self.seed_cache[cache_key]

        prompt = SEED_WORKFLOW_PROMPT.format(
            context=context,
            attribute_keys=", ".join(schema["attribute_keys"]) or "(none)",
            available_days=", ".join(schema["available_days"]) or "(none)",
            review_count=schema["review_count"],
        )

        if self.debug:
            console.print(Panel("PHASE 1: Seed Workflow Generation", style="phase"))
            console.print(f"[context]Context:[/context] {context[:100]}...")
            console.print(f"[dim]Schema: {self._schema_hash(schema)}[/dim]")

        start = time.time()
        seed = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        # Cache it
        self.seed_cache[cache_key] = seed

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print("[subphase]Seed Script:[/subphase]")
            console.print(seed)
            console.rule()

        return seed

    # =========================================================================
    # Phase 2: Workflow Adaptation (ALWAYS runs)
    # =========================================================================

    def count_structure(self, query) -> dict:
        """Count restaurants and reviews in the query (deterministic)."""
        if isinstance(query, dict) and "items" in query:
            # Ranking mode
            items = query["items"]
            return {
                "mode": "ranking",
                "num_restaurants": len(items),
                "reviews_per_restaurant": [len(item.get("item_data", [])) for item in items],
            }
        elif isinstance(query, dict):
            # Single item mode
            return {
                "mode": "single",
                "num_restaurants": 1,
                "num_reviews": len(query.get("item_data", [])),
            }
        else:
            # String mode (ranking with text)
            # Count [N] markers
            indices = re.findall(r'\[(\d+)\]', str(query))
            return {
                "mode": "ranking_text",
                "num_restaurants": len(set(indices)),
                "reviews_per_restaurant": [],  # Unknown in text mode
            }

    def detect_issues(self, query) -> dict:
        """Detect heterogeneity and attack patterns (deterministic checks).

        Returns dict with detected issues.
        """
        issues = {
            "has_length_variance": False,
            "long_review_indices": [],
            "has_attack_patterns": False,
            "suspicious_indices": [],
            "detected_patterns": [],
        }

        # Get reviews based on query type
        reviews = []
        if isinstance(query, dict) and "items" in query:
            for item in query["items"]:
                reviews.extend(item.get("item_data", []))
        elif isinstance(query, dict):
            reviews = query.get("item_data", [])

        if not reviews:
            return issues

        # Check length variance
        lengths = [len(r.get("review", "")) for r in reviews]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            for i, length in enumerate(lengths):
                if avg_length > 0 and length > 3 * avg_length:
                    issues["has_length_variance"] = True
                    issues["long_review_indices"].append(i)

        # Check for attack patterns
        attack_patterns = [
            r'\boutput\s*[:=]',
            r'\bignore\s+(previous|above)',
            r'\banswer\s+is\b',
            r'\byou\s+must\b',
            r'\boverride\b',
            r'\bforget\b.+\binstructions\b',
        ]

        for i, r in enumerate(reviews):
            text = r.get("review", "").lower()
            for pattern in attack_patterns:
                if re.search(pattern, text):
                    issues["has_attack_patterns"] = True
                    if i not in issues["suspicious_indices"]:
                        issues["suspicious_indices"].append(i)
                    issues["detected_patterns"].append(pattern)

        return issues

    def phase2_adapt_workflow(self, seed_script: str, query, context: str) -> str:
        """Phase 2: Adapt seed workflow to actual data structure.

        This ALWAYS runs to:
        1. Count structure (restaurants, reviews)
        2. Arrange modules for correct data paths
        3. Detect and handle issues
        """
        structure = self.count_structure(query)
        issues = self.detect_issues(query)

        if self.debug:
            console.print(Panel("PHASE 2: Workflow Adaptation", style="phase"))

            # Structure table
            table = Table(show_header=True, title="Structure")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Mode", structure["mode"])
            table.add_row("Restaurants", str(structure.get("num_restaurants", 1)))
            if structure["mode"] == "ranking":
                table.add_row("Reviews/Restaurant", str(structure.get("reviews_per_restaurant", [])[:5]) + "...")
            else:
                table.add_row("Reviews", str(structure.get("num_reviews", 0)))
            console.print(table)

            # Issues table
            if any([issues["has_length_variance"], issues["has_attack_patterns"]]):
                issues_table = Table(show_header=True, title="Issues Detected")
                issues_table.add_column("Issue", style="cyan")
                issues_table.add_column("Details", style="white")
                if issues["has_length_variance"]:
                    issues_table.add_row("Length Variance", f"Indices: {issues['long_review_indices']}")
                if issues["has_attack_patterns"]:
                    issues_table.add_row("Attack Patterns", f"Indices: {issues['suspicious_indices']}")
                console.print(issues_table)

        # Build issues description for prompt
        issues_desc = "None detected"
        if issues["has_length_variance"] or issues["has_attack_patterns"]:
            parts = []
            if issues["has_length_variance"]:
                parts.append(f"Length variance detected at review indices: {issues['long_review_indices']}")
            if issues["has_attack_patterns"]:
                parts.append(f"Potential attack patterns at indices: {issues['suspicious_indices']}")
            issues_desc = "\n".join(parts)

        # Build reviews info
        if structure["mode"] == "ranking":
            reviews_info = str(structure.get("reviews_per_restaurant", []))
        else:
            reviews_info = str(structure.get("num_reviews", 0))

        prompt = ADAPTATION_PROMPT.format(
            seed_script=seed_script,
            mode=structure["mode"],
            num_restaurants=structure.get("num_restaurants", 1),
            reviews_info=reviews_info,
            issues_description=issues_desc,
        )

        start = time.time()
        adapted = call_llm(prompt, system=SYSTEM_PROMPT, role="planner")
        duration = time.time() - start

        if self.debug:
            console.print(f"[time]Duration: {duration:.2f}s[/time]")
            console.print("[subphase]Adapted Script:[/subphase]")
            console.print(adapted)
            console.rule()

        return adapted

    # =========================================================================
    # Phase 3: Execution
    # =========================================================================

    def _execute_step(self, idx: str, instr: str, query, context: str) -> str:
        """Execute a single LWT step."""
        filled = substitute_variables(instr, query, context, self.cache)

        if self.debug:
            console.print(f"\n[step]Step ({idx})[/step]")
            console.print(f"  [dim]Instruction:[/dim] {instr[:200]}...")
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
            console.print(f"  [output]-> {output}[/output] [time]({duration:.2f}s)[/time]")

        return output

    def phase3_execute(self, script: str, query, context: str) -> str:
        """Phase 3: Execute the adapted LWT script step-by-step."""
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

        if self.debug:
            console.rule()

        return final

    def _fallback_direct(self, query, context: str) -> str:
        """Fallback when script parsing fails."""
        if isinstance(query, dict):
            query_str = json.dumps(query, indent=2)[:2000]
        else:
            query_str = str(query)[:2000]

        prompt = f"""Based on this data:
{query_str}

User wants: {context}

Should this be recommended?
Output ONLY: -1 (no), 0 (unclear), or 1 (yes)"""

        return call_llm(prompt, system=SYSTEM_PROMPT, role="worker")

    # =========================================================================
    # Entry Points
    # =========================================================================

    def evaluate(self, query, context: str) -> int:
        """Single-item evaluation: returns -1, 0, or 1."""
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT Origin EVALUATE[/bold white]", style="on blue"))
            if isinstance(query, dict):
                console.print(f"[context]Item:[/context] {query.get('item_name', 'Unknown')}")
            console.print(f"[context]Context:[/context] {context[:100]}...")
            console.rule()

        # Phase 1: Extract schema and generate seed workflow
        schema = self.extract_schema(query)
        seed_script = self.phase1_generate_seed_workflow(context, schema)

        # Phase 2: Adapt workflow (always runs)
        adapted_script = self.phase2_adapt_workflow(seed_script, query, context)

        # Phase 3: Execute
        output = self.phase3_execute(adapted_script, query, context)

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

    def evaluate_ranking(self, query, context: str, k: int = 1) -> str:
        """Ranking evaluation: returns string with top-k indices.

        Args:
            query: All restaurants - either dict (with "items" key) or string
            context: User request text
            k: Number of top predictions to return

        Returns:
            String with top-k indices (e.g., "3" or "3, 1, 5")
        """
        if self.debug:
            console.print()
            console.print(Panel.fit("[bold white]ANoT Origin RANKING[/bold white]", style="on magenta"))
            console.print(f"[context]Context:[/context] {context[:100]}...")
            console.print(f"[context]k:[/context] {k}")
            console.rule()

        # Extract schema from query (handles both dict and string)
        schema = self.extract_schema(query)

        # Phase 1: Generate seed workflow
        seed_script = self.phase1_generate_seed_workflow(context, schema)

        # Phase 2: Adapt workflow
        adapted_script = self.phase2_adapt_workflow(seed_script, query, context)

        # Phase 3: Execute
        output = self.phase3_execute(adapted_script, query, context)

        # Parse indices from output
        indices = self._parse_ranking_indices(output, k)
        result = ", ".join(str(i) for i in indices)

        if self.debug:
            console.print(Panel(f"[success]Ranking Result: {result}[/success]", style="green"))
            console.print()

        return result

    def _parse_ranking_indices(self, output: str, k: int = 1) -> List[int]:
        """Parse ranking output to extract indices."""
        numbers = re.findall(r'\b(\d+)\b', str(output))
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


# =============================================================================
# Factory Functions
# =============================================================================

def create_method(run_dir: str = None, defense: bool = False, debug: bool = False):
    """Factory function to create ANoT Origin instance."""
    return AdaptiveNetworkOfThoughtOrigin(run_dir=run_dir, defense=defense, debug=debug)
