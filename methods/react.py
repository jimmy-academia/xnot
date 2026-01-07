#!/usr/bin/env python3
"""ReACT (Reasoning and Acting) method for restaurant recommendation.

Reference: ReAct: Synergizing Reasoning and Acting in Language Models
Yao et al., ICLR 2023
https://arxiv.org/abs/2210.03629

Redesigned to use dict mode with path-based data access (uses tool_read from ANoT).
"""

import json
import re
from typing import Tuple, List

from .base import BaseMethod
from .anot.tools import tool_read
from utils.llm import call_llm
from utils.parsing import parse_indices


MAX_STEPS = 5

SYSTEM_PROMPT_RANKING = """You are ranking restaurants for a user. Be efficient - aim to finish in 3-4 steps.

Actions (items are 1-indexed):
- read("items.1.name") → restaurant name
- read("items.1.reviews") → all reviews (text, stars)
- read("items.1.attributes") → NoiseLevel, WiFi, RestaurantsPriceRange2, etc.
- read("items.1.categories") → category list
- finish("3, 1, 5") → submit ranking (comma-separated, best first)

Strategy: Read a few restaurants, compare to user needs, then finish().

Thought: [reasoning]
Action: [one action]"""


class ReAct(BaseMethod):
    """ReACT (Reasoning and Acting) with dict-mode data access."""

    name = "react"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Single item evaluation (not primary use case)."""
        return 0

    def _build_ranking_prompt(self, query: str, n_items: int, history: List[dict], k: int = 1) -> str:
        """Build prompt for ranking task."""
        if k == 1:
            instruction = "Select the restaurant that BEST matches the user's request."
        else:
            instruction = f"Select the TOP {k} restaurants that best match the user's request."

        parts = []
        parts.append(f"[USER REQUEST]")
        parts.append(query)
        parts.append("")
        parts.append(f"[DATA INFO]")
        parts.append(f"There are {n_items} restaurants (items.1 through items.{n_items}).")
        parts.append("Use read(path) to explore restaurant data, then finish(answer) with your ranking.")
        parts.append("")
        parts.append(f"[INSTRUCTION]")
        parts.append(instruction)
        parts.append("")

        if history:
            parts.append("[PREVIOUS STEPS]")
            for i, step in enumerate(history, 1):
                parts.append(f"Step {i}:")
                parts.append(f"Thought: {step['thought']}")
                parts.append(f"Action: {step['action']}")
                parts.append(f"Observation: {step['observation']}")
                parts.append("")

        parts.append("[YOUR TURN]")
        parts.append("Thought: ")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse Thought and Action from response."""
        thought = ""
        action = ""

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.+?)(?=\nThought:|\nObservation:|\Z)", response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _execute_action(self, action: str, data: dict) -> Tuple[str, bool, str]:
        """Execute an action using tool_read from ANoT.

        Returns:
            Tuple of (observation, is_finished, final_answer)
        """
        action = action.strip()

        # Check for finish action
        finish_match = re.search(r'finish\s*\(\s*["\']?([^"\')\]]+)["\']?\s*\)', action, re.IGNORECASE)
        if finish_match:
            answer = finish_match.group(1).strip()
            return f"Final answer submitted: {answer}", True, answer

        # Check for read action - uses tool_read from ANoT
        read_match = re.search(r'read\s*\(\s*["\']([^"\']+)["\']\s*\)', action, re.IGNORECASE)
        if read_match:
            path = read_match.group(1)
            result = tool_read(path, data)
            # Truncate very long results
            if len(result) > 3000:
                result = result[:3000] + "\n... (truncated, use more specific path)"
            return result, False, ""

        # Unknown action - guide user to valid actions
        return (
            f"Unknown action: {action}. Use read(\"path\") or finish(\"answer\").",
            False,
            ""
        )

    def _format_indices(self, indices: list, k: int) -> str:
        """Format indices as comma-separated string."""
        if not indices:
            return "1"
        return ", ".join(str(i) for i in indices[:k])

    def evaluate_ranking(self, query: str, context, k: int = 1, **kwargs) -> str:
        """Evaluate ranking task using ReACT loop with dict-mode data access.

        Args:
            query: User request text
            context: Restaurant data dict {"items": {"1": {...}, ...}} or JSON string
            k: Number of top predictions

        Returns:
            Comma-separated indices (e.g., "3, 1, 5")
        """
        # Parse context
        if isinstance(context, str):
            data = json.loads(context)
        else:
            data = context

        # Get item count
        items = data.get("items", {})
        if isinstance(items, dict):
            n_items = len(items)
        else:
            n_items = len(items) if items else 0

        history = []

        for step in range(MAX_STEPS):
            prompt = self._build_ranking_prompt(query, n_items, history, k)
            response = call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
            self._log_llm_call(f"step_{step+1}", prompt, response, SYSTEM_PROMPT_RANKING)

            thought, action = self._parse_response(response)

            if not action:
                # No action found, default to reading first item
                action = 'read("items.1")'

            observation, is_finished, answer = self._execute_action(action, data)

            if is_finished:
                indices = parse_indices(answer, max_index=n_items, k=k)
                if not indices:
                    indices = parse_indices(response, max_index=n_items, k=k)
                return self._format_indices(indices, k)

            history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

        # Max steps reached - force decision
        prompt = self._build_ranking_prompt(query, n_items, history, k)
        prompt += f"\nYou have reached the maximum steps. You MUST now submit your answer."
        prompt += f"\nThought: Based on my exploration, I will select the best restaurants."
        prompt += f"\nAction: finish(\""

        response = call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
        self._log_llm_call("force_finish", prompt, response, SYSTEM_PROMPT_RANKING)
        indices = parse_indices(response, max_index=n_items, k=k)
        return self._format_indices(indices, k)
