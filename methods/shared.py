#!/usr/bin/env python3
"""Shared utilities, prompts, and logging for methods."""

import re
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

from utils.llm import call_llm, call_llm_async
from utils.parsing import parse_final_answer, parse_script, substitute_variables

# =============================================================================
# Configuration and Global State
# =============================================================================

# Defense support
_defense = None
_use_defense_prompt = False


def set_defense_mode(enabled: bool):
    """Toggle between normal and defense prompts."""
    global _use_defense_prompt
    _use_defense_prompt = enabled


def set_defense(defense_concept: str):
    """Enable defense prompt."""
    global _defense
    _defense = defense_concept


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested."

TASK_CONCEPT = """You are evaluating whether a restaurant matches a user's SPECIFIC need.
The input contains restaurant info and multiple reviews with varying opinions.
The context describes ONE specific aspect the user cares about.

Output: 1 (good match), 0 (unclear/mixed), -1 (poor match)"""


# =============================================================================
# Utility Functions
# =============================================================================

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


def majority_vote(answers: list) -> int:
    """Return the most common answer, defaulting to 0 on tie."""
    if not answers:
        return 0
    counts = Counter(answers)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] == most_common[1][1]:
        return 0
    return most_common[0][0]
