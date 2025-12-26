#!/usr/bin/env python3
"""Chain-of-Thought method for restaurant recommendation."""

import re
from llm import call_llm

FEW_SHOT_EXAMPLES = []  # No examples - pure zero-shot

SYSTEM_PROMPT = """Rate this restaurant. Output ANSWER: 1, 0, or -1."""

# Defense support
_defense = None

def set_defense(defense_concept: str):
    """Enable defense prompt."""
    global _defense
    _defense = defense_concept


def build_prompt(query: str, context: str) -> str:
    """Build prompt with few-shot examples."""
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"=== Example {i} ===")
        parts.append(f"\n[RESTAURANT INFO]\n{ex['query']}")
        parts.append(f"\n[USER REQUEST]\n{ex['context']}")
        parts.append(f"\n[ANALYSIS]\n{ex['reasoning']}")
        parts.append(f"\nANSWER: {ex['answer']}\n")
    parts.append("=== Your Task ===")
    parts.append(f"\n[RESTAURANT INFO]\n{query}")
    parts.append(f"\n[USER REQUEST]\n{context}")
    parts.append("\n[ANALYSIS]")
    return "\n".join(parts)


def parse_response(text: str) -> int:
    """Extract answer (-1, 0, 1) from LLM response."""
    # Pattern 1: ANSWER: X format
    match = re.search(r'(?:ANSWER|Answer|FINAL ANSWER|Final Answer):\s*(-?[01])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Pattern 2: Standalone number in last lines
    for line in reversed(text.strip().split('\n')[-5:]):
        line = line.strip()
        if line in ['-1', '0', '1']:
            return int(line)
        match = re.search(r':\s*(-?[01])\s*$', line)
        if match:
            return int(match.group(1))

    # Pattern 3: Keywords in last lines
    last = '\n'.join(text.split('\n')[-3:]).lower()
    if 'not recommend' in last:
        return -1
    if 'recommend' in last and 'not' not in last:
        return 1

    raise ValueError(f"Could not parse answer from: {text[-200:]}")


def method(query: str, context: str) -> int:
    """Evaluate restaurant recommendation. Returns -1, 0, or 1."""
    prompt = build_prompt(query, context)
    system = SYSTEM_PROMPT
    if _defense:
        system = _defense + "\n\n" + SYSTEM_PROMPT
    response = call_llm(prompt, system=system)
    return parse_response(response)
