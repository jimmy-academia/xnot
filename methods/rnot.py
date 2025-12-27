#!/usr/bin/env python3
"""Network-of-Thought method for restaurant recommendation."""

import os
import re
import json
import ast

from llm import call_llm

DEBUG = os.environ.get("NOT_DEBUG", "0") == "1"
SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested, no additional explanation."

FIXED_SCRIPT = """(0)=LLM("Extract 3-5 key requirements from the user's request. Be specific. Context: {(context)}")
(1)=LLM("Summarize each review, noting: atmosphere, service, price mentions, food quality, and any specific features. Restaurant info: {(input)}")
(2)=LLM("Based on the requirements in {(0)} and the review summaries in {(1)}, for each requirement state whether reviews provide: POSITIVE evidence, NEGATIVE evidence, or NO CLEAR evidence. Format as a list.")
(3)=LLM("Count the evidence from {(2)}: How many requirements have POSITIVE evidence? How many have NEGATIVE evidence? How many have NO CLEAR evidence? Output the counts.")
(4)=LLM("Based on {(3)}: If POSITIVE > NEGATIVE and POSITIVE >= 2, output 1. If NEGATIVE > POSITIVE and NEGATIVE >= 2, output -1. Otherwise output 0. Output ONLY the number: -1, 0, or 1")"""


def substitute_variables(instruction: str, query: str, context: str, cache: dict) -> str:
    """Substitute {(input)}, {(context)}, {(N)}, {(N)}[i] in instruction."""
    def _sub(match):
        var = match.group(1)
        idx = match.group(2)
        if var == 'input':
            val = query
        elif var == 'context':
            val = context
        else:
            val = cache.get(var, '')

        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)):
                    val = parsed
            except:
                pass

        if idx is not None:
            i = int(idx)
            if isinstance(val, (list, tuple)) and 0 <= i < len(val):
                return str(val[i])
            return ''

        return json.dumps(val) if isinstance(val, (list, tuple)) else str(val)

    return re.sub(r'\{\((\w+)\)\}(?:\[(\d+)\])?', _sub, instruction)


def parse_script(script: str) -> list:
    """Parse script into [(index, instruction), ...]."""
    steps = []
    for line in script.split('\n'):
        if '=LLM(' not in line:
            continue
        idx_match = re.search(r'\((\d+)\)\s*=\s*LLM', line)
        instr_match = re.search(r'LLM\(["\'](.+?)["\']\)', line, re.DOTALL)
        if idx_match and instr_match:
            steps.append((idx_match.group(1), instr_match.group(1)))
    return steps


def parse_final_answer(output: str) -> int:
    """Parse output to -1, 0, or 1."""
    output = output.strip()
    if output in ["-1", "0", "1"]:
        return int(output)

    match = re.search(r'(?:^|[:\s])(-1|0|1)(?:\s|$|\.)', output)
    if match:
        return int(match.group(1))

    lower = output.lower()
    if "not recommend" in lower:
        return -1
    if "recommend" in lower and "not" not in lower:
        return 1
    return 0


class SimpleNetworkOfThought:
    """Fixed-script Network of Thought executor."""

    def __init__(self):
        self.cache = {}

    def solve(self, query: str, context: str) -> int:
        self.cache = {}
        steps = parse_script(FIXED_SCRIPT)
        final = ""

        for idx, instr in steps:
            filled = substitute_variables(instr, query, context, self.cache)
            if DEBUG:
                print(f"Step ({idx}): {filled[:80]}...")

            try:
                output = call_llm(filled, system=SYSTEM_PROMPT)
            except Exception as e:
                output = "0"
                if DEBUG:
                    print(f"  Error: {e}")

            try:
                self.cache[idx] = ast.literal_eval(output)
            except:
                self.cache[idx] = output

            final = output
            if DEBUG:
                print(f"  -> {output[:100]}...")

        return parse_final_answer(final)


_executor = None

def method(query: str, context: str) -> int:
    """Network-of-Thought evaluation. Returns -1, 0, or 1."""
    global _executor
    if _executor is None:
        _executor = SimpleNetworkOfThought()
    try:
        return _executor.solve(query, context)
    except Exception as e:
        if DEBUG:
            print(f"Error: {e}")
        return 0
