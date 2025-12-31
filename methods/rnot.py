#!/usr/bin/env python3
"""Network-of-Thought method for restaurant recommendation."""

import os
import ast

from utils.llm import call_llm
from utils.parsing import parse_final_answer, parse_script, substitute_variables

DEBUG = os.environ.get("NOT_DEBUG", "0") == "1"
SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested, no additional explanation."

FIXED_SCRIPT = """(0)=LLM("Extract 3-5 key requirements from the user's request. Be specific. Context: {(context)}")
(1)=LLM("Summarize each review, noting: atmosphere, service, price mentions, food quality, and any specific features. Restaurant info: {(input)}")
(2)=LLM("Based on the requirements in {(0)} and the review summaries in {(1)}, for each requirement state whether reviews provide: POSITIVE evidence, NEGATIVE evidence, or NO CLEAR evidence. Format as a list.")
(3)=LLM("Count the evidence from {(2)}: How many requirements have POSITIVE evidence? How many have NEGATIVE evidence? How many have NO CLEAR evidence? Output the counts.")
(4)=LLM("Based on {(3)}: If POSITIVE > NEGATIVE and POSITIVE >= 2, output 1. If NEGATIVE > POSITIVE and NEGATIVE >= 2, output -1. Otherwise output 0. Output ONLY the number: -1, 0, or 1")"""

# Note: parse_script, substitute_variables, parse_final_answer imported from utils.parsing


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
