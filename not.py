#!/usr/bin/env python3
"""
Network of Thought (NoT) Method for Restaurant Recommendation

This module implements a Network of Thought approach where:
1. A planner LLM generates a step-by-step script to solve the problem
2. Each step is executed sequentially with variable substitution
3. Steps can reference previous outputs using {(index)} notation

Compatible with evaluate_llm.py - use the `method` function directly.

Usage:
    from not_method import method
    result = method(query, context)

Configuration via environment variables:
    - LLM_PROVIDER: "openai" | "anthropic" | "local" (default: "openai")
    - LLM_MODEL: Model name for worker (default: "gpt-4o-mini")
    - LLM_PLANNER_MODEL: Model name for planner (default: same as LLM_MODEL)
    - LLM_TEMPERATURE: Sampling temperature (default: 0.0)
    - LLM_MAX_TOKENS: Max tokens for response (default: 1024)
    - NOT_DEBUG: Set to "1" to enable debug output
"""

import os
import re
import json
import ast
from typing import Dict, Any, Optional, Tuple, List
from functools import partial

# =============================================================================
# CONFIGURATION
# =============================================================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_PLANNER_MODEL = os.environ.get("LLM_PLANNER_MODEL", LLM_MODEL)
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
DEBUG = os.environ.get("NOT_DEBUG", "0") == "1"


# =============================================================================
# TASK-SPECIFIC PROMPTS FOR RESTAURANT RECOMMENDATION
# =============================================================================

TASK_CONCEPT = """You are evaluating whether a restaurant should be recommended to a user based on their specific request.
Analyze the restaurant information (name, location, price, cuisine) and all customer reviews.
Compare the evidence in the reviews against what the user is looking for.
Output a final recommendation: 1 (recommend), 0 (neutral/uncertain), or -1 (not recommend).
Break down the analysis into clear steps: extract key requirements, check each requirement against reviews, then synthesize."""

TASK_EXAMPLE = """example for restaurant recommendation
(0)=LLM("Extract the user's key requirements from the context as a numbered list. Context: {(context)}")
(1)=LLM("Summarize each review from the restaurant, noting positive and negative aspects. Reviews: {(input)}")
(2)=LLM("For requirement 1 from {(0)}, check if the reviews in {(1)} provide evidence for or against it. Output: supported, not supported, or mixed.")
(3)=LLM("For requirement 2 from {(0)}, check if the reviews in {(1)} provide evidence for or against it. Output: supported, not supported, or mixed.")
(4)=LLM("For requirement 3 from {(0)}, check if the reviews in {(1)} provide evidence for or against it. Output: supported, not supported, or mixed.")
(5)=LLM("Based on the evidence analysis: Req1={(2)}, Req2={(3)}, Req3={(4)}. If most requirements are supported, output 1. If most are not supported, output -1. If mixed or unclear, output 0. Output only the number.")
"""

KNOWLEDGE_PROMPT = """Given the following task:
%s

The Input section contains restaurant information and reviews.
The Context section describes what the user is looking for.

Please create a solution in a step-by-step manner.
Each step should be simple and focused on one sub-task.
Don't use loops or patterns to reduce steps.
Use Step0, Step1, Step2 to represent intermediate results.

Key steps should include:
1. Extract user requirements from the context
2. Summarize relevant information from reviews
3. Check each major requirement against the evidence
4. Synthesize findings into a final recommendation (1, 0, or -1)
"""

SCRIPT_PROMPT = """You must create a script to solve a restaurant recommendation task.
The script is numbered and contains LLM calls executed sequentially.
Use (index) to represent each line, starting from 0.

Use LLM("Your Instruction") for each step.
Use {{(index)}} to reference previous results (e.g., {{(0)}}, {{(1)}}).
Use {{(input)}} for restaurant info and {{(context)}} for user request.
Use Python indexing for list elements: {{(0)}}[0], {{(0)}}[1].

Here is an example:
%s

Based on this knowledge:
%s

Create a script to solve:
%s

Requirements:
- The FINAL step must output exactly one of: -1, 0, or 1
- Do not include any text after the script
- Each step should be on its own line in format: (N)=LLM("instruction")
"""

SYSTEM_PROMPT = "You follow instructions precisely. Output only what is requested, no additional explanation."


# =============================================================================
# VARIABLE SUBSTITUTION
# =============================================================================

def substitute_variables(
    instruction: str,
    query: str,
    context: str,
    cache: Dict[str, Any]
) -> str:
    """
    Substitute variables in instruction with actual values.
    
    Handles:
    - {(input)} -> query (restaurant info)
    - {(context)} -> context (user request)
    - {(N)} -> cached result from step N
    - {(N)}[i] -> indexed access into list result
    
    Args:
        instruction: The instruction template
        query: Restaurant information
        context: User request
        cache: Dictionary of previous step results
        
    Returns:
        Instruction with all variables substituted
    """
    def _sub(match):
        var_name = match.group(1)
        index_str = match.group(2)  # None if no index specified
        
        # Determine base value
        if var_name == 'input':
            base_value = query
        elif var_name == 'context':
            base_value = context
        else:
            base_value = cache.get(var_name, '')
        
        # Try to parse as literal if it looks like a list
        if isinstance(base_value, str):
            try:
                parsed = ast.literal_eval(base_value)
                if isinstance(parsed, (list, tuple)):
                    base_value = parsed
            except (SyntaxError, ValueError):
                pass
        
        # Apply indexing if specified
        if index_str is not None:
            index = int(index_str)
            if isinstance(base_value, (list, tuple)):
                if 0 <= index < len(base_value):
                    return str(base_value[index])
                else:
                    return ''
            elif isinstance(base_value, str):
                # Try to parse and index
                try:
                    parsed = ast.literal_eval(base_value)
                    if isinstance(parsed, (list, tuple)) and 0 <= index < len(parsed):
                        return str(parsed[index])
                except:
                    pass
        
        # Return as string
        if isinstance(base_value, (list, tuple)):
            return json.dumps(base_value)
        return str(base_value)
    
    # Pattern matches {(var)} or {(var)}[index]
    pattern = r'\{\((\w+)\)\}(?:\[(\d+)\])?'
    return re.sub(pattern, _sub, instruction)


# =============================================================================
# SCRIPT PARSING
# =============================================================================

def parse_script(script_text: str) -> List[Tuple[str, str]]:
    """
    Parse the generated script into executable steps.
    
    Args:
        script_text: Raw script from planner LLM
        
    Returns:
        List of (index, instruction) tuples
    """
    steps = []
    
    for line in script_text.split('\n'):
        line = line.strip()
        if not line or '=LLM(' not in line:
            continue
        
        # Extract step index
        index_match = re.search(r'\((\d+)\)\s*=\s*LLM', line)
        if not index_match:
            continue
        index = index_match.group(1)
        
        # Extract instruction (handle both " and ' quotes)
        instr_match = re.search(r'LLM\(["\'](.+?)["\']\)', line, re.DOTALL)
        if not instr_match:
            # Try multiline or escaped
            instr_match = re.search(r'LLM\("(.+?)"\)', line, re.DOTALL)
        if not instr_match:
            instr_match = re.search(r"LLM\('(.+?)'\)", line, re.DOTALL)
        
        if instr_match:
            instruction = instr_match.group(1)
            steps.append((index, instruction))
    
    return steps


# =============================================================================
# LLM API CALLS
# =============================================================================

def call_openai(messages: list, model: str) -> str:
    """Call OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    base_url = os.environ.get("LLM_BASE_URL")
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = openai.OpenAI(**client_kwargs)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    
    return response.choices[0].message.content


def call_anthropic(messages: list, model: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    client = anthropic.Anthropic()
    
    # Extract system message
    system_content = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            chat_messages.append(msg)
    
    # Use appropriate model name
    if "claude" not in model.lower():
        model = "claude-sonnet-4-20250514"
    
    response = client.messages.create(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        system=system_content,
        messages=chat_messages,
    )
    
    return response.content[0].text


def call_local(messages: list, model: str) -> str:
    """Call local/custom LLM endpoint."""
    import urllib.request
    
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise ValueError("LLM_BASE_URL must be set for local provider")
    
    url = base_url.rstrip("/") + "/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    
    return result["choices"][0]["message"]["content"]


def call_llm(prompt: str, is_planner: bool = False) -> str:
    """
    Call LLM with the given prompt.
    
    Args:
        prompt: User prompt
        is_planner: If True, use planner model; otherwise use worker model
        
    Returns:
        LLM response text
    """
    model = LLM_PLANNER_MODEL if is_planner else LLM_MODEL
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    if LLM_PROVIDER == "openai":
        return call_openai(messages, model)
    elif LLM_PROVIDER == "anthropic":
        return call_anthropic(messages, model)
    elif LLM_PROVIDER == "local":
        return call_local(messages, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_final_answer(output: str) -> int:
    """
    Parse the final step output to get recommendation.
    
    Args:
        output: Final step output
        
    Returns:
        -1, 0, or 1
    """
    output = output.strip()
    
    # Direct match
    if output in ["-1", "0", "1"]:
        return int(output)
    
    # Look for standalone number
    match = re.search(r'(?:^|[:\s])(-1|0|1)(?:\s|$|\.)', output)
    if match:
        return int(match.group(1))
    
    # Keyword matching
    lower = output.lower()
    if "not recommend" in lower or "do not recommend" in lower:
        return -1
    if "recommend" in lower and "not" not in lower:
        return 1
    if "neutral" in lower or "uncertain" in lower or "mixed" in lower:
        return 0
    
    # Default
    return 0


# =============================================================================
# NETWORK OF THOUGHT EXECUTOR
# =============================================================================

class NetworkOfThought:
    """
    Network of Thought executor for restaurant recommendation.
    
    Generates a dynamic script using a planner LLM, then executes
    each step sequentially with variable substitution.
    """
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.step_outputs: List[str] = []
    
    def generate_script(self, query: str, context: str) -> str:
        """
        Use planner LLM to generate execution script.
        
        Args:
            query: Restaurant information
            context: User request
            
        Returns:
            Generated script text
        """
        goal = f"Input: {query}\nContext: {context}"
        
        # Step 1: Generate knowledge/approach
        knowledge_prompt = KNOWLEDGE_PROMPT % goal
        knowledge = call_llm(knowledge_prompt, is_planner=True)
        
        if DEBUG:
            print("=" * 50)
            print("KNOWLEDGE:")
            print(knowledge)
            print("=" * 50)
        
        # Step 2: Generate script based on knowledge
        script_prompt = SCRIPT_PROMPT % (TASK_EXAMPLE, knowledge, goal)
        script = call_llm(script_prompt, is_planner=True)
        
        if DEBUG:
            print("SCRIPT:")
            print(script)
            print("=" * 50)
        
        return script
    
    def execute_script(
        self,
        script: str,
        query: str,
        context: str
    ) -> Tuple[str, List[str]]:
        """
        Execute the generated script step by step.
        
        Args:
            script: Generated script text
            query: Restaurant information
            context: User request
            
        Returns:
            Tuple of (final_output, list_of_all_outputs)
        """
        self.cache = {}
        self.step_outputs = []
        
        steps = parse_script(script)
        
        if not steps:
            # Fallback: direct answer
            fallback_prompt = f"""Based on this restaurant information:
{query}

And this user request:
{context}

Should this restaurant be recommended? Output only: -1 (no), 0 (uncertain), or 1 (yes)."""
            output = call_llm(fallback_prompt)
            return output, [output]
        
        final_output = ""
        
        for index, instruction in steps:
            # Substitute variables
            filled_instruction = substitute_variables(
                instruction, query, context, self.cache
            )
            
            if DEBUG:
                print(f"Step ({index}): {filled_instruction[:100]}...")
            
            # Execute step
            try:
                output = call_llm(filled_instruction)
            except Exception as e:
                output = f"Error: {e}"
            
            # Cache result
            try:
                self.cache[index] = ast.literal_eval(output)
            except:
                self.cache[index] = output
            
            self.step_outputs.append(output)
            final_output = output
            
            if DEBUG:
                print(f"  -> {output[:100]}...")
        
        return final_output, self.step_outputs
    
    def solve(self, query: str, context: str) -> int:
        """
        Solve the recommendation task.
        
        Args:
            query: Restaurant information
            context: User request
            
        Returns:
            Recommendation: -1, 0, or 1
        """
        # Generate script
        script = self.generate_script(query, context)
        
        # Execute script
        final_output, _ = self.execute_script(script, query, context)
        
        # Parse result
        return parse_final_answer(final_output)


# =============================================================================
# SIMPLIFIED NETWORK OF THOUGHT (FIXED SCRIPT)
# =============================================================================

FIXED_SCRIPT = """(0)=LLM("Extract 3-5 key requirements from the user's request. Be specific. Context: {(context)}")
(1)=LLM("Summarize each review, noting: atmosphere, service, price mentions, food quality, and any specific features. Restaurant info: {(input)}")
(2)=LLM("Based on the requirements in {(0)} and the review summaries in {(1)}, for each requirement state whether reviews provide: POSITIVE evidence, NEGATIVE evidence, or NO CLEAR evidence. Format as a list.")
(3)=LLM("Count the evidence from {(2)}: How many requirements have POSITIVE evidence? How many have NEGATIVE evidence? How many have NO CLEAR evidence? Output the counts.")
(4)=LLM("Based on {(3)}: If POSITIVE > NEGATIVE and POSITIVE >= 2, output 1. If NEGATIVE > POSITIVE and NEGATIVE >= 2, output -1. Otherwise output 0. Output ONLY the number: -1, 0, or 1")"""


class SimpleNetworkOfThought:
    """
    Simplified Network of Thought with a fixed script structure.
    More reliable than dynamic script generation.
    """
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def solve(self, query: str, context: str) -> int:
        """
        Solve using fixed script structure.
        
        Args:
            query: Restaurant information
            context: User request
            
        Returns:
            Recommendation: -1, 0, or 1
        """
        self.cache = {}
        steps = parse_script(FIXED_SCRIPT)
        
        final_output = ""
        
        for index, instruction in steps:
            filled = substitute_variables(instruction, query, context, self.cache)
            
            if DEBUG:
                print(f"Step ({index}): {filled[:80]}...")
            
            try:
                output = call_llm(filled)
            except Exception as e:
                output = f"0"  # Default on error
                if DEBUG:
                    print(f"  Error: {e}")
            
            try:
                self.cache[index] = ast.literal_eval(output)
            except:
                self.cache[index] = output
            
            final_output = output
            
            if DEBUG:
                print(f"  -> {output[:100]}...")
        
        return parse_final_answer(final_output)


# =============================================================================
# MAIN METHOD (COMPATIBLE WITH evaluate_llm.py)
# =============================================================================

# Global executor instance (reusable)
_executor = None


def method(query: str, context: str) -> int:
    """
    Network of Thought method for restaurant recommendation.
    
    This function is compatible with evaluate_llm.py.
    
    Uses a fixed-structure script for reliability. Set environment
    variable NOT_USE_DYNAMIC=1 to use dynamic script generation.
    
    Args:
        query: Formatted restaurant information (name, location, reviews).
               This should NOT contain any ground truth labels.
        context: User's request describing what they're looking for.
        
    Returns:
        int: Recommendation score
            -1 = Not recommended
             0 = Neutral/uncertain
             1 = Recommended
    """
    global _executor
    
    use_dynamic = os.environ.get("NOT_USE_DYNAMIC", "0") == "1"
    
    if use_dynamic:
        if _executor is None or not isinstance(_executor, NetworkOfThought):
            _executor = NetworkOfThought()
    else:
        if _executor is None or not isinstance(_executor, SimpleNetworkOfThought):
            _executor = SimpleNetworkOfThought()
    
    try:
        return _executor.solve(query, context)
    except Exception as e:
        if DEBUG:
            print(f"Error in NoT method: {e}")
        return 0  # Default to neutral on failure


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def test_variable_substitution():
    """Test variable substitution."""
    print("Testing variable substitution...")
    
    cache = {"0": ["item1", "item2", "item3"], "1": "some result"}
    
    # Test basic substitution
    result = substitute_variables(
        "Process {(input)} with {(context)}",
        "restaurant info",
        "user request",
        cache
    )
    assert "restaurant info" in result
    assert "user request" in result
    print("  ✓ Basic substitution")
    
    # Test cache reference
    result = substitute_variables(
        "Use {(1)} here",
        "", "", cache
    )
    assert "some result" in result
    print("  ✓ Cache reference")
    
    # Test list indexing
    result = substitute_variables(
        "Get {(0)}[1]",
        "", "", cache
    )
    assert "item2" in result
    print("  ✓ List indexing")
    
    print()


def test_script_parsing():
    """Test script parsing."""
    print("Testing script parsing...")
    
    script = """
(0)=LLM("First step with {(input)}")
(1)=LLM("Second step using {(0)}")
Some other text
(2)=LLM("Third step")
"""
    
    steps = parse_script(script)
    assert len(steps) == 3
    assert steps[0][0] == "0"
    assert steps[1][0] == "1"
    assert steps[2][0] == "2"
    print(f"  ✓ Parsed {len(steps)} steps")
    
    print()


def test_answer_parsing():
    """Test final answer parsing."""
    print("Testing answer parsing...")
    
    test_cases = [
        ("1", 1),
        ("-1", -1),
        ("0", 0),
        ("The answer is 1", 1),
        ("Output: -1", -1),
        ("I recommend this restaurant", 1),
        ("I would not recommend", -1),
        ("Mixed reviews, uncertain", 0),
    ]
    
    for text, expected in test_cases:
        result = parse_final_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:30]}...' -> {result} (expected {expected})")
    
    print()


def test_method_mock():
    """Test the method with mock LLM."""
    print("Testing method (mock mode)...")
    
    # Temporarily replace call_llm
    original_call = globals().get('call_llm')
    call_count = [0]
    
    def mock_call(prompt, is_planner=False):
        call_count[0] += 1
        if "requirements" in prompt.lower():
            return "1. Quiet atmosphere\n2. Affordable prices\n3. Good food"
        elif "summarize" in prompt.lower():
            return "Review 1: Great quiet spot. Review 2: Affordable and tasty."
        elif "evidence" in prompt.lower():
            return "1. Quiet: POSITIVE\n2. Affordable: POSITIVE\n3. Good food: POSITIVE"
        elif "count" in prompt.lower():
            return "POSITIVE: 3, NEGATIVE: 0, NO CLEAR: 0"
        else:
            return "1"
    
    globals()['call_llm'] = mock_call
    
    test_query = """Restaurant:
Name: Test Cafe
City: Chicago
Neighborhood: Loop
Price: $
Cuisine: Cafe

Reviews:
[r1] Quiet and affordable!"""
    
    test_context = "Looking for a quiet, affordable cafe."
    
    try:
        global _executor
        _executor = None  # Reset
        result = method(test_query, test_context)
        print(f"  Mock method returned: {result}")
        print(f"  LLM calls made: {call_count[0]}")
        print(f"  Status: {'✓' if result == 1 else '✗'}")
    finally:
        globals()['call_llm'] = original_call
        _executor = None
    
    print()


def main():
    """Run standalone tests."""
    print("=" * 60)
    print("Network of Thought Method - Test Suite")
    print("=" * 60)
    print()
    
    print(f"Configuration:")
    print(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"  LLM_MODEL (worker): {LLM_MODEL}")
    print(f"  LLM_PLANNER_MODEL: {LLM_PLANNER_MODEL}")
    print(f"  DEBUG: {DEBUG}")
    print()
    
    test_variable_substitution()
    test_script_parsing()
    test_answer_parsing()
    test_method_mock()
    
    print("=" * 60)
    print("To use with evaluate_llm.py:")
    print("  1. Set API key: export OPENAI_API_KEY=...")
    print("  2. Import: from not_method import method")
    print("  3. Replace dummy_method with method")
    print()
    print("Options:")
    print("  NOT_DEBUG=1        Enable debug output")
    print("  NOT_USE_DYNAMIC=1  Use dynamic script generation")
    print("=" * 60)


if __name__ == "__main__":
    main()