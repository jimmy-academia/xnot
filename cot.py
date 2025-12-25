#!/usr/bin/env python3
"""
Chain-of-Thought (CoT) Method for Restaurant Recommendation

This module provides an LLM-based method for evaluating restaurant recommendations
using chain-of-thought prompting with few-shot examples.

Compatible with evaluate_llm.py - provides:
    method(query: str, context: str) -> int

Supports multiple LLM backends:
    - Anthropic Claude (default)
    - OpenAI GPT models

Usage as standalone:
    python cot_method.py --query "Restaurant info..." --context "User request..."

Usage with evaluation script:
    from cot_method import method
    # or
    from cot_method import create_method
    method = create_method(provider="anthropic", model="claude-sonnet-4-20250514")
"""

import os
import re
import json
import argparse
from typing import Optional, Callable


# =============================================================================
# FEW-SHOT EXAMPLES FOR CHAIN-OF-THOUGHT
# =============================================================================
# These examples demonstrate the reasoning process for different scenarios.
# Each example shows: restaurant info, user request, step-by-step reasoning, answer.

FEW_SHOT_EXAMPLES = [
    # Example 1: Clear RECOMMEND case
    {
        "query": """Restaurant:
Name: Serene Garden Bistro
City: Chicago
Neighborhood: Lincoln Park
Price: $$
Cuisine: American, Farm-to-Table

Reviews:
[rest_ex1_r1] This place is wonderfully quiet - perfect for our anniversary dinner. The booths are spacious and comfortable, and we could actually hear each other talk. Prices are reasonable for the quality.
[rest_ex1_r2] Great atmosphere for a date night. Not too loud, comfortable seating, and the bill didn't break the bank. Highly recommend for intimate conversations.
[rest_ex1_r3] Peaceful dining experience. The cushioned chairs are a nice touch, and the noise level is kept low. Good value for money.""",
        "context": "I'm looking for a quiet restaurant with comfortable seating that won't break the bank. I want a peaceful dining experience where I can have a conversation without shouting.",
        "reasoning": """Let me analyze this restaurant against the user's requirements:

1. QUIET ATMOSPHERE: Multiple reviews explicitly mention this is a quiet place. Review 1 says "wonderfully quiet" and they "could actually hear each other talk." Review 2 confirms "Not too loud." Review 3 calls it a "Peaceful dining experience" with "noise level kept low." This strongly satisfies the quiet requirement.

2. COMFORTABLE SEATING: Review 1 mentions "booths are spacious and comfortable." Review 3 specifically notes "cushioned chairs are a nice touch." This satisfies the comfortable seating requirement.

3. AFFORDABLE/NOT EXPENSIVE: The price range is $$ (moderate). Review 1 says "Prices are reasonable for the quality." Review 2 says "the bill didn't break the bank." Review 3 mentions "Good value for money." This satisfies the budget requirement.

All three key requirements (quiet, comfortable seating, affordable) are positively addressed by multiple reviews. There are no negative signals about any of these aspects.""",
        "answer": 1
    },
    # Example 2: Clear NOT RECOMMEND case
    {
        "query": """Restaurant:
Name: The Rooftop Grill
City: Chicago
Neighborhood: River North
Price: $$$
Cuisine: Steakhouse, American

Reviews:
[rest_ex2_r1] Great views but SO LOUD. The open concept means noise bounces everywhere. Had to shout to be heard. Food was good though.
[rest_ex2_r2] Trendy spot with a party vibe. Music pumping, lots of energy. The bar scene spills into dining. Not cheap but the steaks are solid.
[rest_ex2_r3] Fun atmosphere if you want excitement. The benches are a bit hard for long dinners. Expect to pay premium prices for the location.""",
        "context": "I'm looking for a quiet restaurant with comfortable seating that won't break the bank. I want a peaceful dining experience where I can have a conversation without shouting.",
        "reasoning": """Let me analyze this restaurant against the user's requirements:

1. QUIET ATMOSPHERE: This is clearly NOT quiet. Review 1 explicitly says "SO LOUD" and they "Had to shout to be heard." Review 2 describes it as having a "party vibe" with "Music pumping, lots of energy." This directly contradicts the user's need for a peaceful dining experience.

2. COMFORTABLE SEATING: Review 3 mentions "The benches are a bit hard for long dinners." This suggests seating is not particularly comfortable.

3. AFFORDABLE/NOT EXPENSIVE: The price range is $$$ (expensive). Review 2 says "Not cheap" and Review 3 mentions "premium prices." This does not satisfy the budget-conscious requirement.

The restaurant fails on all three key requirements: it's loud (not quiet), seating is uncomfortable, and it's expensive. This is a clear mismatch for this user's needs.""",
        "answer": -1
    },
    # Example 3: NEUTRAL/UNCERTAIN case
    {
        "query": """Restaurant:
Name: Corner Kitchen
City: Chicago
Neighborhood: Wicker Park
Price: $$
Cuisine: American, Comfort Food

Reviews:
[rest_ex3_r1] Decent neighborhood spot. Food is reliable comfort fare. Gets busy on weekends but manageable on weekdays.
[rest_ex3_r2] Good portions for the price. The diner-style booths are okay. Can get a bit noisy during rush but usually fine.
[rest_ex3_r3] Solid local option. Nothing fancy but does the job. Service varies depending on the day.""",
        "context": "I'm looking for a quiet restaurant with comfortable seating that won't break the bank. I want a peaceful dining experience where I can have a conversation without shouting.",
        "reasoning": """Let me analyze this restaurant against the user's requirements:

1. QUIET ATMOSPHERE: The evidence is mixed. Review 1 says it "Gets busy on weekends but manageable on weekdays." Review 2 mentions it "Can get a bit noisy during rush but usually fine." There's no strong indication it's particularly quiet or particularly loud - it seems to depend on timing.

2. COMFORTABLE SEATING: Review 2 describes "diner-style booths are okay" - this is neutral, neither particularly comfortable nor uncomfortable.

3. AFFORDABLE/NOT EXPENSIVE: The price range is $$ (moderate). Review 2 mentions "Good portions for the price." This reasonably satisfies the budget requirement.

The restaurant partially meets the requirements: pricing is acceptable, but noise level and seating comfort are inconsistent or merely adequate. The reviews don't provide strong evidence either way - it could work depending on when you visit, but it's not a confident recommendation.""",
        "answer": 0
    },
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert restaurant recommendation assistant. Your task is to evaluate whether a restaurant is a good match for a user's specific needs based on the restaurant information and reviews provided.

You must reason step-by-step through the evidence before making a decision. Consider:
1. What specific requirements does the user have?
2. What evidence in the reviews supports or contradicts each requirement?
3. How strong is the evidence overall?

After your analysis, you must provide a final recommendation:
- Output 1 if you RECOMMEND this restaurant (clear positive match for user's needs)
- Output 0 if you are NEUTRAL/UNCERTAIN (mixed evidence or insufficient information)
- Output -1 if you DO NOT RECOMMEND (clear mismatch for user's needs)

IMPORTANT: Your final answer must be on its own line in the format:
ANSWER: [number]

where [number] is -1, 0, or 1."""


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_few_shot_prompt(query: str, context: str) -> str:
    """
    Build the full prompt with few-shot examples and the current query.
    
    Args:
        query: Restaurant information (name, location, reviews)
        context: User's request/requirements
        
    Returns:
        Complete prompt string with few-shot examples
    """
    prompt_parts = []
    
    # Add few-shot examples
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        prompt_parts.append(f"=== Example {i} ===")
        prompt_parts.append(f"\n[RESTAURANT INFO]\n{example['query']}")
        prompt_parts.append(f"\n[USER REQUEST]\n{example['context']}")
        prompt_parts.append(f"\n[ANALYSIS]\n{example['reasoning']}")
        prompt_parts.append(f"\nANSWER: {example['answer']}")
        prompt_parts.append("\n")
    
    # Add the actual query
    prompt_parts.append("=== Your Task ===")
    prompt_parts.append(f"\n[RESTAURANT INFO]\n{query}")
    prompt_parts.append(f"\n[USER REQUEST]\n{context}")
    prompt_parts.append("\n[ANALYSIS]")
    
    return "\n".join(prompt_parts)


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def parse_response(response_text: str) -> int:
    """
    Parse the LLM response to extract the final answer.
    
    Looks for patterns like:
    - "ANSWER: 1"
    - "ANSWER: -1"
    - "Final answer: 0"
    - Just "-1", "0", or "1" on a line by itself
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        int: Parsed answer (-1, 0, or 1)
        
    Raises:
        ValueError: If no valid answer can be parsed
    """
    # Normalize text
    text = response_text.strip()
    
    # Pattern 1: Look for "ANSWER: X" format (most reliable)
    answer_patterns = [
        r'ANSWER:\s*(-?[01])',
        r'Answer:\s*(-?[01])',
        r'FINAL ANSWER:\s*(-?[01])',
        r'Final Answer:\s*(-?[01])',
        r'final answer:\s*(-?[01])',
        r'RECOMMENDATION:\s*(-?[01])',
        r'Recommendation:\s*(-?[01])',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Pattern 2: Look for explicit statements in last few lines
    last_lines = '\n'.join(text.split('\n')[-3:]).lower()
    if 'not recommend' in last_lines or 'do not recommend' in last_lines:
        return -1
    if 'recommend' in last_lines and 'not' not in last_lines:
        return 1
    
    # Pattern 3: Look for a standalone number at the end
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip()
        if line in ['-1', '0', '1']:
            return int(line)
        # Check for number at end of line
        match = re.search(r':\s*(-?[01])\s*$', line)
        if match:
            return int(match.group(1))
    
    # Pattern 4: Look for bracketed answers
    bracket_match = re.search(r'\[(-?[01])\]', text)
    if bracket_match:
        return int(bracket_match.group(1))
    
    # Pattern 5: Last resort - look for any -1, 0, 1 in the last line
    last_line = lines[-1].strip()
    for val in ['-1', '1', '0']:  # Check -1 before 1
        if val in last_line:
            return int(val)
    
    raise ValueError(f"Could not parse answer from response: {text[-200:]}")


# =============================================================================
# LLM BACKENDS
# =============================================================================

def call_anthropic(
    prompt: str,
    system: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    temperature: float = 0.0
) -> str:
    """
    Call Anthropic Claude API.
    
    Args:
        prompt: User prompt
        system: System prompt
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        str: Model response text
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )
    
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text


def call_openai(
    prompt: str,
    system: str,
    model: str = "gpt-4o",
    max_tokens: int = 1024,
    temperature: float = 0.0
) -> str:
    """
    Call OpenAI API.
    
    Args:
        prompt: User prompt
        system: System prompt
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        str: Model response text
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package not installed. "
            "Install with: pip install openai"
        )
    
    client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content


# =============================================================================
# METHOD FACTORY
# =============================================================================

def create_method(
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    verbose: bool = False
) -> Callable[[str, str], int]:
    """
    Create a method function compatible with evaluate_llm.py.
    
    Args:
        provider: LLM provider ("anthropic" or "openai")
        model: Model name (defaults based on provider)
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens in response
        verbose: Whether to print debug information
        
    Returns:
        Callable: method(query, context) -> int
    """
    # Set default models
    if model is None:
        if provider == "anthropic":
            model = "claude-sonnet-4-20250514"
        elif provider == "openai":
            model = "gpt-4o"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    # Select API caller
    if provider == "anthropic":
        api_call = call_anthropic
    elif provider == "openai":
        api_call = call_openai
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    def method(query: str, context: str) -> int:
        """
        Evaluate a restaurant for a user request using chain-of-thought.
        
        Args:
            query: Restaurant information (name, location, reviews)
            context: User's request/requirements
            
        Returns:
            int: -1 (not recommend), 0 (neutral), 1 (recommend)
        """
        # Build the prompt with few-shot examples
        prompt = build_few_shot_prompt(query, context)
        
        if verbose:
            print(f"\n{'='*60}")
            print("PROMPT (truncated):")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print('='*60)
        
        # Call the LLM
        response = api_call(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if verbose:
            print("\nRESPONSE:")
            print(response)
            print('='*60)
        
        # Parse the response
        answer = parse_response(response)
        
        if verbose:
            print(f"\nPARSED ANSWER: {answer}")
        
        return answer
    
    return method


# =============================================================================
# DEFAULT METHOD INSTANCE
# =============================================================================
# This is the default method that can be imported directly.
# It will be created lazily on first use to allow for environment setup.

_default_method = None

def method(query: str, context: str) -> int:
    """
    Default method instance using Anthropic Claude.
    
    This is the main entry point for use with evaluate_llm.py:
        from cot_method import method
    
    Args:
        query: Restaurant information (name, location, reviews)
        context: User's request/requirements
        
    Returns:
        int: -1 (not recommend), 0 (neutral), 1 (recommend)
    """
    global _default_method
    
    if _default_method is None:
        # Determine provider based on available API keys
        if os.environ.get("ANTHROPIC_API_KEY"):
            _default_method = create_method(provider="anthropic")
        elif os.environ.get("OPENAI_API_KEY"):
            _default_method = create_method(provider="openai")
        else:
            raise EnvironmentError(
                "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "environment variable."
            )
    
    return _default_method(query, context)


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def main():
    """Command-line interface for testing the method."""
    parser = argparse.ArgumentParser(
        description="Test the CoT recommendation method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a query and context
  python cot_method.py --query "Restaurant: Test..." --context "I need..."
  
  # Use OpenAI instead of Anthropic
  python cot_method.py --provider openai --query "..." --context "..."
  
  # Run built-in test
  python cot_method.py --test
  
  # Verbose mode to see full prompt and response
  python cot_method.py --test --verbose
        """
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Restaurant query string"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="User request context"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: auto-select based on provider)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full prompt and response"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a built-in test case"
    )
    
    args = parser.parse_args()
    
    # Create method with specified configuration
    test_method = create_method(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    if args.test:
        # Built-in test case
        test_query = """Restaurant:
Name: Mama Rosa's Italian Kitchen
City: Chicago
Neighborhood: Little Italy
Price: $$
Cuisine: Italian, Pizza

Reviews:
[rest_test_r1] Authentic Italian food just like my grandmother used to make. The homemade pasta is incredible. A bit noisy on weekends but the food makes up for it.
[rest_test_r2] Classic Chicago Italian spot. They've been here for 40 years for good reason. Cash only, no frills, just amazing food. Tourists and locals alike love it.
[rest_test_r3] The deep dish here is different from the chain places - more authentic. Staff treats everyone like family. Prices are very fair for the portions."""
        
        test_context = "I'm visiting Chicago and want an authentic local experience with classic Chicago dishes. I'd like somewhere that's tourist-friendly but still has that genuine local vibe."
        
        print("Running built-in test case...")
        print(f"\nProvider: {args.provider}")
        print(f"Model: {args.model or 'default'}")
        print(f"\nContext: {test_context}")
        print(f"\nQuery (truncated): {test_query[:200]}...")
        
        result = test_method(test_query, test_context)
        
        print(f"\n{'='*40}")
        print(f"RESULT: {result}")
        print(f"{'='*40}")
        
        interpretation = {1: "RECOMMEND", 0: "NEUTRAL", -1: "NOT RECOMMEND"}
        print(f"Interpretation: {interpretation.get(result, 'UNKNOWN')}")
        
    elif args.query and args.context:
        # User-provided query and context
        result = test_method(args.query, args.context)
        print(f"\nResult: {result}")
        
    else:
        parser.print_help()
        print("\nError: Either --test or both --query and --context are required.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())