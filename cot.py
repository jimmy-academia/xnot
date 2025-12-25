#!/usr/bin/env python3
"""Chain-of-Thought method for restaurant recommendation."""

import re
from llm import call_llm

FEW_SHOT_EXAMPLES = [
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
    response = call_llm(prompt, system=SYSTEM_PROMPT)
    return parse_response(response)
