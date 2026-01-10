#!/usr/bin/env python3
"""
LLM-as-Judge Ground Truth Generator for G1 (Allergy Safety)

Generates ground truth using LLM analysis instead of keyword matching.
Results are cached to avoid repeated calls.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


CACHE_PATH = Path(__file__).parent / "data" / "g1_judgment_cache.json"


def load_cache() -> Dict[str, Any]:
    """Load existing judgment cache"""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]):
    """Save judgment cache"""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def generate_cache_key(restaurant: Dict, reviews: List[Dict]) -> str:
    """Generate unique cache key for restaurant + review set"""
    # Use restaurant name + number of reviews as key
    name = restaurant.get('name', 'Unknown').replace(' ', '_')
    return f"{name}_N{len(reviews)}"


JUDGE_PROMPT_TEMPLATE = """You are an expert food safety analyst specializing in severe peanut allergy risk assessment.

# TASK
Analyze the following restaurant reviews to assess peanut allergy safety risk.

# RESTAURANT
Name: {restaurant_name}
Categories: {categories}

# REVIEWS ({num_reviews} total)
{reviews_text}

# ANALYSIS INSTRUCTIONS

Evaluate the restaurant on these specific criteria:

**1. SEVERE_INCIDENTS**: Count reviews mentioning severe allergy reactions (e.g., "reaction", "hospital", "allergic shock", "nuts everywhere")

**2. MILD_CONCERNS**: Count reviews mentioning cross-contamination risks or mild issues (e.g., "peanut oil", "cross contamination", "got sick", "traces")

**3. SAFETY_MEASURES**: Count reviews mentioning positive allergy safety (e.g., "accommodated", "dedicated fryer", "checked with chef", "safe for allergies", "allergy menu")

**4. HYGIENE_FLAGS**: Note any cleanliness issues that could affect safety

**5. OVERALL_RISK_SCORE**: Calculate 0-20 scale:
   - 0-4: Low Risk (no/minimal allergy mentions, good safety practices)
   - 4-8: High Risk (some concerns, limited safety measures)
   - 8+: Critical Risk (severe incidents, poor safety practices)

# OUTPUT FORMAT (JSON)
{{
    "severe_incidents": <integer>,
    "mild_concerns": <integer>,
    "safety_measures": <integer>,
    "hygiene_good": <integer>,
    "hygiene_bad": <integer>,
    "final_risk_score": <float 0-20>,
    "verdict": "<Low Risk|High Risk|Critical Risk>",
    "reasoning": "<1-2 sentence justification>"
}}

Be CONSERVATIVE - when uncertain, err on side of higher risk for allergy safety.
"""


def judge_restaurant_llm(restaurant: Dict, reviews: List[Dict]) -> Dict[str, Any]:
    """
    Use LLM to analyze restaurant and generate ground truth judgment.
    
    Returns structured judgment with risk scores and verdict.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.llm import call_llm
    
    # Format reviews for prompt
    reviews_text = "\n\n".join([
        f"Review {i+1} ({r.get('stars', '?')} stars, {r.get('date', 'unknown date')}):\n{r.get('text', '')[:500]}"
        for i, r in enumerate(reviews[:50])  # Limit to 50 reviews to fit context
    ])
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        restaurant_name=restaurant.get('name', 'Unknown'),
        categories=restaurant.get('categories', 'Unknown'),
        num_reviews=len(reviews),
        reviews_text=reviews_text
    )
    
    # Call LLM
    response = call_llm(prompt)
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        judgment = json.loads(json_str)
        return judgment
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Failed to parse LLM response: {e}")
        print(f"Response: {response[:200]}")
        # Return safe defaults
        return {
            "severe_incidents": 0,
            "mild_concerns": 0,
            "safety_measures": 0,
            "hygiene_good": 0,
            "hygiene_bad": 0,
            "final_risk_score": 5.0,
            "verdict": "High Risk",
            "reasoning": "Failed to parse LLM judgment"
        }


def get_or_generate_judgment(restaurant: Dict, reviews: List[Dict], use_cache: bool = True) -> Dict[str, Any]:
    """
    Get cached judgment or generate new one via LLM.
    """
    cache_key = generate_cache_key(restaurant, reviews)
    
    # Try cache first
    if use_cache:
        cache = load_cache()
        if cache_key in cache:
            print(f"✓ Cache hit: {cache_key}")
            return cache[cache_key]
    
    # Generate new judgment
    print(f"→ Generating judgment: {cache_key}")
    judgment = judge_restaurant_llm(restaurant, reviews)
    
    # Cache it
    cache = load_cache()
    cache[cache_key] = judgment
    save_cache(cache)
    
    return judgment


if __name__ == '__main__':
    # Test on a few restaurants
    import sys
    sys.path.insert(0, 'explore')
    
    files = list(open('data/packed/dataset_N50.jsonl'))[:5]
    
    print("Testing LLM Judge on 5 restaurants...")
    print("="*70)
    
    for line in files:
        data = json.loads(line)
        judgment = get_or_generate_judgment(data, data.get('reviews', []))
        
        print(f"\nRestaurant: {data.get('name', 'Unknown')}")
        print(f"  Risk Score: {judgment['final_risk_score']:.1f}")
        print(f"  Verdict: {judgment['verdict']}")
        print(f"  Reasoning: {judgment['reasoning']}")
