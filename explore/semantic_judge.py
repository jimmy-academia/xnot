#!/usr/bin/env python3
"""
Semantic LLM Judge for G1.v14
Generates ground truth using semantic reasoning primitives.
Outputs structured for human validation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm


SEMANTIC_JUDGE_PROMPT = """You are an expert safety analyst helping parents with severe peanut allergies assess restaurant safety from reviews.

# RESTAURANT CONTEXT
Name: {restaurant_name}
Categories: {categories}
Number of Reviews: {num_reviews}

# YOUR TASK
Analyze the reviews using SEMANTIC UNDERSTANDING (not just keyword matching) to assess 4 key primitives:

---

## PRIMITIVE 1: Firsthand Severe Incidents
**Count how many reviews describe a PERSONAL severe allergic reaction that the reviewer or their dependent experienced at this restaurant.**

Criteria:
- Must be first-person narrative (I/we/my child had the reaction)
- Must be severe (hospitalization, anaphylaxis, serious symptoms)
- NOT hearsay ("I heard someone..."), warnings, or concerns

Examples:
✓ "My daughter had an allergic reaction and we rushed to the hospital"
✓ "Despite asking about peanuts, I still got very sick"
✗ "I heard someone had a bad reaction" (hearsay)
✗ "Be careful if you have allergies" (warning, not incident)

**Output:** Integer count

---

## PRIMITIVE 2: Safety Trajectory  
**Assess whether allergy safety is improving, stable, or worsening over time.**

Consider:
- Temporal pattern in allergy incidents
- Management changes mentioned
- Recent vs old evidence
- Systemic vs isolated issues

Output: One of:
- "improving" (recent evidence shows better practices)
- "stable" (consistent pattern over time)
- "worsening" (recent deterioration in safety)

---

## PRIMITIVE 3: False Assurance Count
**Count reviews where the restaurant CLAIMED to be allergy-safe BUT the customer still had issues.**

Requires understanding:
- What constitutes a safety claim (explicit or implicit)
- Whether adverse outcome occurred DESPITE the claim
- Distinguishing accommodation from lip service

Examples:
✓ "They said they could handle allergies but used contaminated oil"
✓ "Menu says 'allergy-friendly' but I had a reaction"
✗ "Great allergy menu and felt safe" (claim validated)

**Output:** Integer count

---

## PRIMITIVE 4: Evidence Consensus
**Among reviews discussing allergy safety, assess agreement vs conflict.**

Scale: 0.0 (conflicting) to 1.0 (consensus)

Examples:
- Low (0.2): "Excellent for allergies" vs "Had severe reaction"
- High (0.9): Multiple consistent "very accommodating" reviews
- High (0.8): Multiple consistent "not safe" reviews

**Output:** Float 0.0-1.0

---

# REVIEWS
{reviews_text}

---

# OUTPUT FORMAT (JSON)
Provide your assessment in this exact JSON structure:

{{
    "firsthand_severe_count": <integer>,
    "safety_trajectory": "<improving|stable|worsening>",
    "false_assurance_count": <integer>,
    "evidence_consensus": <float 0.0-1.0>,
    "final_risk_score": <float 0-20>,
    "verdict": "<Low Risk|High Risk|Critical Risk>",
    "reasoning": "<2-3 sentence justification explaining your assessment>"
}}

Be CONSERVATIVE - err on the side of higher risk for allergy safety.
Think step-by-step about each primitive before outputting.
"""


def generate_semantic_judgment(restaurant: dict, reviews: list) -> dict:
    """Generate semantic ground truth using LLM judge."""
    
    # Format reviews
    reviews_text = "\n\n".join([
        f"Review {i+1} ({r.get('stars', '?')} stars, {r.get('date', 'no date')}):\n{r.get('text', '')[:500]}"
        for i, r in enumerate(reviews[:50])
    ])
    
    prompt = SEMANTIC_JUDGE_PROMPT.format(
        restaurant_name=restaurant.get('name', 'Unknown'),
        categories=restaurant.get('categories', 'Unknown'),
        num_reviews=len(reviews),
        reviews_text=reviews_text
    )
    
    # Call LLM
    response = call_llm(prompt)
    
    # Parse JSON
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        judgment = json.loads(json_str)
        
        # Validate structure
        required = ['firsthand_severe_count', 'safety_trajectory', 'false_assurance_count',
                   'evidence_consensus', 'final_risk_score', 'verdict', 'reasoning']
        for key in required:
            if key not in judgment:
                raise ValueError(f"Missing key: {key}")
        
        return judgment
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR parsing judgment: {e}")
        print(f"Response: {response[:300]}")
        return {
            'firsthand_severe_count': 0,
            'safety_trajectory': 'stable',
            'false_assurance_count': 0,
            'evidence_consensus': 0.5,
            'final_risk_score': 5.0,
            'verdict': 'High Risk',
            'reasoning': f'Parse error: {str(e)}'
        }


def save_for_human_review(judgments: list, output_path: str):
    """Save judgments in human-reviewable format."""
    
    # Create review directory
    review_dir = Path(output_path)
    review_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full JSON
    with open(review_dir / 'all_judgments.json', 'w') as f:
        json.dump(judgments, f, indent=2)
    
    # Create human-friendly markdown
    with open(review_dir / 'human_review.md', 'w') as f:
        f.write("# LLM Judge Ground Truth - Human Validation\n\n")
        f.write(f"Total Restaurants: {len(judgments)}\n\n")
        f.write("## Instructions\n")
        f.write("Review each assessment. Mark ✓ if correct, ✗ if incorrect, ? if uncertain.\n\n")
        f.write("---\n\n")
        
        for i, j in enumerate(judgments):
            f.write(f"## {i+1}. {j['restaurant']}\n\n")
            f.write(f"**Firsthand Severe:** {j['firsthand_severe_count']}\n\n")
            f.write(f"**Safety Trajectory:** {j['safety_trajectory']}\n\n")
            f.write(f"**False Assurance:** {j['false_assurance_count']}\n\n")
            f.write(f"**Consensus:** {j['evidence_consensus']:.2f}\n\n")
            f.write(f"**Final Risk:** {j['final_risk_score']:.1f} → {j['verdict']}\n\n")
            f.write(f"**Reasoning:** {j['reasoning']}\n\n")
            f.write(f"**Human Assessment:** [ ] ✓ Correct  [ ] ✗ Incorrect  [ ] ? Uncertain\n\n")
            f.write("---\n\n")
    
    print(f"\n✓ Saved {len(judgments)} judgments to {review_dir}")
    print(f"  - all_judgments.json (machine-readable)")
    print(f"  - human_review.md (human validation)")


if __name__ == '__main__':
    print("Use generate_judgments.py to run full judgment generation")
