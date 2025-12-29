#!/usr/bin/env python3
"""Expand data.jsonl with more heterogeneous reviews using LLM."""

import json
from llm import call_llm

EXPAND_PROMPT = """Given this restaurant and its existing reviews, generate {n} NEW additional reviews.

Restaurant: {name}
Neighborhood: {neighborhood}
Cuisine: {cuisine}
Price: {price}

Existing reviews:
{existing_reviews}

Generate {n} new reviews that:
1. Have CONTRADICTORY opinions (some positive, some negative, some mixed)
2. Include TIME-DEPENDENT observations ("quiet on weekdays, loud on weekends")
3. Cover DIFFERENT ASPECTS (some about noise, some about seating, some about price, some about allergies, some about authenticity)
4. Use VARIED STYLES (short vs long, emotional vs factual)
5. Include PARTIAL INFORMATION (not every review mentions every aspect)

For each review, also provide condition_satisfy scores (-1, 0, 1, or null) for:
- quiet_ambience, comfortable_seating, not_too_expensive
- allergen_care, clear_ingredient_labeling, low_cross_contamination_risk
- classic_chicago_dishes, authentic_local_vibe, tourist_friendly

Output as JSON array:
[{{"review_id": "...", "review": "...", "condition_satisfy": {{...}}}}, ...]
"""

def expand_restaurant(item, target_reviews=8):
    """Add more reviews to a restaurant."""
    current = len(item['item_data'])
    needed = target_reviews - current
    if needed <= 0:
        return item

    existing = "\n".join([f"- {r['review'][:200]}..." for r in item['item_data']])

    prompt = EXPAND_PROMPT.format(
        n=needed,
        name=item['item_name'],
        neighborhood=item['neighborhood'],
        cuisine=", ".join(item['cuisine']),
        price=item['price_range'],
        existing_reviews=existing
    )

    try:
        response = call_llm(prompt, system="You are a creative writer generating realistic restaurant reviews. Output valid JSON only.")
        # Parse JSON from response
        # Find JSON array in response
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            new_reviews = json.loads(response[start:end])
            # Add review IDs
            for i, r in enumerate(new_reviews):
                r['review_id'] = f"{item['item_id']}_r{current + i + 1}"
            item['item_data'].extend(new_reviews)
            print(f"  Added {len(new_reviews)} reviews")
    except Exception as e:
        print(f"  Error: {e}")

    return item


def main():
    with open('data.jsonl', 'r') as f:
        restaurants = [json.loads(line) for line in f]

    print(f"Expanding {len(restaurants)} restaurants to ~8 reviews each...")

    for item in restaurants:
        print(f"\n{item['item_name']} ({len(item['item_data'])} reviews)")
        expand_restaurant(item, target_reviews=8)
        print(f"  Now has {len(item['item_data'])} reviews")

    # Save expanded data
    with open('data_expanded.jsonl', 'w') as f:
        for item in restaurants:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved to data_expanded.jsonl")


if __name__ == "__main__":
    main()
