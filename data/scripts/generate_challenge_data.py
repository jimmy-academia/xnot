#!/usr/bin/env python3
"""Generate challenging dataset from real Yelp reviews that favors dynamic reasoning."""

import json
import random
from collections import defaultdict

# Load real Yelp data
def load_yelp_data(max_businesses=50, max_reviews_per_biz=15):
    """Load restaurants with diverse review patterns."""
    print("Loading Yelp businesses...")
    businesses = {}
    with open('data/yelp_academic_dataset_business.json') as f:
        for line in f:
            b = json.loads(line)
            cats = b.get('categories', '') or ''
            if 'Restaurant' in cats and b.get('review_count', 0) >= 20:
                businesses[b['business_id']] = {
                    'item_id': b['business_id'][:12],
                    'item_name': b['name'],
                    'city': b.get('city', ''),
                    'neighborhood': b.get('neighborhood', '') or b.get('city', ''),
                    'price_range': '$' * int(b.get('RestaurantsPriceRange2', 2) or 2),
                    'cuisine': [c.strip() for c in cats.split(',')[:3]],
                    'stars': b.get('stars', 3.0),
                    'item_data': []
                }
            if len(businesses) >= max_businesses * 3:
                break

    print(f"Loading reviews for {len(businesses)} businesses...")
    biz_ids = set(businesses.keys())
    review_count = defaultdict(int)

    with open('data/yelp_academic_dataset_review.json') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0:
                print(f"  Processed {i} reviews...")
            r = json.loads(line)
            bid = r['business_id']
            if bid in biz_ids and review_count[bid] < max_reviews_per_biz:
                businesses[bid]['item_data'].append({
                    'review_id': r['review_id'][:10],
                    'review': r['text'],
                    'stars': r['stars'],
                    'date': r['date'][:10]
                })
                review_count[bid] += 1
            if all(review_count[b] >= max_reviews_per_biz for b in biz_ids):
                break
            if i > 2000000:
                break

    # Filter to those with enough reviews
    result = [b for b in businesses.values() if len(b['item_data']) >= 8]
    random.shuffle(result)
    return result[:max_businesses]


def categorize_reviews(reviews):
    """Analyze review patterns present in a restaurant."""
    patterns = {
        'has_time_dependent': False,
        'has_contradictions': False,
        'has_conditional': False,
        'has_quantified': False,
        'star_variance': 0,
        'length_variance': 0
    }

    stars = [r['stars'] for r in reviews]
    lengths = [len(r['review']) for r in reviews]

    patterns['star_variance'] = max(stars) - min(stars) if stars else 0
    patterns['length_variance'] = max(lengths) - min(lengths) if lengths else 0

    time_words = ['used to', 'has changed', 'recently', 'now it', 'before', 'last time', 'first time', 'this time', 'anymore']
    conditional = ['great for', 'perfect for', 'not for', 'if you', 'unless', 'depends on', 'only if']
    quantified = ['waited', 'minute', 'hour', 'times', 'visits']

    for r in reviews:
        text = r['review'].lower()
        if any(tw in text for tw in time_words):
            patterns['has_time_dependent'] = True
        if any(c in text for c in conditional):
            patterns['has_conditional'] = True
        if any(q in text for q in quantified):
            patterns['has_quantified'] = True
        if 'but' in text or 'however' in text:
            if ('good' in text or 'great' in text) and ('bad' in text or 'slow' in text or 'terrible' in text):
                patterns['has_contradictions'] = True

    return patterns


# User requests that require different reasoning approaches
USER_REQUESTS = [
    # R0: Speed-focused (requires finding time/wait mentions)
    {
        'id': 'R0',
        'context': "I'm in a hurry and need quick service. Is the wait time reasonable?",
        'focus': ['speed', 'wait', 'time', 'quick', 'fast', 'slow', 'minute', 'hour'],
        'reasoning': 'Must find and analyze wait time mentions specifically'
    },
    # R1: Consistency check (requires comparing across reviews/time)
    {
        'id': 'R1',
        'context': "I've heard mixed things. Is this place consistent in quality?",
        'focus': ['consistent', 'always', 'sometimes', 'depends', 'varies', 'used to', 'changed'],
        'reasoning': 'Must compare across reviews and check for temporal changes'
    },
    # R2: Occasion-specific (requires conditional reasoning)
    {
        'id': 'R2',
        'context': "Planning a special dinner date. Good for romantic occasions?",
        'focus': ['date', 'romantic', 'ambiance', 'quiet', 'intimate', 'loud', 'crowded', 'atmosphere'],
        'reasoning': 'Must find ambiance mentions and infer romantic suitability'
    },
    # R3: Value assessment (requires price vs quality analysis)
    {
        'id': 'R3',
        'context': "Is it worth the price? Looking for good value, not necessarily cheap.",
        'focus': ['price', 'value', 'worth', 'expensive', 'cheap', 'quality', 'portion'],
        'reasoning': 'Must weigh price mentions against quality mentions'
    },
    # R4: Aspect contradiction (requires multi-aspect synthesis)
    {
        'id': 'R4',
        'context': "I care more about food quality than service. How's the food?",
        'focus': ['food', 'dish', 'taste', 'flavor', 'fresh', 'delicious', 'quality'],
        'reasoning': 'Must separate food comments from service comments and weight food higher'
    }
]


def compute_gold_label(restaurant, request):
    """Compute gold label based on review analysis."""
    reviews = restaurant['item_data']
    focus = request['focus']

    positive_signals = 0
    negative_signals = 0
    neutral_signals = 0

    for r in reviews:
        text = r['review'].lower()
        stars = r['stars']

        # Check if review is relevant to this request
        relevance = sum(1 for f in focus if f in text)
        if relevance == 0:
            continue

        # Weight by relevance and star rating
        if stars >= 4:
            positive_signals += relevance
        elif stars <= 2:
            negative_signals += relevance
        else:
            neutral_signals += relevance

        # Check for explicit negative mentions of focus topics
        negatives = ['slow', 'long wait', 'bad', 'terrible', 'awful', 'worst', 'disappointing', 'overpriced', 'loud', 'crowded', 'rude']
        positives = ['fast', 'quick', 'great', 'excellent', 'amazing', 'best', 'worth', 'quiet', 'friendly', 'perfect']

        for neg in negatives:
            if neg in text:
                negative_signals += 1
        for pos in positives:
            if pos in text:
                positive_signals += 1

    # Compute final label
    total = positive_signals + negative_signals + neutral_signals
    if total == 0:
        return 0  # No relevant info

    pos_ratio = positive_signals / (positive_signals + negative_signals + 0.1)

    if pos_ratio > 0.65:
        return 1
    elif pos_ratio < 0.35:
        return -1
    else:
        return 0


def main():
    restaurants = load_yelp_data(max_businesses=20, max_reviews_per_biz=12)
    print(f"\nLoaded {len(restaurants)} restaurants")

    # Analyze patterns
    for r in restaurants:
        r['patterns'] = categorize_reviews(r['item_data'])
        print(f"  {r['item_name']}: {len(r['item_data'])} reviews, variance={r['patterns']['star_variance']}")

    # Select diverse restaurants
    # Prefer those with high star variance (contradictions) and pattern diversity
    restaurants.sort(key=lambda x: (
        x['patterns']['star_variance'] +
        x['patterns']['has_contradictions'] * 2 +
        x['patterns']['has_time_dependent'] * 2 +
        x['patterns']['has_conditional']
    ), reverse=True)

    selected = restaurants[:10]

    # Generate gold labels
    output = []
    for rest in selected:
        # Remove patterns from output (internal use only)
        patterns = rest.pop('patterns')

        # Compute gold for each request
        rest['gold_labels'] = {}
        for req in USER_REQUESTS:
            gold = compute_gold_label(rest, req)
            rest['gold_labels'][req['id']] = gold

        output.append(rest)
        print(f"\n{rest['item_name']}:")
        print(f"  Reviews: {len(rest['item_data'])}")
        print(f"  Gold labels: {rest['gold_labels']}")

    # Save
    with open('data_challenge.jsonl', 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')

    # Also save user requests
    with open('requests.json', 'w') as f:
        json.dump(USER_REQUESTS, f, indent=2)

    print(f"\nSaved {len(output)} restaurants to data_challenge.jsonl")
    print(f"Saved {len(USER_REQUESTS)} requests to requests.json")


if __name__ == "__main__":
    main()
