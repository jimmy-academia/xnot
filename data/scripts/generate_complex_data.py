#!/usr/bin/env python3
"""Generate complex evaluation dataset with 5-level scoring and structured requests."""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# 5-level evidence scoring keywords
# Score: -2 (strongly negative), -1 (somewhat negative), 0 (neutral), +1 (somewhat positive), +2 (strongly positive)
ASPECT_KEYWORDS = {
    "speed": {
        2: ["fastest", "immediate", "no wait at all", "instantly", "lightning fast", "right away"],
        1: ["fast", "quick", "prompt", "efficient", "didn't wait long"],
        -1: ["slow", "had to wait", "took a while", "bit slow", "could be faster"],
        -2: ["extremely slow", "waited forever", "over an hour", "terrible wait", "ridiculous wait"],
    },
    "food": {
        2: ["best food ever", "absolutely delicious", "incredible", "perfection", "amazing flavor", "phenomenal", "outstanding", "exceptional"],
        1: ["good food", "tasty", "fresh", "enjoyable", "solid", "well prepared", "delicious", "great", "excellent", "yummy", "loved it", "really good", "so good", "amazing", "fantastic"],
        -1: ["mediocre", "bland", "nothing special", "underwhelming", "could be better", "ok food", "just ok", "average"],
        -2: ["terrible food", "disgusting", "inedible", "awful", "worst ever", "made me sick", "horrible", "gross"],
    },
    "value": {
        2: ["incredible value", "amazing deal", "steal", "best bang for buck", "unbelievably cheap", "great prices"],
        1: ["good value", "reasonable", "fair price", "worth it", "affordable", "cheap", "inexpensive", "good prices", "great deal", "bang for the buck", "reasonably priced", "not expensive"],
        -1: ["bit pricey", "slightly overpriced", "expected more for price", "not great value", "pricey"],
        -2: ["rip off", "way overpriced", "highway robbery", "not worth it at all", "total waste", "too expensive", "expensive", "overpriced"],
    },
    "ambiance": {
        2: ["perfect atmosphere", "most romantic", "absolutely beautiful", "stunning decor", "magical"],
        1: ["nice atmosphere", "cozy", "pleasant", "comfortable", "good vibe", "charming"],
        -1: ["nothing special", "bit cramped", "could use work", "dated", "meh atmosphere"],
        -2: ["terrible atmosphere", "way too loud", "uncomfortable", "dirty", "awful vibe"],
    },
    "consistency": {
        2: ["always perfect", "never disappoints", "consistently excellent", "every single time"],
        1: ["usually good", "reliable", "consistent", "dependable", "trustworthy"],
        -1: ["hit or miss", "inconsistent", "varies", "sometimes good sometimes bad"],
        -2: ["used to be good", "completely changed", "went downhill", "unreliable", "never the same"],
    },
    "service": {
        2: ["best service ever", "incredibly attentive", "went above and beyond", "exceptional staff"],
        1: ["good service", "friendly staff", "helpful", "attentive", "professional"],
        -1: ["slow service", "inattentive", "had to flag down", "could improve"],
        -2: ["terrible service", "rude staff", "ignored us", "worst service", "unprofessional"],
    },
    "portions": {
        2: ["huge portions", "couldn't finish", "extremely generous", "massive servings"],
        1: ["good portions", "decent size", "satisfying amount", "filling"],
        -1: ["small portions", "bit skimpy", "left hungry", "could be bigger"],
        -2: ["tiny portions", "laughably small", "rip off portions", "barely any food"],
    },
    "drinks": {
        2: ["amazing wine list", "best cocktails", "incredible bar", "outstanding drinks"],
        1: ["good drinks", "nice selection", "solid bar", "good wine"],
        -1: ["limited selection", "drinks were ok", "nothing special on drinks"],
        -2: ["awful drinks", "watered down", "terrible bar", "overpriced drinks"],
    },
    "quiet": {
        2: ["perfectly quiet", "intimate", "peaceful", "serene", "whisper quiet"],
        1: ["reasonably quiet", "not too loud", "can have conversation", "pleasant noise level"],
        -1: ["bit noisy", "loud at times", "hard to hear", "somewhat loud"],
        -2: ["extremely loud", "deafening", "couldn't hear anything", "unbearable noise"],
    },
    "private": {
        2: ["private rooms", "secluded", "very private", "intimate booths"],
        1: ["some privacy", "booths available", "not too crowded"],
        -1: ["no privacy", "tables close together", "felt exposed"],
        -2: ["zero privacy", "packed like sardines", "everyone hears everything"],
    },
    "outdoor": {
        2: ["beautiful patio", "amazing outdoor space", "perfect outdoor seating"],
        1: ["has outdoor seating", "nice patio", "outdoor option"],
        -1: ["limited outdoor", "outdoor area needs work"],
        -2: ["no outdoor seating", "terrible patio", "outdoor unusable"],
    },
    "parking": {
        2: ["easy parking", "plenty of spots", "valet available", "parking lot"],
        1: ["parking nearby", "street parking", "manageable parking"],
        -1: ["parking difficult", "had to circle", "limited parking"],
        -2: ["no parking", "impossible to park", "parking nightmare"],
    },
    "freshness": {
        2: ["incredibly fresh", "farm to table", "freshest ingredients"],
        1: ["fresh ingredients", "quality produce", "fresh tasting"],
        -1: ["not very fresh", "questionable freshness"],
        -2: ["stale", "not fresh at all", "old ingredients", "freezer food"],
    },
    "vegetarian": {
        2: ["amazing vegetarian options", "vegetarian paradise", "best veggie menu"],
        1: ["good vegetarian options", "veggie friendly", "several vegetarian dishes"],
        -1: ["limited vegetarian", "few veggie options"],
        -2: ["no vegetarian options", "nothing for vegetarians", "meat only"],
    },
    "staff": {
        2: ["friendliest staff", "made us feel special", "remembered us"],
        1: ["friendly staff", "nice servers", "pleasant interaction"],
        -1: ["staff was ok", "nothing special about service"],
        -2: ["unfriendly staff", "rude servers", "made us uncomfortable"],
    },
    "ordering": {
        2: ["easy online ordering", "great app", "seamless ordering"],
        1: ["can order ahead", "phone ordering works", "takeout available"],
        -1: ["ordering confusing", "app doesn't work well"],
        -2: ["no advance ordering", "terrible ordering system", "order always wrong"],
    },
}


def compute_aspect_score(reviews: List[Dict], aspect: str) -> float:
    """Compute average 5-level score for an aspect from reviews."""
    if aspect not in ASPECT_KEYWORDS:
        return 0.0

    keywords = ASPECT_KEYWORDS[aspect]
    scores = []

    for review in reviews:
        text = review.get("review", review.get("text", "")).lower()
        review_score = 0
        found = False

        # Check each level
        for level in [2, 1, -1, -2]:
            for kw in keywords.get(level, []):
                if kw in text:
                    review_score = level
                    found = True
                    break
            if found:
                break

        if found:
            scores.append(review_score)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_condition(condition: Dict, aspect_scores: Dict[str, float]) -> int:
    """Evaluate a single condition based on aspect scores.

    Returns: 1 (satisfied), 0 (neutral), -1 (not satisfied)
    """
    if "op" in condition:
        # Compound condition
        return evaluate_compound(condition, aspect_scores)

    aspect = condition["aspect"]
    level = condition["level"]
    score = aspect_scores.get(aspect, 0.0)

    if level == "MUST":
        # Must have: avg >= 0.5 to pass, else fail (lowered from 1.0 for achievability)
        return 1 if score >= 0.5 else -1
    elif level == "SHOULD":
        # Should have: >= 0.3 good, < -0.3 bad, else neutral
        if score >= 0.3:
            return 1
        elif score < -0.3:
            return -1
        else:
            return 0
    elif level == "NICE":
        # Nice to have: > 0 is bonus, else neutral (never penalize)
        return 1 if score > 0 else 0
    else:
        return 0


def evaluate_compound(structure: Dict, aspect_scores: Dict[str, float]) -> int:
    """Evaluate compound AND/OR conditions."""
    op = structure["op"]
    conditions = structure["conditions"]

    results = [evaluate_condition(c, aspect_scores) for c in conditions]

    if op == "AND":
        return min(results)  # All must be satisfied
    elif op == "OR":
        return max(results)  # Any can satisfy
    else:
        return 0


def compute_final_recommendation(structure: Dict, aspect_scores: Dict[str, float]) -> int:
    """Compute final recommendation based on structured evaluation.

    Returns: 1 (recommend), 0 (unclear), -1 (not recommend)
    """
    # Check for any MUST failures first
    def has_must_failure(cond: Dict) -> bool:
        if "op" in cond:
            return any(has_must_failure(c) for c in cond["conditions"])
        elif cond.get("level") == "MUST":
            score = aspect_scores.get(cond["aspect"], 0.0)
            return score < 1.0
        return False

    if has_must_failure(structure):
        return -1  # Automatic fail if any MUST condition fails

    # Compute weighted sum
    def compute_weighted(cond: Dict) -> float:
        if "op" in cond:
            results = [compute_weighted(c) for c in cond["conditions"]]
            if cond["op"] == "AND":
                return min(results)
            else:  # OR
                return max(results)
        else:
            level = cond["level"]
            result = evaluate_condition(cond, aspect_scores)
            weights = {"MUST": 3, "SHOULD": 2, "NICE": 1}
            return result * weights.get(level, 1)

    total = compute_weighted(structure)

    if total >= 2:
        return 1  # Recommend
    elif total <= -2:
        return -1  # Not recommend
    else:
        return 0  # Unclear


def generate_complex_data(input_path: str, requests_path: str, output_path: str):
    """Generate complex dataset from existing restaurant data and complex requests."""
    # Load existing restaurant data
    restaurants = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                restaurants.append(json.loads(line))

    # Load complex requests
    with open(requests_path) as f:
        requests = json.load(f)

    print(f"Loaded {len(restaurants)} restaurants")
    print(f"Loaded {len(requests)} complex requests")

    # Generate new dataset with complex gold labels
    output_data = []
    label_stats = defaultdict(lambda: defaultdict(int))

    for rest in restaurants:
        reviews = rest.get("item_data", [])

        # Compute 5-level scores for all aspects
        aspect_scores = {}
        for aspect in ASPECT_KEYWORDS:
            aspect_scores[aspect] = compute_aspect_score(reviews, aspect)

        # Compute gold labels for each complex request
        gold_labels = {}
        for req in requests:
            label = compute_final_recommendation(req["structure"], aspect_scores)
            gold_labels[req["id"]] = label
            label_stats[req["id"]][label] += 1

        # Create output item
        output_item = {
            "item_id": rest["item_id"],
            "item_name": rest["item_name"],
            "city": rest.get("city", "Unknown"),
            "neighborhood": rest.get("neighborhood", "Unknown"),
            "price_range": rest.get("price_range", "$$"),
            "cuisine": rest.get("cuisine", []),
            "item_data": reviews,
            "aspect_scores": {k: round(v, 2) for k, v in aspect_scores.items()},
            "gold_labels": gold_labels,
        }
        output_data.append(output_item)

    # Save output
    with open(output_path, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(output_data)} items to {output_path}")
    print("\nLabel distribution:")
    for req_id in sorted(label_stats.keys()):
        dist = dict(label_stats[req_id])
        print(f"  {req_id}: {dist}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate complex evaluation dataset")
    parser.add_argument("--input", default="data/real_data.jsonl", help="Input restaurant data")
    parser.add_argument("--requests", default="data/complex_requests.json", help="Complex requests JSON")
    parser.add_argument("--output", default="data/complex_data.jsonl", help="Output file")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    requests_path = script_dir / args.requests
    output_path = script_dir / args.output

    generate_complex_data(str(input_path), str(requests_path), str(output_path))


if __name__ == "__main__":
    main()
