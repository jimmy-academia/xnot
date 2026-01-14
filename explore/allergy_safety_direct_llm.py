"""
Direct LLM test for Allergy Safety Policy task (G1a).

Each restaurant evaluated SEPARATELY (one LLM call per restaurant).
"""

import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import call_llm

DATA_DIR = Path(__file__).parent / "data"
GT_FILE = DATA_DIR / "semantic_gt" / "task_G1a" / "computed_gt.json"
DATASET_FILE = DATA_DIR / "dataset_K200.jsonl"

POLICY_PROMPT = """# ALLERGY SAFETY POLICY EVALUATION

## Goal
Assess peanut/nut allergy safety risk for this restaurant based on its reviews.

## Risk Levels
- **Low Risk**: No concerning allergy incidents
- **High Risk**: Some allergy concerns (1 moderate incident, or mild incidents with concerning patterns)
- **Critical Risk**: Severe allergy safety issues (severe incidents, multiple moderate incidents, or betrayal of trust)

## What to Look For
1. **Firsthand allergic reactions** - Customer experienced symptoms (hives, swelling, breathing issues, needed medication)
2. **Staff awareness issues** - Server didn't know about allergens, wrong item served despite allergy request
3. **Severity indicators**:
   - mild: discomfort, itching
   - moderate: required Benadryl/medication, significant reaction
   - severe: hospital visit, life-threatening, anaphylaxis

## Output Format
Output JSON:
```json
{{
  "restaurant_name": "{restaurant_name}",
  "verdict": "Low Risk|High Risk|Critical Risk",
  "evidence": [
    {{
      "review_idx": <number>,
      "incident_type": "allergic_reaction|staff_issue|wrong_order|none",
      "severity": "mild|moderate|severe|none",
      "quote": "<key phrase from review>"
    }}
  ],
  "reasoning": "<brief explanation>"
}}
```

---

# RESTAURANT: {restaurant_name}
- Categories: {categories}
- Stars: {stars}

## Reviews ({n_reviews} total):

{reviews_text}

---

Evaluate this restaurant's allergy safety risk. Output only JSON.
"""


def load_ground_truth():
    """Load computed ground truth."""
    with open(GT_FILE) as f:
        return json.load(f)


def load_dataset():
    """Load dataset indexed by restaurant name."""
    dataset = {}
    with open(DATASET_FILE) as f:
        for line in f:
            item = json.loads(line)
            name = item["business"]["name"]
            dataset[name] = item
    return dataset


def get_test_restaurants():
    """Get restaurants with known risk levels for testing."""
    gt = load_ground_truth()

    test_set = []
    for name, data in gt["restaurants"].items():
        verdict = data["verdict"]
        if verdict in ["High Risk", "Critical Risk"]:
            test_set.append({
                "name": name,
                "expected_verdict": verdict,
                "expected_score": data["final_risk_score"],
                "n_severe": data["n_severe"],
                "n_moderate": data["n_moderate"],
                "n_mild": data["n_mild"]
            })

    # Add a few low risk for comparison
    low_risk = [n for n, d in gt["restaurants"].items() if d["verdict"] == "Low Risk"][:3]
    for name in low_risk:
        data = gt["restaurants"][name]
        test_set.append({
            "name": name,
            "expected_verdict": "Low Risk",
            "expected_score": data["final_risk_score"],
            "n_severe": data["n_severe"],
            "n_moderate": data["n_moderate"],
            "n_mild": data["n_mild"]
        })

    return test_set


def format_reviews(reviews: list, max_reviews: int = 200) -> str:
    """Format reviews for prompt."""
    parts = []
    for i, rev in enumerate(reviews[:max_reviews]):
        text = rev["text"][:500] + "..." if len(rev["text"]) > 500 else rev["text"]
        parts.append(f"[Review #{i}] Date: {rev['date'][:10]} | Stars: {rev['stars']}\n{text}\n")
    return "\n".join(parts)


def evaluate_single_restaurant(name: str, dataset: dict, max_reviews: int = 200) -> dict:
    """Evaluate ONE restaurant with LLM."""
    if name not in dataset:
        return {"error": f"Restaurant not found: {name}"}

    item = dataset[name]
    biz = item["business"]
    reviews = item["reviews"][:max_reviews]

    prompt = POLICY_PROMPT.format(
        restaurant_name=name,
        categories=biz.get("categories", "N/A"),
        stars=biz.get("stars", "N/A"),
        n_reviews=len(reviews),
        reviews_text=format_reviews(reviews, max_reviews)
    )

    response = call_llm(prompt)

    # Parse JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            return {"raw_response": response, "error": "JSON parse failed"}
    return {"raw_response": response, "error": "No JSON found"}


def run_direct_llm_test(max_reviews: int = 200):
    """Run the direct LLM test - ONE restaurant at a time."""
    print("=" * 60)
    print("ALLERGY SAFETY - DIRECT LLM (One Restaurant at a Time)")
    print("=" * 60)

    # Load data
    print("\n1. Loading ground truth and dataset...")
    test_restaurants = get_test_restaurants()
    dataset = load_dataset()

    print(f"   Test set: {len(test_restaurants)} restaurants")
    print(f"   Reviews per restaurant: {max_reviews}")
    print("\n   Expected verdicts:")
    for r in test_restaurants:
        incidents = f"severe={r['n_severe']} mod={r['n_moderate']} mild={r['n_mild']}"
        print(f"   - {r['expected_verdict']:13} | {incidents:25} | {r['name']}")

    # Evaluate each restaurant separately
    print(f"\n2. Evaluating each restaurant (separate LLM calls)...")

    results = []
    for i, r in enumerate(test_restaurants):
        name = r["name"]
        print(f"   [{i+1}/{len(test_restaurants)}] {name}...", end=" ", flush=True)

        prediction = evaluate_single_restaurant(name, dataset, max_reviews)
        prediction["expected"] = r["expected_verdict"]
        results.append(prediction)

        if "verdict" in prediction:
            match = "✓" if prediction["verdict"] == r["expected_verdict"] else "✗"
            print(f"{match} {prediction['verdict']}")
        else:
            print(f"ERROR: {prediction.get('error', 'unknown')}")

    # Save results
    output_dir = Path(__file__).parent / "results" / "allergy_safety_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_dir / f"direct_llm_separate_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n3. Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON WITH GROUND TRUTH")
    print("=" * 60)

    correct = 0
    total = len(results)

    for r in results:
        name = r.get("restaurant_name", "UNKNOWN")
        expected = r.get("expected", "?")
        predicted = r.get("verdict", "ERROR")

        match = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1

        # Show evidence if any
        evidence_str = ""
        if r.get("evidence"):
            for e in r["evidence"][:2]:
                evidence_str += f"\n      idx={e.get('review_idx')}, {e.get('severity')}: \"{e.get('quote', '')[:50]}...\""

        print(f"{match} {expected:13} vs {predicted:13} | {name}{evidence_str}")

    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=int, default=200, help="Reviews per restaurant")
    args = parser.parse_args()

    run_direct_llm_test(max_reviews=args.reviews)
