"""
Hard test for Allergy Safety - tests semantic operator challenges.

Challenges:
1. sem_filter: Find relevant reviews among many
2. sem_extract: Extract severity, firsthand, date correctly
3. sem_join: Map allergens to ontology (exclude coconut, etc.)
4. sem_distinct: Deduplicate same incident from multiple reviewers
"""

import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import call_llm

DATA_DIR = Path(__file__).parent / "data" / "synthetic_hard"

HARD_SPEC_PROMPT = """# Agenda Spec v3 — Peanut/Tree Nut Allergy Risk (HARD)

## Scope
Assess **peanut and tree nut** allergy risk for one restaurant based on customer reviews.

## Allergen Ontology (CRITICAL - must follow exactly)

### COVERED allergens (count these):
- **Peanuts** (Arachis hypogaea)
- **Tree nuts**: almond, Brazil nut, cashew, chestnut, hazelnut (filbert), macadamia, pecan, pine nut, pistachio, walnut

### NOT COVERED (must EXCLUDE):
- **Coconut** — NOT a tree nut per FDA (it's a drupe)
- **Nutmeg** — a seed, not a nut
- **Water chestnut** — a vegetable
- **Seeds**: sesame, sunflower, pumpkin
- **Other allergens**: dairy, eggs, shellfish, gluten, soy, fish sauce

If a review describes a reaction to a NON-COVERED allergen, do NOT count it.

## Definitions

### Incident
An allergic reaction to a COVERED allergen that occurred after eating at this restaurant.

### Firsthand
- ✓ Firsthand: The reviewer OR their immediate dining party (spouse, child, friend at same meal) experienced it
- ✗ Secondhand: Staff told them, read online, friend mentioned later, "I heard that...", witness but not victim

### Severity
- **Severe**: Anaphylaxis, throat closing, EpiPen, ER, hospitalization, "almost died"
- **Moderate**: Benadryl/antihistamine required, significant hives, facial swelling, vomiting
- **Mild**: Minor itching, tingling lips, slight discomfort
- **NOT an incident**: "I carry EpiPen but didn't need it", "was worried but nothing happened"

### Recency
- Recent: 2017-01-01 or later (given today is 2020-01-01)

### Independence (CRITICAL)
Two reviews are the SAME incident if:
- They cross-reference each other (e.g., "see my wife's review", "as mentioned in R002")
- Same date + same described party + same allergen

Count same incident only ONCE even if multiple reviews describe it.

## Labeling Policy

```
IF independent_firsthand_recent_severe_covered >= 2:
    verdict = "Critical"
ELIF independent_firsthand_recent_any_covered >= 1:
    verdict = "High Risk"
ELSE:
    verdict = "Low Risk"
```

## Output Schema

Output this exact JSON structure:

```json
{{
  "verdict": "Low Risk | High Risk | Critical",

  "sem_filter_results": {{
    "total_reviews": <int>,
    "potentially_relevant_ids": ["<review_ids that mention any allergy>"]
  }},

  "sem_extract_results": [
    {{
      "review_id": "<id>",
      "allergen_raw": "<what review says>",
      "severity": "mild | moderate | severe | none",
      "firsthand": true | false,
      "date": "YYYY-MM-DD",
      "is_incident": true | false,
      "exclusion_reason": "<if not an incident, why>"
    }}
  ],

  "sem_join_results": [
    {{
      "review_id": "<id>",
      "allergen_raw": "<what review says>",
      "allergen_canonical": "peanut | tree_nut:X | NOT_COVERED:X",
      "is_covered": true | false
    }}
  ],

  "sem_distinct_results": {{
    "incident_groups": [
      {{
        "group_id": <int>,
        "review_ids": ["<ids>"],
        "reason": "independent | same_as:<other_id>"
      }}
    ],
    "independent_incident_count": <int>
  }},

  "final_counts": {{
    "independent_firsthand_recent_severe_covered": <int>,
    "independent_firsthand_recent_any_covered": <int>
  }},

  "reasoning": "<explanation>"
}}
```

---

# RESTAURANT: {restaurant_name}
Categories: {categories}

## Reviews:

{reviews_text}

---

Analyze carefully following the spec. Show your work for each semantic operator.
"""


def load_synthetic_data(filename: str) -> dict:
    """Load synthetic test data."""
    with open(DATA_DIR / filename) as f:
        return json.load(f)


def format_reviews(reviews: list) -> str:
    """Format reviews for prompt."""
    parts = []
    for rev in reviews:
        parts.append(f"[review_id: {rev['review_id']}] [date: {rev['date']}] [stars: {rev['stars']}]\n{rev['text']}\n")
    return "\n".join(parts)


def evaluate_restaurant(data: dict) -> dict:
    """Evaluate one restaurant with LLM."""
    restaurant = data["restaurant"]
    reviews = data["reviews"]

    prompt = HARD_SPEC_PROMPT.format(
        restaurant_name=restaurant["name"],
        categories=restaurant.get("categories", "N/A"),
        reviews_text=format_reviews(reviews)
    )

    print(f"   Calling LLM for {restaurant['name']} ({len(reviews)} reviews)...")
    import time
    start = time.time()
    response = call_llm(prompt)
    elapsed = time.time() - start
    print(f"   LLM response received in {elapsed:.1f}s")

    # Parse JSON
    import re
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            result = json.loads(json_match.group())
            result["raw_response"] = response
            return result
        except Exception as e:
            return {"raw_response": response, "error": f"JSON parse failed: {e}"}
    return {"raw_response": response, "error": "No JSON found"}


def compare_with_ground_truth(prediction: dict, ground_truth: dict) -> dict:
    """Compare LLM prediction with ground truth."""
    errors = []

    # 1. Check verdict
    gt_verdict = ground_truth["final_verdict"]["verdict"]
    pred_verdict = prediction.get("verdict", "ERROR")
    verdict_correct = pred_verdict == gt_verdict

    if not verdict_correct:
        errors.append(f"VERDICT: expected {gt_verdict}, got {pred_verdict}")

    # 2. Check sem_join (allergen ontology)
    gt_join = ground_truth["sem_join"]
    pred_join = {r["review_id"]: r for r in prediction.get("sem_join_results", [])}

    join_errors = []
    for rid, gt in gt_join.items():
        if rid in pred_join:
            pred = pred_join[rid]
            if gt["covered"] != pred.get("is_covered"):
                join_errors.append(f"{rid}: expected covered={gt['covered']}, got {pred.get('is_covered')}")
        else:
            join_errors.append(f"{rid}: missing from sem_join_results")

    if join_errors:
        errors.append(f"SEM_JOIN errors: {join_errors}")

    # 3. Check sem_distinct (deduplication)
    gt_distinct = ground_truth["sem_distinct"]
    gt_independent = gt_distinct.get("independent_incidents", gt_distinct.get("independent_firsthand_recent_covered", -1))
    pred_distinct = prediction.get("sem_distinct_results", {})
    pred_independent = pred_distinct.get("independent_incident_count", -1)

    if gt_independent != -1 and pred_independent != gt_independent:
        errors.append(f"SEM_DISTINCT: expected {gt_independent} independent incidents, got {pred_independent}")

    # 4. Check final counts
    gt_severe_count = ground_truth["final_verdict"]["independent_firsthand_recent_severe_covered"]
    pred_counts = prediction.get("final_counts", {})
    pred_severe_count = pred_counts.get("independent_firsthand_recent_severe_covered", -1)

    if pred_severe_count != gt_severe_count:
        errors.append(f"SEVERE_COUNT: expected {gt_severe_count}, got {pred_severe_count}")

    return {
        "verdict_correct": verdict_correct,
        "errors": errors,
        "error_count": len(errors)
    }


def run_hard_test(dataset: str = "restaurant_nutty_thai.json"):
    """Run the hard test on synthetic data."""
    print("=" * 70)
    print("ALLERGY SAFETY - HARD TEST (Semantic Operators)")
    print("=" * 70)

    # Load test data
    print(f"\n1. Loading synthetic test data: {dataset}")
    data = load_synthetic_data(dataset)
    gt = data["ground_truth"]

    print(f"   Restaurant: {data['restaurant']['name']}")
    print(f"   Reviews: {len(data['reviews'])}")
    print(f"   Ground truth verdict: {gt['final_verdict']['verdict']}")
    independent_count = gt['sem_distinct'].get('independent_incidents', gt['sem_distinct'].get('independent_firsthand_recent_covered', '?'))
    print(f"   Independent incidents: {independent_count}")
    print(f"   Severe+covered: {gt['final_verdict']['independent_firsthand_recent_severe_covered']}")

    # Evaluate
    print("\n2. Running LLM evaluation...")
    prediction = evaluate_restaurant(data)

    if "error" in prediction:
        print(f"   ERROR: {prediction['error']}")
        print(f"   Raw response:\n{prediction.get('raw_response', '')[:1000]}")
        return

    # Save results
    output_dir = Path(__file__).parent / "results" / "allergy_safety_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_dir / f"hard_test_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"prediction": prediction, "ground_truth": gt}, f, indent=2)

    print(f"   Results saved to: {results_file}")

    # Compare with ground truth
    print("\n3. Comparing with ground truth...")
    comparison = compare_with_ground_truth(prediction, gt)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n## Verdict")
    print(f"   Expected: {gt['final_verdict']['verdict']}")
    print(f"   Got:      {prediction.get('verdict', 'ERROR')}")
    print(f"   Correct:  {'✓' if comparison['verdict_correct'] else '✗'}")

    print(f"\n## Semantic Operator Performance")

    # sem_filter
    pred_filter = prediction.get("sem_filter_results", {})
    print(f"\n### sem_filter")
    print(f"   Relevant IDs found: {pred_filter.get('potentially_relevant_ids', [])}")

    # sem_join
    print(f"\n### sem_join (allergen ontology)")
    for r in prediction.get("sem_join_results", []):
        covered = "✓" if r.get("is_covered") else "✗"
        print(f"   {r.get('review_id')}: {r.get('allergen_raw')} → {r.get('allergen_canonical')} [{covered}]")

    # sem_distinct
    print(f"\n### sem_distinct (deduplication)")
    pred_distinct = prediction.get("sem_distinct_results", {})
    print(f"   Independent incidents: {pred_distinct.get('independent_incident_count', '?')} (expected: {gt['sem_distinct']['independent_incidents']})")
    for group in pred_distinct.get("incident_groups", []):
        print(f"   Group {group.get('group_id')}: {group.get('review_ids')} - {group.get('reason')}")

    # Final counts
    print(f"\n### Final Counts")
    pred_counts = prediction.get("final_counts", {})
    print(f"   Severe+covered: {pred_counts.get('independent_firsthand_recent_severe_covered', '?')} (expected: {gt['final_verdict']['independent_firsthand_recent_severe_covered']})")
    print(f"   Any+covered:    {pred_counts.get('independent_firsthand_recent_any_covered', '?')}")

    # Errors
    if comparison["errors"]:
        print(f"\n## Errors ({comparison['error_count']})")
        for err in comparison["errors"]:
            print(f"   ✗ {err}")
    else:
        print(f"\n## All checks passed! ✓")

    return prediction, comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="restaurant_nutty_thai.json",
                        help="Synthetic dataset to test")
    args = parser.parse_args()

    run_hard_test(dataset=args.dataset)
