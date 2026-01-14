"""
Direct LLM test for Allergy Safety Policy - V2 Agenda Spec.

Uses the precise agenda spec with clear labeling policy.
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

AGENDA_SPEC_PROMPT = """# Agenda Spec — Peanut/Nut Allergy Risk (per restaurant) — v1

## Scope
You are given all reviews for one restaurant. Produce exactly one risk verdict for this restaurant.

## Goal
Assess peanut/nut allergy risk from customer reviews and output a risk verdict supported by evidence.

## Risk Verdict (choose exactly one)
- Low Risk
- High Risk
- Critical

## Definitions

- **Review / Report**: one customer review text with review_id and a review date.
- **Incident**: the review describes an allergic reaction that happened after eating at the restaurant.
- **First-hand**: the author (or their party/child) directly experienced the incident.
- **Severe**: life-threatening symptoms or emergency response (e.g., anaphylaxis, throat closing, EpiPen, ER/hospital).
- **Independent**: incidents written by different reviewers (different author identity/user id; if unavailable, treat different review_id as independent).
- **Recent**: within the last 3 years (i.e., date >= 2017-01-01 given today is 2020-01-01).

## Semantic relevance requirement (important)
When searching for allergy incidents, you must consider semantic descriptions, not just the word "allergy".
Examples of relevant descriptions include symptoms (hives, swelling, throat closing), emergency actions (EpiPen, ER), "nut-free" assurances, cross-contamination, etc.

## Evidence extraction requirement
For every evidence item you include, you must extract these fields (or output "unknown" if not determinable from the review):
- severity: one of mild | moderate | severe | unknown
- firsthand: one of true | false | unknown
- date: YYYY-MM-DD | unknown

## Labeling policy

- **Critical** if there exist at least 2 independent, first-hand, recent incidents with severity = severe.
- **High Risk** if not Critical and there exists at least 1 first-hand incident (any severity) that is recent.
- **Low Risk** otherwise.

## Deliverables (expected result schema)
Output a single JSON object with this schema:

```json
{{
  "verdict": "Low Risk | High Risk | Critical",
  "evidences": [
    {{
      "review_id": "string",
      "quote": "string",
      "extracted_fields": {{
        "severity": "mild | moderate | severe | unknown",
        "firsthand": "true | false | unknown",
        "date": "YYYY-MM-DD | unknown"
      }}
    }}
  ]
}}
```

## Evidence list rules
- For Critical and High Risk, evidences must include the minimum set of items needed to justify the verdict under the labeling policy.
- For Low Risk, evidences may be empty.

---

# RESTAURANT: {restaurant_name}
- Categories: {categories}

## Reviews:

{reviews_text}

---

Analyze the reviews and output the JSON verdict with evidence.
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
                "gt_verdict": verdict,
                "gt_score": data["final_risk_score"],
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
            "gt_verdict": "Low Risk",
            "gt_score": data["final_risk_score"],
            "n_severe": data["n_severe"],
            "n_moderate": data["n_moderate"],
            "n_mild": data["n_mild"]
        })

    return test_set


def format_reviews(reviews: list, max_reviews: int = 200) -> str:
    """Format reviews for prompt with review_id."""
    parts = []
    for i, rev in enumerate(reviews[:max_reviews]):
        review_id = rev.get("review_id", f"review_{i}")
        date = rev["date"][:10]
        text = rev["text"][:600] + "..." if len(rev["text"]) > 600 else rev["text"]
        parts.append(f"[review_id: {review_id}] [date: {date}]\n{text}\n")
    return "\n".join(parts)


def evaluate_single_restaurant(name: str, dataset: dict, max_reviews: int = 200) -> dict:
    """Evaluate ONE restaurant with LLM."""
    if name not in dataset:
        return {"error": f"Restaurant not found: {name}"}

    item = dataset[name]
    biz = item["business"]
    reviews = item["reviews"][:max_reviews]

    prompt = AGENDA_SPEC_PROMPT.format(
        restaurant_name=name,
        categories=biz.get("categories", "N/A"),
        reviews_text=format_reviews(reviews, max_reviews)
    )

    response = call_llm(prompt)

    # Parse JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            result = json.loads(json_match.group())
            result["raw_response"] = response
            return result
        except:
            return {"raw_response": response, "error": "JSON parse failed"}
    return {"raw_response": response, "error": "No JSON found"}


def evaluate_against_agenda_spec(result: dict) -> dict:
    """
    Evaluate if the LLM's verdict is consistent with its own evidence per the agenda spec.

    Returns analysis of whether evidence supports the verdict.
    """
    verdict = result.get("verdict", "")
    evidences = result.get("evidences", [])

    # Count qualifying evidence
    recent_firsthand_severe = 0
    recent_firsthand_any = 0

    for e in evidences:
        fields = e.get("extracted_fields", {})
        firsthand = fields.get("firsthand", "unknown")
        severity = fields.get("severity", "unknown")
        date_str = fields.get("date", "unknown")

        # Check if recent (>= 2017-01-01, given "today" is 2020-01-01)
        is_recent = False
        if date_str != "unknown":
            try:
                year = int(date_str[:4])
                is_recent = year >= 2017
            except:
                pass

        is_firsthand = firsthand == "true"
        is_severe = severity == "severe"

        if is_firsthand and is_recent:
            recent_firsthand_any += 1
            if is_severe:
                recent_firsthand_severe += 1

    # Determine what verdict SHOULD be per agenda spec
    if recent_firsthand_severe >= 2:
        expected_verdict = "Critical"
    elif recent_firsthand_any >= 1:
        expected_verdict = "High Risk"
    else:
        expected_verdict = "Low Risk"

    # Normalize verdict for comparison
    verdict_normalized = verdict.replace("Critical Risk", "Critical")

    return {
        "verdict_given": verdict,
        "verdict_expected_from_evidence": expected_verdict,
        "consistent": verdict_normalized == expected_verdict,
        "recent_firsthand_severe_count": recent_firsthand_severe,
        "recent_firsthand_any_count": recent_firsthand_any,
        "total_evidences": len(evidences)
    }


def run_test(max_reviews: int = 200):
    """Run the direct LLM test with agenda spec v2."""
    print("=" * 70)
    print("ALLERGY SAFETY - DIRECT LLM (Agenda Spec V2)")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    test_restaurants = get_test_restaurants()
    dataset = load_dataset()

    print(f"   Test set: {len(test_restaurants)} restaurants")
    print(f"   Reviews per restaurant: up to {max_reviews}")

    # Evaluate each restaurant
    print(f"\n2. Evaluating each restaurant...")

    results = []
    for i, r in enumerate(test_restaurants):
        name = r["name"]
        print(f"   [{i+1}/{len(test_restaurants)}] {name}...", end=" ", flush=True)

        prediction = evaluate_single_restaurant(name, dataset, max_reviews)
        prediction["restaurant_name"] = name
        prediction["gt_verdict"] = r["gt_verdict"]

        # Evaluate consistency with agenda spec
        if "verdict" in prediction:
            analysis = evaluate_against_agenda_spec(prediction)
            prediction["agenda_spec_analysis"] = analysis

            verdict = prediction["verdict"]
            consistent = "✓" if analysis["consistent"] else "✗"
            print(f"{verdict:12} | evidence→{analysis['verdict_expected_from_evidence']:12} {consistent}")
        else:
            print(f"ERROR: {prediction.get('error', 'unknown')}")

        results.append(prediction)

    # Save results
    output_dir = Path(__file__).parent / "results" / "allergy_safety_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = output_dir / f"agenda_spec_v2_{timestamp}.json"

    # Remove raw_response for cleaner output
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "raw_response"}
        clean_results.append(clean)

    with open(results_file, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\n3. Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    print("\n## Verdict vs Evidence Consistency (per Agenda Spec)")
    print("-" * 70)

    consistent_count = 0
    for r in results:
        name = r.get("restaurant_name", "?")
        verdict = r.get("verdict", "ERROR")
        analysis = r.get("agenda_spec_analysis", {})

        if analysis:
            expected = analysis["verdict_expected_from_evidence"]
            consistent = analysis["consistent"]
            n_severe = analysis["recent_firsthand_severe_count"]
            n_any = analysis["recent_firsthand_any_count"]

            mark = "✓" if consistent else "✗"
            if consistent:
                consistent_count += 1

            print(f"{mark} {name[:35]:35} | verdict={verdict:12} | from_evidence={expected:12} | severe={n_severe}, any={n_any}")

    print(f"\nConsistency: {consistent_count}/{len(results)} = {100*consistent_count/len(results):.1f}%")

    print("\n## Detailed Evidence")
    print("-" * 70)

    for r in results:
        name = r.get("restaurant_name", "?")
        verdict = r.get("verdict", "?")
        evidences = r.get("evidences", [])

        if evidences:
            print(f"\n{name} → {verdict}")
            for e in evidences:
                fields = e.get("extracted_fields", {})
                print(f"   - {fields.get('date', '?'):10} | firsthand={fields.get('firsthand', '?'):5} | severity={fields.get('severity', '?'):8} | \"{e.get('quote', '')[:60]}...\"")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=int, default=200, help="Reviews per restaurant")
    args = parser.parse_args()

    run_test(max_reviews=args.reviews)
