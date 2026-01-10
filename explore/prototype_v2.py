#!/usr/bin/env python3
"""
Prototype v2: Find LLM failure modes with real data + complex rules.

Key changes from v1:
- NO injection - pure real data
- Rules require COMPLETE data scanning (counting, aggregation)
- Ground truth computed programmatically
- Scale testing to find failure threshold
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm, configure

configure(max_tokens_reasoning=32000)

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Rule:
    """A rule that requires complete data processing."""
    name: str
    description: str
    compute_gt: Callable[[list[dict]], any]  # Ground truth function
    format_question: Callable[[any], str]  # Format question with GT
    check_answer: Callable[[str, any], tuple[bool, str]]  # (correct, explanation)


def load_reviews(filename: str, max_reviews: int = None) -> tuple[dict, list[dict]]:
    """Load restaurant data. If max_reviews set, takes most recent."""
    filepath = DATA_DIR / filename
    with open(filepath) as f:
        meta = json.loads(f.readline())
        all_reviews = [json.loads(line) for line in f]

    if max_reviews and max_reviews < len(all_reviews):
        # Take most recent
        reviews = all_reviews[-max_reviews:]
    else:
        reviews = all_reviews

    return meta, reviews


# =============================================================================
# RULE DEFINITIONS
# These rules require complete data scanning - you can't sample
# =============================================================================

def make_keyword_count_rule(keyword: str) -> Rule:
    """Rule: Count exact number of reviews mentioning a keyword."""

    def compute_gt(reviews):
        return sum(1 for r in reviews if keyword.lower() in r["text"].lower())

    def format_question(gt):
        return f"Exactly how many reviews mention the word '{keyword}'? Give only a number."

    def check_answer(response, gt):
        # Extract number from response
        numbers = re.findall(r'\b(\d+)\b', response)
        if not numbers:
            return False, f"No number found in response. Expected {gt}"
        predicted = int(numbers[0])
        if predicted == gt:
            return True, f"Correct: {gt}"
        else:
            return False, f"Predicted {predicted}, actual {gt} (off by {abs(predicted - gt)})"

    return Rule(
        name=f"count_{keyword}",
        description=f"Count reviews mentioning '{keyword}'",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


def make_percentage_rule(keyword: str) -> Rule:
    """Rule: What percentage of reviews mention a keyword?"""

    def compute_gt(reviews):
        count = sum(1 for r in reviews if keyword.lower() in r["text"].lower())
        return round(count / len(reviews) * 100, 1)

    def format_question(gt):
        return f"What percentage of reviews mention '{keyword}'? Give a number (e.g., '23.5')."

    def check_answer(response, gt):
        numbers = re.findall(r'(\d+\.?\d*)', response)
        if not numbers:
            return False, f"No number found. Expected {gt}%"
        predicted = float(numbers[0])
        tolerance = 2.0  # Allow 2% error
        if abs(predicted - gt) <= tolerance:
            return True, f"Correct within tolerance: {predicted}% vs {gt}%"
        else:
            return False, f"Predicted {predicted}%, actual {gt}% (off by {abs(predicted - gt):.1f}%)"

    return Rule(
        name=f"pct_{keyword}",
        description=f"Percentage of reviews mentioning '{keyword}'",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


def make_filtered_sentiment_rule(keyword: str) -> Rule:
    """Rule: Among reviews mentioning X, what % are positive (4+ stars)?"""

    def compute_gt(reviews):
        filtered = [r for r in reviews if keyword.lower() in r["text"].lower()]
        if not filtered:
            return None
        positive = sum(1 for r in filtered if r["stars"] >= 4)
        return {
            "total": len(filtered),
            "positive": positive,
            "percentage": round(positive / len(filtered) * 100, 1)
        }

    def format_question(gt):
        if gt is None:
            return None
        return (
            f"Among reviews that mention '{keyword}', what percentage are positive (4+ stars)? "
            f"First state how many reviews mention '{keyword}', then give the percentage."
        )

    def check_answer(response, gt):
        if gt is None:
            return None, "No reviews match filter"

        numbers = re.findall(r'(\d+\.?\d*)', response)
        if len(numbers) < 2:
            return False, f"Expected count and percentage. Actual: {gt['total']} reviews, {gt['percentage']}% positive"

        # Try to find the percentage (likely the number with decimal or the larger one if it's a %)
        predicted_pct = None
        for n in numbers:
            val = float(n)
            if 0 <= val <= 100 and ('.' in n or val > gt['total']):
                predicted_pct = val
                break

        if predicted_pct is None:
            predicted_pct = float(numbers[-1])  # Last number as fallback

        tolerance = 5.0
        if abs(predicted_pct - gt['percentage']) <= tolerance:
            return True, f"Correct within tolerance: {predicted_pct}% vs {gt['percentage']}%"
        else:
            return False, f"Predicted {predicted_pct}%, actual {gt['percentage']}% (off by {abs(predicted_pct - gt['percentage']):.1f}%)"

    return Rule(
        name=f"sentiment_{keyword}",
        description=f"Sentiment of reviews mentioning '{keyword}'",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


def make_temporal_comparison_rule() -> Rule:
    """Rule: Compare average rating of first half vs second half of reviews (by date)."""

    def compute_gt(reviews):
        # Reviews should already be sorted by date
        mid = len(reviews) // 2
        first_half = reviews[:mid]
        second_half = reviews[mid:]

        avg_first = sum(r["stars"] for r in first_half) / len(first_half)
        avg_second = sum(r["stars"] for r in second_half) / len(second_half)

        return {
            "first_half_avg": round(avg_first, 2),
            "second_half_avg": round(avg_second, 2),
            "difference": round(avg_second - avg_first, 2),
            "improved": avg_second > avg_first + 0.1,
            "declined": avg_second < avg_first - 0.1,
        }

    def format_question(gt):
        return (
            "Compare the average star rating of the FIRST half of reviews (older) vs the SECOND half (newer). "
            "Did quality improve, decline, or stay stable? "
            "Give the two averages and your conclusion (IMPROVED/DECLINED/STABLE)."
        )

    def check_answer(response, gt):
        response_upper = response.upper()

        if gt["improved"]:
            expected = "IMPROVED"
            correct = "IMPROVED" in response_upper or "IMPROV" in response_upper
        elif gt["declined"]:
            expected = "DECLINED"
            correct = "DECLINED" in response_upper or "DECLIN" in response_upper or "WORSE" in response_upper
        else:
            expected = "STABLE"
            correct = "STABLE" in response_upper or "SAME" in response_upper or "CONSISTENT" in response_upper

        detail = f"First half: {gt['first_half_avg']}, Second half: {gt['second_half_avg']}, Diff: {gt['difference']}"
        if correct:
            return True, f"Correct: {expected}. {detail}"
        else:
            return False, f"Expected {expected}. {detail}"

    return Rule(
        name="temporal_trend",
        description="Compare first half vs second half average ratings",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


def make_rare_event_rule(keywords: list[str]) -> Rule:
    """Rule: Find if ANY review mentions rare concerning keywords."""

    def compute_gt(reviews):
        matches = []
        for i, r in enumerate(reviews):
            text = r["text"].lower()
            for kw in keywords:
                if kw.lower() in text:
                    matches.append({
                        "review_idx": i + 1,
                        "keyword": kw,
                        "date": r["date"][:10],
                        "stars": r["stars"],
                        "snippet": r["text"][:100]
                    })
                    break  # One match per review is enough
        return {
            "found": len(matches) > 0,
            "count": len(matches),
            "matches": matches[:5]  # First 5 for reference
        }

    def format_question(gt):
        kw_list = ", ".join(f"'{k}'" for k in keywords)
        return (
            f"Do ANY reviews mention concerning food safety terms: {kw_list}? "
            f"Answer YES or NO, and if YES, state how many reviews mention these terms."
        )

    def check_answer(response, gt):
        response_upper = response.upper()
        predicted_yes = "YES" in response_upper[:50]
        actual_yes = gt["found"]

        if predicted_yes == actual_yes:
            if actual_yes:
                return True, f"Correct: YES, found {gt['count']} matches"
            else:
                return True, "Correct: NO matches found"
        else:
            if actual_yes:
                return False, f"Said NO but there are {gt['count']} matches"
            else:
                return False, "Said YES but there are no matches"

    return Rule(
        name="rare_safety",
        description="Find any concerning food safety mentions",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


def make_multi_keyword_count_rule(keywords: list[str]) -> Rule:
    """Rule: Count reviews mentioning ANY of multiple keywords."""

    def compute_gt(reviews):
        counts = {k: 0 for k in keywords}
        total_matching = 0
        for r in reviews:
            text = r["text"].lower()
            matched = False
            for k in keywords:
                if k.lower() in text:
                    counts[k] += 1
                    matched = True
            if matched:
                total_matching += 1
        return {
            "per_keyword": counts,
            "total_matching": total_matching,
        }

    def format_question(gt):
        kw_list = ", ".join(f"'{k}'" for k in keywords)
        return (
            f"Count how many reviews mention each of these words: {kw_list}. "
            f"Also count total reviews mentioning ANY of these words. "
            f"Format: word1: N, word2: N, ..., TOTAL: N"
        )

    def check_answer(response, gt):
        # This is hard to parse, just check total
        numbers = re.findall(r'total[:\s]+(\d+)', response.lower())
        if not numbers:
            numbers = re.findall(r'(\d+)', response)

        if not numbers:
            return False, f"No numbers found. Expected total: {gt['total_matching']}"

        # Take the last number as likely the total
        predicted = int(numbers[-1])
        actual = gt['total_matching']
        tolerance = max(2, actual * 0.1)  # 10% or 2, whichever is larger

        if abs(predicted - actual) <= tolerance:
            return True, f"Close enough: {predicted} vs {actual}"
        else:
            return False, f"Predicted {predicted}, actual {actual} (off by {abs(predicted - actual)})"

    return Rule(
        name="multi_keyword",
        description=f"Count reviews mentioning any of: {keywords}",
        compute_gt=compute_gt,
        format_question=format_question,
        check_answer=check_answer,
    )


# =============================================================================
# TEST HARNESS
# =============================================================================

def build_prompt(meta: dict, reviews: list[dict], question: str) -> str:
    """Build prompt with all reviews."""

    reviews_text = []
    for i, r in enumerate(reviews, 1):
        reviews_text.append(
            f"[R{i}] {r['date'][:10]} | {r['stars']}★ | {r['text']}"
        )

    return f"""You are analyzing restaurant reviews. Answer the question precisely.

RESTAURANT: {meta['name']}
TOTAL REVIEWS: {len(reviews)}

REVIEWS:
{"".join(chr(10) + r for r in reviews_text)}

QUESTION: {question}

Be precise. Count carefully. Give exact numbers when asked."""


def run_rule_test(
    filename: str,
    rule: Rule,
    max_reviews: int = None,
    verbose: bool = True
) -> dict:
    """Run a single rule test."""

    meta, reviews = load_reviews(filename, max_reviews)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rule: {rule.name}")
        print(f"Reviews: {len(reviews)}")
        print(f"Date range: {reviews[0]['date'][:10]} to {reviews[-1]['date'][:10]}")

    # Compute ground truth
    gt = rule.compute_gt(reviews)
    question = rule.format_question(gt)

    if question is None:
        if verbose:
            print("SKIP: Rule not applicable to this data")
        return {"status": "skip", "rule": rule.name}

    if verbose:
        print(f"Question: {question}")
        print(f"Ground truth: {gt}")

    # Build prompt
    prompt = build_prompt(meta, reviews, question)

    if verbose:
        print(f"Prompt length: {len(prompt):,} chars")

    # Call LLM
    if verbose:
        print("Calling LLM...")

    response = call_llm(prompt)

    if verbose:
        print(f"Response: {response[:500]}...")

    # Check answer
    correct, explanation = rule.check_answer(response, gt)

    if verbose:
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"{status}: {explanation}")

    return {
        "status": "pass" if correct else "fail",
        "rule": rule.name,
        "reviews": len(reviews),
        "ground_truth": gt,
        "response": response,
        "explanation": explanation,
    }


def run_scale_test(filename: str, rule: Rule, scales: list[int]) -> list[dict]:
    """Test a rule at multiple scales to find failure threshold."""

    print(f"\n{'#'*60}")
    print(f"SCALE TEST: {rule.name}")
    print(f"Scales: {scales}")
    print('#'*60)

    results = []
    for n in scales:
        result = run_rule_test(filename, rule, max_reviews=n, verbose=True)
        result["scale"] = n
        results.append(result)

    # Summary
    print(f"\n--- Scale Test Summary: {rule.name} ---")
    for r in results:
        status = "PASS" if r["status"] == "pass" else "FAIL" if r["status"] == "fail" else "SKIP"
        print(f"  {r['scale']:5d} reviews: {status}")

    return results


def main():
    """Run comprehensive failure-finding tests."""

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl")
    parser.add_argument("--scale-test", action="store_true", help="Run scale testing")
    parser.add_argument("--max-reviews", type=int, default=500)
    args = parser.parse_args()

    # Define rules to test
    rules = [
        make_keyword_count_rule("wait"),
        make_keyword_count_rule("oyster"),
        make_percentage_rule("service"),
        make_filtered_sentiment_rule("service"),
        make_temporal_comparison_rule(),
        make_rare_event_rule(["food poisoning", "sick", "ill", "vomit", "hospital"]),
        make_multi_keyword_count_rule(["wait", "line", "busy", "crowded"]),
    ]

    if args.scale_test:
        # Test at increasing scales
        scales = [50, 100, 200, 500, 1000, 2000]
        for rule in rules[:3]:  # Test first 3 rules for scale
            run_scale_test(args.file, rule, scales)
    else:
        # Single scale test
        print(f"Testing {len(rules)} rules at {args.max_reviews} reviews")
        results = []
        for rule in rules:
            result = run_rule_test(args.file, rule, args.max_reviews)
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        skipped = sum(1 for r in results if r["status"] == "skip")
        print(f"Passed: {passed}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        print(f"Skipped: {skipped}/{len(results)}")

        for r in results:
            status = {"pass": "✓", "fail": "✗", "skip": "-"}[r["status"]]
            print(f"  {status} {r['rule']}: {r.get('explanation', 'N/A')[:60]}")


if __name__ == "__main__":
    main()
