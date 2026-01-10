#!/usr/bin/env python3
"""
LLM-based review annotation for Task v6.

Labels each review with semantic categories:
- wait_complaint: Expresses frustration about wait time
- service_complaint: Criticizes staff behavior or service quality
- food_quality_complaint: Criticizes food taste, temperature, or preparation
- price_complaint: Says it's too expensive / not worth the price
- would_return: Explicitly states they would come back
- would_not_return: Explicitly states they would NOT return
- mentions_health_issue: Mentions illness, food poisoning, health department
- overall_positive: Overall sentiment is positive
- overall_negative: Overall sentiment is negative
"""

import json
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm import call_llm_async, configure

DATA_DIR = Path(__file__).parent / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"

# Labels to extract
LABELS = [
    "wait_complaint",
    "service_complaint",
    "food_quality_complaint",
    "price_complaint",
    "would_return",
    "would_not_return",
    "mentions_health_issue",
    "overall_positive",
    "overall_negative",
]


@dataclass
class LabeledReview:
    """A review with semantic labels."""
    review_id: int
    date: str
    stars: float
    text: str
    labels: dict
    confidence: dict  # Confidence for each label (high/medium/low)


def build_annotation_prompt(review_id: int, stars: float, date: str, text: str) -> str:
    """Build prompt for annotating a single review."""
    # Truncate very long reviews
    truncated = text[:2000] + "..." if len(text) > 2000 else text

    return f'''Analyze this restaurant review and label it. Output JSON only.

REVIEW (R{review_id}, {stars} stars, {date}):
"{truncated}"

LABELS (true/false for each):
- wait_complaint: Frustration about wait time, long lines, slow service
- service_complaint: Criticism of staff behavior or attitude (not speed)
- food_quality_complaint: Criticism of food taste, temperature, preparation
- price_complaint: Says too expensive or not worth the price
- would_return: Explicitly says they will come back or recommend
- would_not_return: Explicitly says they won't return or don't recommend
- mentions_health_issue: Mentions illness, food poisoning, health department
- overall_positive: Overall sentiment is positive
- overall_negative: Overall sentiment is negative

Output ONLY this JSON (no explanation):
{{"wait_complaint":false,"service_complaint":false,"food_quality_complaint":false,"price_complaint":false,"would_return":false,"would_not_return":false,"mentions_health_issue":false,"overall_positive":false,"overall_negative":false}}'''


def parse_annotation_response(response: str) -> tuple:
    """Parse LLM response into labels and confidence dicts."""
    import re

    # Look for JSON block
    json_match = re.search(r'\{[^{}]*\}', response)
    if not json_match:
        return None, None

    try:
        labels = json.loads(json_match.group())

        # Validate and normalize labels
        for label in LABELS:
            if label not in labels:
                labels[label] = False
            else:
                # Ensure boolean
                labels[label] = bool(labels[label])

        # Default confidence to "high" for successful parse
        confidence = {label: "high" for label in LABELS}

        return labels, confidence
    except json.JSONDecodeError:
        return None, None


async def annotate_review(review_id: int, review: dict, max_retries: int = 3) -> Optional[LabeledReview]:
    """Annotate a single review using LLM with retry."""
    prompt = build_annotation_prompt(
        review_id=review_id,
        stars=review['stars'],
        date=review['date'][:10],
        text=review['text']
    )

    for attempt in range(max_retries):
        try:
            response = await call_llm_async(prompt)

            # Empty response = retry
            if not response or len(response.strip()) < 10:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                print(f"  R{review_id}: Empty response after {max_retries} attempts")
                return None

            labels, confidence = parse_annotation_response(response)

            if labels is None:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    continue
                print(f"  R{review_id}: Failed to parse response")
                return None

            return LabeledReview(
                review_id=review_id,
                date=review['date'],
                stars=review['stars'],
                text=review['text'],
                labels=labels,
                confidence=confidence
            )
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            print(f"  R{review_id}: Error - {e}")
            return None

    return None


async def annotate_reviews(reviews: list, max_concurrent: int = 10) -> list:
    """Annotate multiple reviews concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def annotate_with_semaphore(review_id, review):
        async with semaphore:
            return await annotate_review(review_id, review)

    tasks = [
        annotate_with_semaphore(i, r)
        for i, r in enumerate(reviews, 1)
    ]

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def load_reviews(filename: str, max_reviews: int = None) -> tuple:
    """Load restaurant data."""
    filepath = DATA_DIR / filename
    with open(filepath) as f:
        meta = json.loads(f.readline())
        all_reviews = [json.loads(line) for line in f]

    if max_reviews and max_reviews < len(all_reviews):
        reviews = all_reviews[-max_reviews:]  # Most recent
    else:
        reviews = all_reviews

    return meta, reviews


def save_annotations(labeled_reviews: list, output_path: Path):
    """Save labeled reviews to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for lr in labeled_reviews:
            f.write(json.dumps(asdict(lr)) + '\n')


def print_summary(labeled_reviews: list):
    """Print summary statistics of annotations."""
    print(f"\n{'='*60}")
    print("ANNOTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotated: {len(labeled_reviews)}")

    # Count labels
    label_counts = {label: 0 for label in LABELS}
    low_confidence = {label: 0 for label in LABELS}

    for lr in labeled_reviews:
        for label in LABELS:
            if lr.labels.get(label, False):
                label_counts[label] += 1
            if lr.confidence.get(label) == "low":
                low_confidence[label] += 1

    print(f"\n{'Label':<25} {'Count':<8} {'Low Conf'}")
    print("-" * 45)
    for label in LABELS:
        print(f"{label:<25} {label_counts[label]:<8} {low_confidence[label]}")

    # Show reviews needing verification (many low confidence labels)
    needs_review = []
    for lr in labeled_reviews:
        low_count = sum(1 for c in lr.confidence.values() if c == "low")
        if low_count >= 3:
            needs_review.append((lr.review_id, low_count))

    if needs_review:
        print(f"\n--- Reviews Needing Verification ({len(needs_review)}) ---")
        for rid, lc in needs_review[:10]:
            print(f"  R{rid}: {lc} low-confidence labels")
        if len(needs_review) > 10:
            print(f"  ... and {len(needs_review) - 10} more")


async def main(filename: str, max_reviews: int, output_name: str = None):
    """Main annotation pipeline."""
    print(f"{'='*60}")
    print("REVIEW ANNOTATION")
    print(f"{'='*60}")

    # Load reviews
    meta, reviews = load_reviews(filename, max_reviews)
    print(f"Restaurant: {meta['name']}")
    print(f"Reviews to annotate: {len(reviews)}")

    # Annotate
    print(f"\nAnnotating reviews...")
    labeled_reviews = await annotate_reviews(reviews)

    # Save
    if output_name is None:
        base = filename.replace('.jsonl', '')
        output_name = f"{base}_labeled.jsonl"

    output_path = ANNOTATED_DIR / output_name
    save_annotations(labeled_reviews, output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    print_summary(labeled_reviews)

    return labeled_reviews


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate reviews with semantic labels")
    parser.add_argument("--file", default="Acme_Oyster_House__ab50qdW.jsonl")
    parser.add_argument("--max-reviews", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    configure(temperature=0.0)  # Deterministic labeling
    asyncio.run(main(args.file, args.max_reviews, args.output))
