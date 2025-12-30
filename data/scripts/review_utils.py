#!/usr/bin/env python3
"""Review processing utilities for Yelp data.

These functions bucket and format reviews by star rating for later use.
"""


def bucket_reviews_by_star(reviews: list) -> dict:
    """Bucket reviews by star rating (1-5), sorted by length within each bucket.

    Args:
        reviews: List of review dicts with 'stars' and 'text' fields

    Returns:
        Dict mapping star rating (1-5) to list of reviews, sorted by text length descending
    """
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for r in reviews:
        star = int(r.get("stars", 3))
        star = max(1, min(5, star))  # Clamp to 1-5
        buckets[star].append(r)

    # Sort each bucket by text length (longest first)
    for star in buckets:
        buckets[star].sort(key=lambda r: -len(r.get("text", "")))

    return buckets


def format_reviews_by_star(buckets: dict) -> dict:
    """Format bucketed reviews with metadata for output.

    Args:
        buckets: Dict from bucket_reviews_by_star()

    Returns:
        Dict with keys like "1_star", "2_star", etc., each containing list of formatted reviews
    """
    result = {}
    for star, bucket in buckets.items():
        result[f"{star}_star"] = [
            {
                "review_id": r.get("review_id"),
                "user_id": r.get("user_id"),
                "text": r.get("text"),
                "stars": r.get("stars"),
                "date": r.get("date"),
                "useful": r.get("useful", 0),
                "length": len(r.get("text", ""))
            }
            for r in bucket
        ]
    return result
