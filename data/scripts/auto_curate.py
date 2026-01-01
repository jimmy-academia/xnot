#!/usr/bin/env python3
"""Automated Yelp data curation - non-interactive version of yelp_curation.py.

Usage:
    python data/scripts/auto_curate.py --city Philadelphia --category Bars --selection 2
    python data/scripts/auto_curate.py --city Philadelphia --category Restaurants --selection 3
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.llm import call_llm, call_llm_async

# File paths
YELP_DIR = Path("data/yelp")
RAW_DIR = YELP_DIR / "raw"
BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
METALOG_FILE = YELP_DIR / "meta_log.json"


class AutoCurator:
    """Automated curation for Yelp data."""

    def __init__(self, city: str, categories: List[str], selection_num: int,
                 target: int = 100, threshold: int = 70, batch_size: int = 20,
                 skip_llm: bool = False):
        self.city = city
        self.categories = categories
        self.selection_num = selection_num
        self.selection_id = f"selection_{selection_num}"
        self.output_file = YELP_DIR / f"{self.selection_id}.jsonl"

        self.target = target
        self.threshold = threshold
        self.batch_size = batch_size
        self.skip_llm = skip_llm

        self.businesses: Dict[str, dict] = {}
        self.reviews_by_biz: Dict[str, List[dict]] = defaultdict(list)
        self.category_keywords: List[str] = []
        self.selections: List[dict] = []

    def load_business_data(self) -> None:
        """Load all restaurant businesses from Yelp data."""
        print(f"Loading business data from {BUSINESS_FILE}...")
        with open(BUSINESS_FILE) as f:
            for line in f:
                biz = json.loads(line)
                cats = biz.get("categories", "") or ""
                # Keep businesses that are restaurants OR bars
                if "Restaurant" in cats or "Bars" in cats or "Nightlife" in cats:
                    self.businesses[biz["business_id"]] = biz
        print(f"Loaded {len(self.businesses)} restaurants/bars")

    def get_filtered_businesses(self) -> List[dict]:
        """Get businesses matching city and ANY of the categories."""
        results = []
        for biz in self.businesses.values():
            if biz.get("city") != self.city:
                continue
            cats = biz.get("categories", "") or ""
            if any(cat in cats for cat in self.categories):
                results.append(biz)
        return results

    def load_reviews_for_businesses(self, business_ids: set) -> None:
        """Load reviews only for specified businesses."""
        print(f"Loading reviews for {len(business_ids)} businesses...")
        self.reviews_by_biz.clear()
        count = 0
        with open(REVIEW_FILE) as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0 and i > 0:
                    print(f"  Processed {i:,} reviews...")
                review = json.loads(line)
                bid = review["business_id"]
                if bid in business_ids:
                    self.reviews_by_biz[bid].append(review)
                    count += 1
        print(f"Loaded {count:,} reviews")

    def compute_richness_scores(self) -> List[Tuple[dict, int]]:
        """Compute richness (total review char count) for filtered businesses."""
        scored = []
        for biz in self.get_filtered_businesses():
            bid = biz["business_id"]
            reviews = self.reviews_by_biz.get(bid, [])
            richness = sum(len(r.get("text", "")) for r in reviews)
            scored.append((biz, richness))
        return sorted(scored, key=lambda x: -x[1])

    def generate_category_keywords(self) -> List[str]:
        """Use LLM to generate keywords for the categories."""
        cats = ", ".join(self.categories)
        prompt = f"""For the restaurant/bar category "{cats}", list keywords that would appear in reviews if the business truly belongs to this category.

Think about:
- Specific dishes, drinks, cooking styles
- Cultural/regional terms
- Ambiance or service style typical of this type

Return ONLY a comma-separated list of 10-15 lowercase keywords."""

        try:
            response = call_llm(prompt, system="You are a cuisine expert.")
            keywords = [kw.strip().lower() for kw in response.split(",") if kw.strip()]
            keywords.extend([cat.lower() for cat in self.categories])
            return list(set(keywords))
        except Exception:
            return [cat.lower() for cat in self.categories]

    def get_keyword_evidence(self, biz: dict, max_snippets: int = 5) -> Tuple[List[str], int, int]:
        """Find review snippets containing any of the keywords."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total = len(reviews)
        matches = []

        for r in reviews:
            text = r.get("text", "")
            text_lower = text.lower()
            for kw in self.category_keywords:
                if kw in text_lower:
                    idx = text_lower.find(kw)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + 300)
                    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                    matches.append(snippet)
                    break

        return matches[:max_snippets], len(matches), total

    def parse_percentage(self, response: str) -> int:
        """Extract percentage number from LLM response."""
        match = re.search(r'(\d+)%', response)
        return int(match.group(1)) if match else 0

    async def estimate_category_fit_async(self, biz: dict) -> Tuple[dict, int, str]:
        """Async version for batch processing."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total_reviews = len(reviews)

        evidence_snippets, evidence_count, _ = self.get_keyword_evidence(biz, max_snippets=5)
        evidence_texts = "\n---\n".join(evidence_snippets) if evidence_snippets else "(None found)"

        first_5 = reviews[:5]
        remaining = reviews[5:]
        random_5 = random.sample(remaining, min(5, len(remaining))) if remaining else []
        sample_reviews = first_5 + random_5
        review_texts = "\n---\n".join([r.get("text", "")[:500] for r in sample_reviews])

        cats = ", ".join(self.categories)
        prompt = f"""Based on these reviews and evidence, estimate the probability (0-100%) that this business truly belongs to the category "{cats}".

Business: {biz.get('name')}
Listed categories: {biz.get('categories', 'Unknown')}

Keywords used for evidence: {', '.join(self.category_keywords[:10])}...
Keyword matches: {evidence_count} / {total_reviews} reviews contain category-related keywords

=== Evidence snippets ===
{evidence_texts}

=== Sample reviews ===
{review_texts}

Reply with just the percentage and one sentence explanation. Example: "85% - Reviews consistently mention bar drinks and nightlife atmosphere."
"""

        try:
            response = await call_llm_async(prompt, system="You are a data quality evaluator.")
            response = response.strip()
            pct = self.parse_percentage(response)
            return (biz, pct, response)
        except Exception as e:
            return (biz, 0, f"[Error: {e}]")

    def estimate_simple(self, biz: dict) -> Tuple[dict, int, str]:
        """Simple heuristic scoring without LLM."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total_reviews = len(reviews)

        if total_reviews == 0:
            return (biz, 0, "No reviews")

        # Count keyword matches
        _, match_count, _ = self.get_keyword_evidence(biz)
        match_ratio = match_count / total_reviews if total_reviews > 0 else 0

        # Stars bonus
        stars = biz.get("stars", 3)

        # Review count bonus
        review_bonus = min(20, total_reviews // 10)

        # Calculate score
        pct = int(match_ratio * 60 + stars * 5 + review_bonus)
        pct = min(100, max(0, pct))

        reason = f"{pct}% - {match_count}/{total_reviews} keyword matches, {stars}★, {total_reviews} reviews"
        return (biz, pct, reason)

    async def run_auto_mode(self) -> None:
        """Auto mode: batch estimate with early stopping."""
        scored = self.compute_richness_scores()
        print(f"\nAuto mode: Processing {len(scored)} businesses...")

        all_results = []
        above_threshold_count = 0

        for batch_start in range(0, len(scored), self.batch_size):
            batch = scored[batch_start:batch_start + self.batch_size]
            print(f"Processing batch {batch_start//self.batch_size + 1} ({batch_start+1}-{batch_start+len(batch)})...")

            if self.skip_llm:
                results = [self.estimate_simple(biz) for biz, _ in batch]
            else:
                tasks = [self.estimate_category_fit_async(biz) for biz, _ in batch]
                results = await asyncio.gather(*tasks)

            all_results.extend(results)

            above_threshold_count = sum(1 for _, pct, _ in all_results if pct >= self.threshold)

            if above_threshold_count >= self.target:
                print(f"Reached {self.target} businesses above {self.threshold}%. Stopping early.")
                break

        # Sort by percentage descending
        all_results.sort(key=lambda x: -x[1])

        # Save all results
        for biz, pct, exp in all_results:
            self.selections.append({
                "item_id": biz["business_id"],
                "llm_percent": pct,
                "llm_reasoning": exp
            })

        self.write_selections_file()
        self.update_metalog()

        print(f"\nSaved {len(all_results)} businesses.")
        print(f"Above {self.threshold}%: {above_threshold_count} | Below: {len(all_results) - above_threshold_count}")
        print(f"\nTop 10:")
        for i, (biz, pct, exp) in enumerate(all_results[:10], 1):
            print(f"  {i}. {biz['name'][:40]:<40} ({pct}%)")

    def write_selections_file(self) -> None:
        """Write all selections to file as JSONL."""
        with open(self.output_file, "w") as f:
            for sel in self.selections:
                f.write(json.dumps(sel) + "\n")
        print(f"Saved {len(self.selections)} selections to {self.output_file}")

    def load_metalog(self) -> dict:
        """Load the metalog file or return empty dict."""
        if METALOG_FILE.exists():
            with open(METALOG_FILE) as f:
                return json.load(f)
        return {"selections": {}}

    def update_metalog(self) -> None:
        """Update metalog with current selection metadata."""
        metalog = self.load_metalog()

        metalog["selections"][self.selection_id] = {
            "file": self.output_file.name,
            "created": datetime.now().isoformat(),
            "city": self.city,
            "categories": self.categories,
            "restaurant_count": len(self.selections),
        }

        json_str = json.dumps(metalog, indent=4)
        json_str = re.sub(
            r'\[\s*\n\s*"([^"]+)"(?:,\s*\n\s*"([^"]+)")*\s*\n\s*\]',
            lambda m: "[" + ", ".join(f'"{x}"' for x in re.findall(r'"([^"]+)"', m.group(0))) + "]",
            json_str
        )
        with open(METALOG_FILE, "w") as f:
            f.write(json_str)
        print(f"Updated {METALOG_FILE}")

    def run(self) -> None:
        """Main entry point."""
        print(f"=" * 60)
        print(f"Auto Curation: {self.city} > {', '.join(self.categories)}")
        print(f"Output: {self.output_file}")
        print(f"=" * 60)

        # Load business data
        self.load_business_data()

        # Get filtered businesses
        filtered = self.get_filtered_businesses()
        print(f"Found {len(filtered)} businesses matching criteria")

        if not filtered:
            print("No businesses found. Exiting.")
            return

        # Load reviews
        business_ids = {b["business_id"] for b in filtered}
        self.load_reviews_for_businesses(business_ids)

        # Generate category keywords
        if not self.skip_llm:
            print("Generating category keywords with LLM...")
            self.category_keywords = self.generate_category_keywords()
            print(f"Keywords: {', '.join(self.category_keywords[:15])}...")
        else:
            self.category_keywords = [cat.lower() for cat in self.categories]
            print(f"Using simple keywords: {self.category_keywords}")

        # Run auto mode
        asyncio.run(self.run_auto_mode())

        print(f"\n{'=' * 60}")
        print(f"Complete! Output: {self.output_file}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Automated Yelp data curation")
    parser.add_argument("--city", required=True, help="City name (e.g., Philadelphia)")
    parser.add_argument("--category", required=True, nargs="+", help="Category names (e.g., Bars Nightlife)")
    parser.add_argument("--selection", required=True, type=int, help="Selection number (e.g., 2)")
    parser.add_argument("--target", type=int, default=100, help="Target number of businesses above threshold")
    parser.add_argument("--threshold", type=int, default=70, help="Minimum percentage threshold")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for async processing")
    parser.add_argument("--skip-llm", action="store_true", help="Use simple heuristic instead of LLM")

    args = parser.parse_args()

    curator = AutoCurator(
        city=args.city,
        categories=args.category,
        selection_num=args.selection,
        target=args.target,
        threshold=args.threshold,
        batch_size=args.batch_size,
        skip_llm=args.skip_llm
    )
    curator.run()


if __name__ == "__main__":
    main()
