#!/usr/bin/env python3
"""Flexible pipeline for selecting real Yelp data for knot vs cot evaluation."""

import json
import random
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "restaurant": {
        "min_reviews": 8,
        "max_reviews": 15,
        "min_star_variance": 2.0,
        "max_restaurants": 20,
    },
    "review": {
        "min_length": 50,
        "max_length": 500,
        "max_per_restaurant": 12,
    },
    "labeling": {
        "positive_threshold": 1.5,
        "keywords": {
            "speed": {
                "positive": ["fast", "quick", "quickly", "prompt", "no wait", "right away", "immediately"],
                "negative": ["slow", "wait", "waited", "waiting", "long time", "forever", "hour", "minutes"],
            },
            "consistency": {
                "positive": ["always", "every time", "consistent", "reliable", "never disappoints"],
                "negative": ["used to", "changed", "inconsistent", "hit or miss", "varies", "sometimes"],
            },
            "ambiance": {
                "positive": ["quiet", "romantic", "cozy", "intimate", "relaxed", "peaceful", "nice atmosphere"],
                "negative": ["loud", "noisy", "crowded", "chaotic", "cramped", "uncomfortable"],
            },
            "value": {
                "positive": ["worth", "good value", "reasonable", "affordable", "cheap", "great price", "deal"],
                "negative": ["expensive", "overpriced", "pricey", "not worth", "rip off", "too much"],
            },
            "food": {
                "positive": ["delicious", "tasty", "fresh", "amazing food", "great food", "excellent", "flavorful"],
                "negative": ["bland", "tasteless", "stale", "bad food", "terrible", "disgusting", "undercooked", "overcooked"],
            },
        },
    },
}


class RealDataSelector:
    """Flexible selector for creating evaluation datasets from Yelp data."""

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """Initialize with config dict or path to YAML/JSON config file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    if HAS_YAML:
                        self.config = yaml.safe_load(f)
                    else:
                        print("Warning: yaml not installed, using default config")
                        self.config = DEFAULT_CONFIG
                else:
                    self.config = json.load(f)
        else:
            self.config = config or DEFAULT_CONFIG

        self.yelp_dir = Path(__file__).parent / "yelp_raw"
        self.businesses = {}
        self.reviews_by_biz = defaultdict(list)

    def load_yelp_data(self, max_businesses: int = 1000):
        """Load Yelp business and review data."""
        print("Loading Yelp businesses...")
        biz_path = self.yelp_dir / "yelp_academic_dataset_business.json"
        with open(biz_path) as f:
            for line in f:
                b = json.loads(line)
                cats = b.get("categories", "") or ""
                if "Restaurant" in cats:
                    self.businesses[b["business_id"]] = b
                if len(self.businesses) >= max_businesses:
                    break

        print(f"Loaded {len(self.businesses)} restaurants")
        print("Loading reviews...")
        biz_ids = set(self.businesses.keys())
        review_path = self.yelp_dir / "yelp_academic_dataset_review.json"
        max_reviews = self.config["restaurant"]["max_reviews"]

        with open(review_path) as f:
            for i, line in enumerate(f):
                if i % 500000 == 0:
                    print(f"  Processed {i} reviews...")
                r = json.loads(line)
                bid = r["business_id"]
                if bid in biz_ids and len(self.reviews_by_biz[bid]) < max_reviews:
                    self.reviews_by_biz[bid].append(r)
                # Stop if all businesses have enough reviews
                if all(len(self.reviews_by_biz[b]) >= max_reviews for b in biz_ids):
                    break
                if i > 3000000:  # Safety limit
                    break

        print(f"Loaded reviews for {len(self.reviews_by_biz)} restaurants")

    def compute_star_variance(self, reviews: List[Dict]) -> float:
        """Compute variance in star ratings."""
        if not reviews:
            return 0
        stars = [r["stars"] for r in reviews]
        return max(stars) - min(stars)

    def has_aspect_mentions(self, reviews: List[Dict]) -> Dict[str, int]:
        """Count aspect-specific mentions in reviews."""
        counts = {aspect: 0 for aspect in self.config["labeling"]["keywords"]}
        for r in reviews:
            text = r["text"].lower()
            for aspect, keywords in self.config["labeling"]["keywords"].items():
                for kw in keywords["positive"] + keywords["negative"]:
                    if kw in text:
                        counts[aspect] += 1
                        break
        return counts

    def select_restaurants(self, n: Optional[int] = None) -> List[Dict]:
        """Select restaurants meeting criteria."""
        n = n or self.config["restaurant"]["max_restaurants"]
        min_reviews = self.config["restaurant"]["min_reviews"]
        min_variance = self.config["restaurant"]["min_star_variance"]

        candidates = []
        for bid, biz in self.businesses.items():
            reviews = self.reviews_by_biz.get(bid, [])
            if len(reviews) < min_reviews:
                continue
            variance = self.compute_star_variance(reviews)
            if variance < min_variance:
                continue
            aspect_counts = self.has_aspect_mentions(reviews)
            # Prefer restaurants with mentions across multiple aspects
            aspect_coverage = sum(1 for c in aspect_counts.values() if c > 0)
            candidates.append({
                "business": biz,
                "reviews": reviews,
                "variance": variance,
                "aspect_coverage": aspect_coverage,
            })

        # Sort by aspect coverage and variance
        candidates.sort(key=lambda x: (x["aspect_coverage"], x["variance"]), reverse=True)
        return candidates[:n]

    def select_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """Filter and select reviews for a restaurant."""
        min_len = self.config["review"]["min_length"]
        max_len = self.config["review"]["max_length"]
        max_count = self.config["review"]["max_per_restaurant"]

        # Filter by length
        filtered = [r for r in reviews if min_len <= len(r["text"]) <= max_len]

        # If not enough, relax length constraint
        if len(filtered) < 6:
            filtered = reviews

        # Ensure diversity in star ratings
        by_star = defaultdict(list)
        for r in filtered:
            by_star[int(r["stars"])].append(r)

        # Select mix of ratings
        selected = []
        for _ in range(max_count):
            if not by_star:
                break
            # Pick from least represented star rating
            star = min(by_star.keys(), key=lambda s: len([x for x in selected if int(x["stars"]) == s]))
            if by_star[star]:
                selected.append(by_star[star].pop(0))
                if not by_star[star]:
                    del by_star[star]

        return selected[:max_count]

    def compute_gold_labels(self, reviews: List[Dict]) -> Dict[str, int]:
        """Compute gold labels for all request types."""
        labels = {}
        threshold = self.config["labeling"]["positive_threshold"]

        for aspect, keywords in self.config["labeling"]["keywords"].items():
            pos_count = 0
            neg_count = 0
            for r in reviews:
                text = r["text"].lower()
                for kw in keywords["positive"]:
                    if kw in text:
                        pos_count += 1
                for kw in keywords["negative"]:
                    if kw in text:
                        neg_count += 1

            if pos_count > neg_count * threshold:
                labels[aspect] = 1
            elif neg_count > pos_count * threshold:
                labels[aspect] = -1
            else:
                labels[aspect] = 0

        # Map to request IDs
        aspect_to_request = {
            "speed": "R0",
            "consistency": "R1",
            "ambiance": "R2",
            "value": "R3",
            "food": "R4",
        }
        return {aspect_to_request[a]: v for a, v in labels.items()}

    def format_restaurant(self, candidate: Dict) -> Dict:
        """Format restaurant data for output."""
        biz = candidate["business"]
        reviews = self.select_reviews(candidate["reviews"])
        gold_labels = self.compute_gold_labels(reviews)

        return {
            "item_id": biz["business_id"][:12],
            "item_name": biz["name"],
            "city": biz.get("city", "Unknown"),
            "neighborhood": biz.get("neighborhood", "") or biz.get("city", "Unknown"),
            "price_range": "$" * int(biz.get("RestaurantsPriceRange2", 2) or 2),
            "cuisine": [c.strip() for c in (biz.get("categories", "") or "").split(",")[:3]],
            "stars": biz.get("stars", 3.0),
            "item_data": [
                {
                    "review_id": r["review_id"][:10],
                    "review": r["text"],
                    "stars": r["stars"],
                    "date": r["date"][:10],
                }
                for r in reviews
            ],
            "gold_labels": gold_labels,
        }

    def validate_selection(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate the selection meets quality criteria."""
        stats = {
            "count": len(data),
            "label_distribution": defaultdict(lambda: defaultdict(int)),
            "avg_reviews": 0,
            "issues": [],
        }

        total_reviews = 0
        for item in data:
            total_reviews += len(item["item_data"])
            for req_id, label in item["gold_labels"].items():
                stats["label_distribution"][req_id][label] += 1

        stats["avg_reviews"] = total_reviews / len(data) if data else 0

        # Check for issues
        for req_id, dist in stats["label_distribution"].items():
            if len(dist) < 2:
                stats["issues"].append(f"{req_id}: Only {list(dist.keys())} labels present")
            if any(c < 2 for c in dist.values()):
                stats["issues"].append(f"{req_id}: Imbalanced labels {dict(dist)}")

        return stats

    def generate(self, n: int = 20) -> List[Dict]:
        """Full pipeline: load data, select, format, validate."""
        if not self.businesses:
            self.load_yelp_data()

        candidates = self.select_restaurants(n)
        data = [self.format_restaurant(c) for c in candidates]
        stats = self.validate_selection(data)

        print(f"\nGenerated {stats['count']} restaurants")
        print(f"Avg reviews per restaurant: {stats['avg_reviews']:.1f}")
        print("Label distribution:")
        for req_id in sorted(stats["label_distribution"].keys()):
            dist = stats["label_distribution"][req_id]
            print(f"  {req_id}: {dict(dist)}")
        if stats["issues"]:
            print("Issues:")
            for issue in stats["issues"]:
                print(f"  - {issue}")

        return data

    def save(self, data: List[Dict], path: str):
        """Save data to JSONL file."""
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved to {path}")

    def adjust_for_ambiguity(self):
        """Readjust selection to increase ambiguity (Attempt 1)."""
        self.config["restaurant"]["min_star_variance"] = 3.0
        print("Adjusted: Increased min_star_variance to 3.0")

    def adjust_for_complexity(self):
        """Readjust selection to increase complexity (Attempt 2)."""
        self.config["restaurant"]["min_star_variance"] = 2.5
        # Add more nuanced keywords
        self.config["labeling"]["keywords"]["consistency"]["negative"].extend([
            "not as good", "declining", "went downhill"
        ])
        self.config["labeling"]["keywords"]["ambiance"]["negative"].extend([
            "too dark", "too bright", "cold", "hot"
        ])
        print("Adjusted: Added complexity keywords")

    def adjust_for_balance(self):
        """Readjust to balance label distribution (Attempt 3)."""
        self.config["labeling"]["positive_threshold"] = 1.3
        print("Adjusted: Lowered threshold to 1.3 for more neutral labels")


def main():
    """Generate real_data.jsonl from Yelp data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate real Yelp data for evaluation")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--n", type=int, default=20, help="Number of restaurants")
    parser.add_argument("--output", default="real_data.jsonl", help="Output file")
    parser.add_argument("--adjust", type=int, choices=[1, 2, 3], help="Apply adjustment strategy")
    args = parser.parse_args()

    selector = RealDataSelector(config_path=args.config)

    # Apply adjustments if specified
    if args.adjust == 1:
        selector.adjust_for_ambiguity()
    elif args.adjust == 2:
        selector.adjust_for_complexity()
    elif args.adjust == 3:
        selector.adjust_for_balance()

    data = selector.generate(args.n)
    output_path = Path(__file__).parent / args.output
    selector.save(data, output_path)


if __name__ == "__main__":
    main()
