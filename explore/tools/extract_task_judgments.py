#!/usr/bin/env python3
"""
Extract Task Judgments - LLM-based semantic extraction for task-relevant reviews.

Only processes reviews that matched keywords in the index.
Resumable from checkpoint for long-running extraction.

Usage:
    python tools/extract_task_judgments.py --task G1 --limit 5   # Test
    python tools/extract_task_judgments.py --task G1 --all       # Full run
    python tools/extract_task_judgments.py --task G1 --resume    # Resume from checkpoint
    python tools/extract_task_judgments.py --task G1 --status    # Check progress
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_FILE = DATA_DIR / "dataset_K200.jsonl"
CONFIG_FILE = DATA_DIR / "semantic_gt" / "pipeline_config.json"
INDEX_DIR = DATA_DIR / "semantic_gt" / "review_index"

# Task-specific directories
def get_task_dir(task_id: str) -> Path:
    return DATA_DIR / "semantic_gt" / f"task_{task_id}"


# G1.1 Peanut Allergy Extraction Prompt
G1_EXTRACTION_PROMPT = '''Analyze this restaurant review for peanut/nut allergy safety signals.

REVIEW:
Date: {date}
Stars: {stars}
Useful votes: {useful}
Text: {text}

Extract these fields (respond with ONLY valid JSON, no other text):
{{
  "incident_severity": "none" | "mild" | "moderate" | "severe",
  "account_type": "none" | "firsthand" | "secondhand" | "hypothetical",
  "safety_interaction": "none" | "positive" | "negative" | "betrayal",
  "excerpt": "key quote if allergy-related, else empty string"
}}

DEFINITIONS:
- incident_severity:
  - none: No allergic reaction described in this review
  - mild: Minor symptoms mentioned (stomach upset, mild discomfort)
  - moderate: Visible symptoms (hives, swelling, needed antihistamine/Benadryl)
  - severe: Life-threatening (anaphylaxis, EpiPen used, ER visit, hospitalization)

- account_type (only if incident_severity != "none"):
  - none: No incident described
  - firsthand: Personal experience ("I had", "my child", "we experienced")
  - secondhand: Reported by others ("I heard", "friend told me", "reviews say")
  - hypothetical: Concern without actual incident ("I worry about", "not sure if safe")

- safety_interaction:
  - none: No interaction with staff about allergies mentioned
  - positive: Staff asked about allergies AND successfully accommodated
  - negative: Staff dismissive, careless, refused to accommodate, or seemed annoyed
  - betrayal: Staff CLAIMED it was safe BUT customer still had a reaction

RESPOND WITH JSON ONLY. No explanation.'''


def sanitize_filename(name: str) -> str:
    """Convert restaurant name to safe filename."""
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name).strip().replace(' ', '_')


def load_config():
    """Load pipeline configuration."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def load_source_data() -> List[Dict]:
    """Load all restaurants from source file."""
    restaurants = []
    with open(SOURCE_FILE, 'r') as f:
        for line in f:
            if line.strip():
                restaurants.append(json.loads(line))
    return restaurants


def load_review_index(restaurant_name: str) -> Optional[Dict]:
    """Load review index for a restaurant."""
    filename = sanitize_filename(restaurant_name) + ".json"
    index_file = INDEX_DIR / filename
    if not index_file.exists():
        return None
    with open(index_file, 'r') as f:
        return json.load(f)


def load_judgments(task_id: str) -> Dict[str, Dict]:
    """Load existing judgments for a task."""
    task_dir = get_task_dir(task_id)
    judgments_file = task_dir / "judgments.json"
    if not judgments_file.exists():
        return {}
    with open(judgments_file, 'r') as f:
        data = json.load(f)
    return data.get("judgments", {})


def save_judgments(task_id: str, judgments: Dict[str, Dict]):
    """Save judgments to file."""
    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    output = {
        "task_id": task_id,
        "extraction_prompt_version": config["tasks"][task_id]["extraction_prompt_version"],
        "extracted_at": datetime.now().isoformat(),
        "judgments": judgments
    }

    with open(task_dir / "judgments.json", 'w') as f:
        json.dump(output, f, indent=2)


def load_extraction_log(task_id: str) -> Dict:
    """Load extraction log for progress tracking."""
    task_dir = get_task_dir(task_id)
    log_file = task_dir / "extraction_log.json"
    if not log_file.exists():
        return {
            "task_id": task_id,
            "started": datetime.now().isoformat(),
            "completed_restaurants": [],
            "progress": {}
        }
    with open(log_file, 'r') as f:
        return json.load(f)


def save_extraction_log(task_id: str, log: Dict):
    """Save extraction log."""
    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    log["last_updated"] = datetime.now().isoformat()
    with open(task_dir / "extraction_log.json", 'w') as f:
        json.dump(log, f, indent=2)


def parse_json_response(response: str) -> Dict:
    """Parse JSON from LLM response, handling common issues."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in response
    match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Return default on failure
    return {
        "incident_severity": "none",
        "account_type": "none",
        "safety_interaction": "none",
        "excerpt": "",
        "_parse_error": True
    }


async def extract_single_judgment_g1(review: Dict, llm_func) -> Dict:
    """Extract G1.1 judgment for a single review."""
    prompt = G1_EXTRACTION_PROMPT.format(
        date=review.get("date", ""),
        stars=review.get("stars", 0),
        useful=review.get("useful", 0),
        text=review.get("text", "")
    )

    try:
        response = await llm_func(prompt)
        judgment = parse_json_response(response)
    except Exception as e:
        judgment = {
            "incident_severity": "none",
            "account_type": "none",
            "safety_interaction": "none",
            "excerpt": "",
            "_error": str(e)
        }

    return judgment


async def extract_restaurant_judgments(
    task_id: str,
    restaurant: Dict,
    index: Dict,
    llm_func
) -> Dict:
    """Extract judgments for all keyword-matched reviews in a restaurant."""
    name = restaurant["business"]["name"]
    categories = restaurant["business"]["categories"]
    reviews = restaurant.get("reviews", [])

    # Get keyword-matched review indices
    relevant_indices = [
        r["idx"] for r in index["reviews"]
        if r["task_relevance"].get(task_id, {}).get("keyword_match", False)
    ]

    if not relevant_indices:
        return {
            "restaurant_meta": {"categories": categories},
            "reviews": []
        }

    # Build list of (index_entry, review_data) pairs
    relevant_reviews = []
    for idx in relevant_indices:
        if idx < len(reviews):
            index_entry = next((r for r in index["reviews"] if r["idx"] == idx), None)
            if index_entry:
                relevant_reviews.append((index_entry, reviews[idx]))

    print(f"  Extracting {len(relevant_reviews)} reviews for {name}...")

    # Extract judgments
    if task_id == "G1":
        tasks = [extract_single_judgment_g1(r, llm_func) for _, r in relevant_reviews]
    else:
        raise ValueError(f"Unknown task: {task_id}")

    judgments = await asyncio.gather(*tasks)

    # Combine with metadata
    review_judgments = []
    for (idx_entry, _), judgment in zip(relevant_reviews, judgments):
        review_judgments.append({
            "idx": idx_entry["idx"],
            "date": idx_entry["date"],
            "stars": idx_entry["stars"],
            "useful": idx_entry["useful"],
            **judgment
        })

    return {
        "restaurant_meta": {"categories": categories},
        "reviews": review_judgments
    }


async def run_extraction(
    task_id: str,
    limit: int = None,
    resume: bool = True,
    llm_func = None
):
    """
    Run extraction for a task.

    Args:
        task_id: Task identifier (e.g., "G1")
        limit: Max restaurants to process (None = all)
        resume: If True, skip already-processed restaurants
        llm_func: Async function to call LLM
    """
    config = load_config()
    if task_id not in config["tasks"]:
        raise ValueError(f"Unknown task: {task_id}")

    restaurants = load_source_data()
    if limit:
        restaurants = restaurants[:limit]

    log = load_extraction_log(task_id) if resume else {
        "task_id": task_id,
        "started": datetime.now().isoformat(),
        "completed_restaurants": [],
        "progress": {}
    }

    all_judgments = load_judgments(task_id) if resume else {}
    completed = set(log.get("completed_restaurants", []))

    total_reviews = 0
    processed_restaurants = 0

    for i, restaurant in enumerate(restaurants):
        name = restaurant["business"]["name"]

        # Skip if already processed
        if resume and name in completed:
            continue

        # Load review index
        index = load_review_index(name)
        if not index:
            print(f"  Warning: No index for {name}, skipping")
            continue

        # Extract judgments
        result = await extract_restaurant_judgments(task_id, restaurant, index, llm_func)

        all_judgments[name] = result
        completed.add(name)
        total_reviews += len(result["reviews"])
        processed_restaurants += 1

        # Update log and save checkpoint
        log["completed_restaurants"] = list(completed)
        log["progress"] = {
            "restaurants_processed": len(completed),
            "restaurants_total": len(restaurants),
            "reviews_extracted": total_reviews
        }

        save_judgments(task_id, all_judgments)
        save_extraction_log(task_id, log)

        print(f"  [{len(completed)}/{len(restaurants)}] {name}: {len(result['reviews'])} reviews")

    return {
        "restaurants_processed": processed_restaurants,
        "reviews_extracted": total_reviews,
        "total_restaurants": len(all_judgments)
    }


def show_status(task_id: str):
    """Show extraction status for a task."""
    log = load_extraction_log(task_id)
    judgments = load_judgments(task_id)

    print(f"\n{'='*60}")
    print(f"Extraction Status: Task {task_id}")
    print(f"{'='*60}")
    print(f"Started: {log.get('started', 'N/A')}")
    print(f"Last Updated: {log.get('last_updated', 'N/A')}")
    print(f"\nProgress:")
    progress = log.get("progress", {})
    print(f"  Restaurants: {progress.get('restaurants_processed', 0)}/{progress.get('restaurants_total', '?')}")
    print(f"  Reviews Extracted: {progress.get('reviews_extracted', 0)}")
    print(f"\nJudgments stored: {len(judgments)} restaurants")

    # Count reviews
    total_reviews = sum(len(j.get("reviews", [])) for j in judgments.values())
    print(f"Total reviews judged: {total_reviews}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract Task Judgments")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., G1)")
    parser.add_argument("--limit", type=int, help="Limit number of restaurants")
    parser.add_argument("--all", action="store_true", help="Process all restaurants")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    if args.status:
        show_status(args.task)
        return

    if args.dry_run:
        config = load_config()
        restaurants = load_source_data()
        log = load_extraction_log(args.task)
        completed = set(log.get("completed_restaurants", []))

        print(f"\nDry run for Task {args.task}")
        print(f"Already completed: {len(completed)} restaurants")

        remaining = [r for r in restaurants if r["business"]["name"] not in completed]
        limit = args.limit or (len(remaining) if args.all else 5)
        to_process = remaining[:limit]

        total_reviews = 0
        for r in to_process:
            name = r["business"]["name"]
            index = load_review_index(name)
            if index:
                matches = sum(1 for rev in index["reviews"]
                            if rev["task_relevance"].get(args.task, {}).get("keyword_match", False))
                total_reviews += matches
                print(f"  {name}: {matches} keyword-matched reviews")

        print(f"\nWould process {len(to_process)} restaurants, ~{total_reviews} reviews")
        return

    # Import LLM function
    try:
        from utils.llm import call_llm_async
        llm_func = call_llm_async
    except ImportError:
        print("Error: Could not import utils.llm. Make sure you're in the right directory.")
        print("Run from: /Users/jimmyyeh/Documents/Station/anot")
        return

    limit = None if args.all else (args.limit or 5)

    print(f"\nExtracting judgments for Task {args.task}")
    print(f"Limit: {limit or 'all'}, Resume: {args.resume}")

    result = await run_extraction(
        task_id=args.task,
        limit=limit,
        resume=args.resume,
        llm_func=llm_func
    )

    print(f"\nComplete!")
    print(f"  Restaurants processed: {result['restaurants_processed']}")
    print(f"  Reviews extracted: {result['reviews_extracted']}")
    print(f"  Total restaurants: {result['total_restaurants']}")


if __name__ == "__main__":
    asyncio.run(main())
