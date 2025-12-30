#!/usr/bin/env python3
"""Human-in-the-Loop Yelp data curation tool for benchmark dataset creation."""

import asyncio
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from utils.llm import call_llm, call_llm_async


# File paths
YELP_DIR = Path("data/yelp")
RAW_DIR = YELP_DIR / "raw"
BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
OUTPUT_DIR = YELP_DIR
METALOG_FILE = YELP_DIR / "meta_log.json"


class YelpCurator:
    """Human-in-the-loop curation tool for Yelp data."""

    def __init__(self):
        self.console = Console()
        self.businesses: Dict[str, dict] = {}
        self.reviews_by_biz: Dict[str, List[dict]] = defaultdict(list)
        self.selected_city: Optional[str] = None
        self.selected_categories: List[str] = []  # Multi-category support
        self.category_keywords: List[str] = []  # LLM-generated keywords for category
        self.selections: List[dict] = []
        self.output_file: Optional[Path] = None
        self.selection_id: Optional[str] = None
        
        self.target = 100
        self.threshold = 70
        self.batch_size = 20


    def get_next_selection_path(self) -> Tuple[Path, str]:
        """Determine the next selection file path (selection_1, selection_2, etc.)."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Find existing selection files
        existing = list(OUTPUT_DIR.glob("selection_*.jsonl"))
        if not existing:
            selection_id = "selection_1"
        else:
            # Extract numbers and find max
            nums = []
            for f in existing:
                try:
                    num = int(f.stem.split("_")[1])
                    nums.append(num)
                except (IndexError, ValueError):
                    pass
            next_num = max(nums) + 1 if nums else 1
            selection_id = f"selection_{next_num}"

        return OUTPUT_DIR / f"{selection_id}.jsonl", selection_id

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
            "city": self.selected_city,
            "categories": self.selected_categories,
            "restaurant_count": len(self.selections),
        }

        # Write with indent, then compact short arrays to one line
        json_str = json.dumps(metalog, indent=4)
        # Compact arrays that span multiple lines (e.g., categories)
        json_str = re.sub(
            r'\[\s*\n\s*"([^"]+)"(?:,\s*\n\s*"([^"]+)")*\s*\n\s*\]',
            lambda m: "[" + ", ".join(f'"{x}"' for x in re.findall(r'"([^"]+)"', m.group(0))) + "]",
            json_str
        )
        with open(METALOG_FILE, "w") as f:
            f.write(json_str)

    def choose_output_mode(self) -> bool:
        """Ask user to create new or replace existing selection. Returns True to continue."""
        metalog = self.load_metalog()
        existing = list(metalog.get("selections", {}).keys())

        if not existing:
            # No existing selections, create new
            self.output_file, self.selection_id = self.get_next_selection_path()
            self.console.print(f"\n[bold]No existing selections found.[/bold]")
            self.console.print(f"[cyan]Will create: {self.output_file}[/cyan]")
            return True

        sorted_existing = sorted(existing)
        self.console.print("\n[bold]Existing selections:[/bold]")
        for i, sel_id in enumerate(sorted_existing, 1):
            info = metalog["selections"][sel_id]
            cats = ", ".join(info.get("categories", [])[:2])
            self.console.print(f"  {i}. {sel_id}: {info.get('city')} > {cats} ({info.get('restaurant_count', 0)} restaurants)")

        choice = Prompt.ask(f"\n[N]ew / [1-{len(sorted_existing)}] to replace", default="n").strip().lower()

        if choice == "n":
            self.output_file, self.selection_id = self.get_next_selection_path()
            self.console.print(f"[cyan]Will create: {self.output_file}[/cyan]")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_existing):
                sel_id = sorted_existing[idx]
                self.selection_id = sel_id
                self.output_file = OUTPUT_DIR / f"{sel_id}.jsonl"
                # Clear existing file
                self.output_file.unlink(missing_ok=True)
                self.console.print(f"[yellow]Will replace: {self.output_file}[/yellow]")
            else:
                self.console.print("[red]Invalid selection number[/red]")
                return False
        else:
            self.console.print("[red]Invalid input[/red]")
            return False
        return True

    def load_business_data(self) -> None:
        """Load all restaurant businesses from Yelp data."""
        if not BUSINESS_FILE.exists():
            self.console.print(f"[red]Error: Business file not found: {BUSINESS_FILE}[/red]")
            sys.exit(1)

        with self.console.status("[bold green]Loading business data..."):
            with open(BUSINESS_FILE) as f:
                for line in f:
                    biz = json.loads(line)
                    cats = biz.get("categories", "") or ""
                    if "Restaurant" in cats:
                        self.businesses[biz["business_id"]] = biz

        self.console.print(f"[green]Loaded {len(self.businesses)} restaurants[/green]")

    def load_reviews_for_businesses(self, business_ids: set) -> None:
        """Load reviews only for specified businesses (memory optimization)."""
        if not REVIEW_FILE.exists():
            self.console.print(f"[red]Error: Review file not found: {REVIEW_FILE}[/red]")
            sys.exit(1)

        self.reviews_by_biz.clear()
        with self.console.status("[bold green]Loading reviews for selected businesses..."):
            with open(REVIEW_FILE) as f:
                for i, line in enumerate(f):
                    if i % 500000 == 0 and i > 0:
                        self.console.print(f"  [dim]Processed {i:,} reviews...[/dim]")
                    review = json.loads(line)
                    bid = review["business_id"]
                    if bid in business_ids:
                        self.reviews_by_biz[bid].append(review)

        total_reviews = sum(len(r) for r in self.reviews_by_biz.values())
        self.console.print(f"[green]Loaded {total_reviews:,} reviews for {len(self.reviews_by_biz)} restaurants[/green]")

    def get_city_counts(self) -> Dict[str, int]:
        """Count restaurants per city."""
        counts = defaultdict(int)
        for biz in self.businesses.values():
            city = biz.get("city", "Unknown")
            counts[city] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_category_counts(self, city: str) -> Dict[str, int]:
        """Count categories within a city."""
        counts = defaultdict(int)
        for biz in self.businesses.values():
            if biz.get("city") != city:
                continue
            cats = biz.get("categories", "") or ""
            for cat in cats.split(", "):
                cat = cat.strip()
                if cat and cat != "Restaurants":
                    counts[cat] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_filtered_businesses(self) -> List[dict]:
        """Get businesses matching selected city and ANY of the selected categories."""
        results = []
        for biz in self.businesses.values():
            if biz.get("city") != self.selected_city:
                continue
            cats = biz.get("categories", "") or ""
            # Match ANY of selected categories
            if any(cat in cats for cat in self.selected_categories):
                results.append(biz)
        return results

    def compute_richness_scores(self) -> List[tuple]:
        """Compute richness (total review char count) for filtered businesses.
        Returns: [(biz, richness_score), ...] sorted descending by score.
        """
        scored = []
        for biz in self.get_filtered_businesses():
            bid = biz["business_id"]
            reviews = self.reviews_by_biz.get(bid, [])
            richness = sum(len(r.get("text", "")) for r in reviews)
            scored.append((biz, richness))
        return sorted(scored, key=lambda x: -x[1])

    def generate_category_keywords(self, categories: List[str]) -> List[str]:
        """Use LLM to generate keywords that reveal if restaurant belongs to category."""
        cats = ", ".join(categories)
        prompt = f"""For the restaurant category "{cats}", list keywords that would appear in reviews if the restaurant truly belongs to this category.

Think about:
- Specific dishes, ingredients, cooking styles
- Cultural/regional terms
- Ambiance or service style typical of this cuisine

Return ONLY a comma-separated list of 10-15 lowercase keywords. Example for "Italian": pasta, pizza, marinara, alfredo, tiramisu, espresso, vino, authentic italian, nonna, trattoria"""

        try:
            response = call_llm(prompt, system="You are a cuisine expert.")
            # Parse comma-separated keywords
            keywords = [kw.strip().lower() for kw in response.split(",") if kw.strip()]
            # Add original category names
            keywords.extend([cat.lower() for cat in categories])
            return list(set(keywords))
        except Exception:
            return [cat.lower() for cat in categories]

    def get_keyword_evidence(self, biz: dict, keywords: List[str], max_snippets: int = 5) -> Tuple[List[str], int, int]:
        """Find review snippets containing any of the keywords.

        Returns: (snippets, match_count, total_reviews)
        """
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total = len(reviews)
        matches = []

        for r in reviews:
            text = r.get("text", "")
            text_lower = text.lower()
            for kw in keywords:
                if kw in text_lower:
                    # Extract snippet around keyword
                    idx = text_lower.find(kw)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + 300)
                    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                    matches.append(snippet)
                    break

        return matches[:max_snippets], len(matches), total

    def estimate_category_fit(self, biz: dict) -> str:
        """Use LLM to estimate how well restaurant fits selected categories."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total_reviews = len(reviews)

        # Use pre-generated category keywords
        keywords = self.category_keywords

        # Step 1: Get evidence snippets using keywords
        evidence_snippets, evidence_count, _ = self.get_keyword_evidence(biz, keywords, max_snippets=5)
        evidence_texts = "\n---\n".join(evidence_snippets) if evidence_snippets else "(None found)"

        # Step 2: First 5 reviews + random 5 other reviews
        first_5 = reviews[:5]
        remaining = reviews[5:]
        random_5 = random.sample(remaining, min(5, len(remaining))) if remaining else []
        sample_reviews = first_5 + random_5
        review_texts = "\n---\n".join([r.get("text", "")[:500] for r in sample_reviews])

        cats = ", ".join(self.selected_categories)
        prompt = f"""Based on these reviews and evidence, estimate the probability (0-100%) that this restaurant truly belongs to the category "{cats}".

Restaurant: {biz.get('name')}
Listed categories: {biz.get('categories', 'Unknown')}

Keywords used for evidence: {', '.join(keywords[:10])}...
Keyword matches: {evidence_count} / {total_reviews} reviews contain category-related keywords

=== Evidence snippets (reviews mentioning keywords) ===
{evidence_texts}

=== Sample reviews (first 5 + random 5) ===
{review_texts}

Consider both the keyword match ratio and the content of reviews.
Reply with just the percentage and one sentence explanation. Example: "85% - Reviews consistently mention authentic Italian dishes and pasta."
"""

        try:
            with self.console.status("[bold blue]LLM estimating category fit..."):
                response = call_llm(prompt, system="You are a data quality evaluator.")
            return response.strip()
        except Exception as e:
            return f"[Error: {e}]"

    def parse_percentage(self, response: str) -> int:
        """Extract percentage number from LLM response like '85% - explanation'."""
        match = re.search(r'(\d+)%', response)
        return int(match.group(1)) if match else 0

    async def estimate_category_fit_async(self, biz: dict) -> Tuple[dict, int, str]:
        """Async version for batch processing. Returns (biz, percentage, explanation)."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        total_reviews = len(reviews)

        # Use pre-generated category keywords
        keywords = self.category_keywords

        # Get evidence snippets using keywords
        evidence_snippets, evidence_count, _ = self.get_keyword_evidence(biz, keywords, max_snippets=5)
        evidence_texts = "\n---\n".join(evidence_snippets) if evidence_snippets else "(None found)"

        # First 5 reviews + random 5 other reviews
        first_5 = reviews[:5]
        remaining = reviews[5:]
        random_5 = random.sample(remaining, min(5, len(remaining))) if remaining else []
        sample_reviews = first_5 + random_5
        review_texts = "\n---\n".join([r.get("text", "")[:500] for r in sample_reviews])

        cats = ", ".join(self.selected_categories)
        prompt = f"""Based on these reviews and evidence, estimate the probability (0-100%) that this restaurant truly belongs to the category "{cats}".

Restaurant: {biz.get('name')}
Listed categories: {biz.get('categories', 'Unknown')}

Keywords used for evidence: {', '.join(keywords[:10])}...
Keyword matches: {evidence_count} / {total_reviews} reviews contain category-related keywords

=== Evidence snippets (reviews mentioning keywords) ===
{evidence_texts}

=== Sample reviews (first 5 + random 5) ===
{review_texts}

Consider both the keyword match ratio and the content of reviews.
Reply with just the percentage and one sentence explanation. Example: "85% - Reviews consistently mention authentic Italian dishes and pasta."
"""

        try:
            response = await call_llm_async(prompt, system="You are a data quality evaluator.")
            response = response.strip()
            pct = self.parse_percentage(response)
            return (biz, pct, response)
        except Exception as e:
            return (biz, 0, f"[Error: {e}]")

    async def run_auto_mode(self) -> None:
        """Auto mode: batch estimate with early stopping, save ALL estimated."""
        scored = self.compute_richness_scores()
        

        self.console.print(f"\n[bold]Auto mode: Estimating {len(scored)} restaurants...[/bold]")

        all_results = []
        above_threshold_count = 0

        # Process in batches with early stopping
        for batch_start in range(0, len(scored), self.batch_size):
            batch = scored[batch_start:batch_start + self.batch_size]
            self.console.print(f"[dim]Processing batch {batch_start//self.batch_size + 1} ({batch_start+1}-{batch_start+len(batch)})...[/dim]")

            tasks = [self.estimate_category_fit_async(biz) for biz, _ in batch]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            # Count above self.threshold for early stopping
            above_threshold_count = sum(1 for _, pct, _ in all_results if pct >= self.threshold)

            # Early stop if we have enough above self.threshold
            if above_threshold_count >= self.target:
                self.console.print(f"[green]Reached {self.target} restaurants above {self.threshold}%. Stopping early.[/green]")
                break

        # Sort ALL estimated by percentage descending
        all_results.sort(key=lambda x: -x[1])

        # Save ALL estimated results (not just above self.threshold)
        for biz, pct, exp in all_results:
            self.save_selection(biz, pct, exp)

        # Write JSONL file
        self.write_selections_file()

        self.console.print(f"\n[bold green]Saved {len(all_results)} estimated restaurants.[/bold green]")
        self.console.print(f"[dim]Above {self.threshold}%: {above_threshold_count} | Below: {len(all_results) - above_threshold_count}[/dim]")

        # Print summary (top 20)
        self.console.print("\n[bold]═══ Top 20 Restaurants ═══[/bold]")
        for i, (biz, pct, exp) in enumerate(all_results[:20], 1):
            self.console.print(f"{i:3}. {biz['name'][:40]:<40} ({pct}%) {exp}")

    def get_category_evidence(self, biz: dict, page: int = 0, per_page: int = 3) -> Tuple[List[Tuple[str, str]], int, bool]:
        """Find review snippets containing category keywords with pagination.

        Returns: ([(snippet, matched_keyword), ...], total_matches, has_more)
        """
        # Use expanded keywords if available, otherwise fall back to category names
        if self.category_keywords:
            keywords = self.category_keywords
        else:
            keywords = [cat.lower() for cat in self.selected_categories]

        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        matches = []
        for r in reviews:
            text = r.get("text", "")
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                matches.append((len(text), text))

        # Sort by length (longest first)
        matches.sort(key=lambda x: -x[0])
        total = len(matches)

        # Pagination
        start_idx = page * per_page
        end_idx = start_idx + per_page
        page_matches = matches[start_idx:end_idx]

        snippets = []
        for _, text in page_matches:
            # Find keyword position, extract ~400 char window (100 before, 300 after)
            for kw in keywords:
                idx = text.lower().find(kw)
                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(text), idx + 300)
                    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                    snippets.append((snippet, kw))  # Return tuple with matched keyword
                    break

        has_more = end_idx < total
        return snippets, total, has_more

    def preview_city(self, city: str) -> None:
        """Show preview of selected city."""
        city_businesses = [b for b in self.businesses.values() if b.get("city") == city]
        cat_counts = self.get_category_counts(city)
        top_cats = list(cat_counts.items())[:5]
        samples = random.sample(city_businesses, min(5, len(city_businesses)))

        panel_content = Text()
        panel_content.append(f"Total Restaurants: ", style="bold")
        panel_content.append(f"{len(city_businesses)}\n\n")
        panel_content.append("Top Categories:\n", style="bold")
        for cat, count in top_cats:
            panel_content.append(f"  - {cat}: {count}\n")
        panel_content.append("\nSample Restaurants:\n", style="bold")
        for s in samples:
            panel_content.append(f"  - {s['name']} ({s.get('stars', '?')} stars)\n")

        self.console.print(Panel(panel_content, title=f"[bold cyan]Preview: {city}[/bold cyan]"))

    def preview_category(self, category: str) -> None:
        """Show preview of selected category within city."""
        filtered = [b for b in self.businesses.values()
                    if b.get("city") == self.selected_city and category in (b.get("categories") or "")]

        # Star distribution
        star_dist = defaultdict(int)
        for b in filtered:
            stars = b.get("stars", 0)
            star_dist[int(stars)] += 1

        samples = random.sample(filtered, min(5, len(filtered)))

        panel_content = Text()
        panel_content.append(f"Matching Restaurants: ", style="bold")
        panel_content.append(f"{len(filtered)}\n\n")
        panel_content.append("Star Distribution:\n", style="bold")
        max_count = max(star_dist.values()) if star_dist else 1
        for star in range(1, 6):
            count = star_dist.get(star, 0)
            bar_len = int((count / max_count) * 19) + 1 if count > 0 else 0
            bar = "█" * bar_len
            panel_content.append(f"  {star}★: {bar} ({count})\n")
        panel_content.append("\nSample Restaurants:\n", style="bold")
        for s in samples:
            panel_content.append(f"  - {s['name']} ({s.get('stars', '?')}★, {s.get('review_count', 0)} reviews)\n")

        self.console.print(Panel(panel_content, title=f"[bold cyan]Preview: {self.selected_city} > {category}[/bold cyan]"))

    def preview_categories(self, categories: List[str], cat_counts: dict) -> None:
        """Show preview of multiple selected categories within city."""
        # Get all matching businesses
        filtered = [b for b in self.businesses.values()
                    if b.get("city") == self.selected_city and
                    any(cat in (b.get("categories") or "") for cat in categories)]

        # Star distribution
        star_dist = defaultdict(int)
        for b in filtered:
            stars = b.get("stars", 0)
            star_dist[int(stars)] += 1

        samples = random.sample(filtered, min(5, len(filtered))) if filtered else []

        panel_content = Text()
        panel_content.append("Selected Categories:\n", style="bold")
        for cat in categories:
            count = cat_counts.get(cat, 0)
            panel_content.append(f"  - {cat} ({count})\n")
        panel_content.append(f"\nTotal Matching Restaurants: ", style="bold")
        panel_content.append(f"{len(filtered)}\n\n")
        panel_content.append("Star Distribution:\n", style="bold")
        max_count = max(star_dist.values()) if star_dist else 1
        for star in range(1, 6):
            count = star_dist.get(star, 0)
            bar_len = int((count / max_count) * 19) + 1 if count > 0 else 0
            bar = "█" * bar_len
            panel_content.append(f"  {star}★: {bar} ({count})\n")
        if samples:
            panel_content.append("\nSample Restaurants:\n", style="bold")
            for s in samples:
                panel_content.append(f"  - {s['name']} ({s.get('stars', '?')}★, {s.get('review_count', 0)} reviews)\n")

        cats_str = ", ".join(categories[:3])
        if len(categories) > 3:
            cats_str += f" +{len(categories) - 3} more"
        self.console.print(Panel(panel_content, title=f"[bold cyan]Preview: {self.selected_city} > {cats_str}[/bold cyan]"))

    def search_items(self, query: str, items: list) -> list:
        """Search items by name (case-insensitive partial match)."""
        query_lower = query.lower()
        return [(name, count) for name, count in items if query_lower in name.lower()]

    def paginated_select(self, items: list, title: str, prompt_text: str,
                         page_size: int = 20, allow_back: bool = False) -> tuple:
        """Reusable paginated selection. Returns (action, selection, page).

        Actions: 'select' (with selection), 'back', 'quit', or None (continue loop).
        """
        page = 0
        total_pages = max(1, (len(items) + page_size - 1) // page_size)

        while True:
            start = page * page_size
            end = min(start + page_size, len(items))
            displayed = items[start:end]

            table = Table(title=f"{title} {start + 1}-{end} of {len(items)} (Page {page + 1}/{total_pages})")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Name", style="bold")
            table.add_column("Count", justify="right")

            for i, (name, count) in enumerate(displayed, start + 1):
                table.add_row(str(i), name, str(count))

            self.console.print(table)
            nav = "[n/→]ext | [p/←]rev | [number] | [text] search"
            if allow_back:
                nav += " | [b]ack"
            self.console.print(f"[dim]{nav} | [q]uit[/dim]")
            self.console.print()

            choice = Prompt.ask(prompt_text, default="1")
            c = choice.lower().strip()

            # Navigation (n for next, p for prev, arrow keys also work)
            # Arrow keys send escape sequences: \x1b[C (right), \x1b[D (left)
            if c in ("n", "next") or "\x1b[C" in choice or "\x1b[B" in choice:
                page = min(page + 1, total_pages - 1)
            elif c in ("p", "prev") or "\x1b[D" in choice or "\x1b[A" in choice:
                page = max(page - 1, 0)
            elif c == "q":
                return ("quit", None, page)
            elif c == "b" and allow_back:
                return ("back", None, page)
            elif choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(items):
                    return ("select", items[idx - 1][0], page)
                self.console.print("[red]Invalid number[/red]")
            else:
                # Search
                matches = self.search_items(choice, items)
                if not matches:
                    self.console.print(f"[red]No match for '{choice}'[/red]")
                elif len(matches) == 1:
                    return ("select", matches[0][0], page)
                else:
                    self.console.print(f"\n[bold]Found {len(matches)} matches:[/bold]")
                    for i, (name, count) in enumerate(matches[:10], 1):
                        self.console.print(f"  {i}. {name} ({count})")
                    sub = Prompt.ask("Select number, or Enter to go back", default="")
                    if sub.isdigit() and 1 <= int(sub) <= min(len(matches), 10):
                        return ("select", matches[int(sub) - 1][0], page)
                    self.console.print("[yellow]Returning to list...[/yellow]")

    def select_city_loop(self) -> bool:
        """Iterative city selection with preview and confirm."""
        all_cities = list(self.get_city_counts().items())

        while True:
            action, selected, _ = self.paginated_select(
                all_cities, "Cities", "Select city", allow_back=False)

            if action == "quit":
                return False
            if action != "select":
                continue

            self.preview_city(selected)
            confirm = Prompt.ask("[C]onfirm / [B]ack", default="c").lower()
            if confirm == "b":
                continue  # Back to city selection
            if confirm == "c":
                self.selected_city = selected
                return True

    def parse_category_input(self, input_str: str, available_cats: list) -> List[str]:
        """Parse '1,3,5' or 'Italian, Mexican' into category list."""
        selected = []
        parts = [p.strip() for p in input_str.split(",")]
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(available_cats):
                    selected.append(available_cats[idx][0])
            else:
                # Match by name (partial, case-insensitive)
                for cat, _ in available_cats:
                    if part.lower() in cat.lower():
                        if cat not in selected:
                            selected.append(cat)
                        break
        return selected

    def select_category_loop(self) -> bool:
        """Iterative category selection with pagination. Supports multi-select."""
        cat_counts = self.get_category_counts(self.selected_city)
        all_cats = list(cat_counts.items())
        page = 0
        page_size = 20

        while True:
            total_pages = max(1, (len(all_cats) + page_size - 1) // page_size)
            start = page * page_size
            end = min(start + page_size, len(all_cats))
            displayed = all_cats[start:end]

            table = Table(title=f"Categories in {self.selected_city} {start + 1}-{end} of {len(all_cats)} (Page {page + 1}/{total_pages})")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Category", style="bold")
            table.add_column("Count", justify="right")

            for i, (cat, count) in enumerate(displayed, start + 1):
                table.add_row(str(i), cat, str(count))

            self.console.print(table)
            self.console.print("[dim][n/→]ext | [p/←]rev | [numbers] 1,3,5 | [text] search | [b]ack | [q]uit[/dim]")
            self.console.print()

            choice = Prompt.ask("Select categories", default="1")
            c = choice.lower().strip()

            # Navigation (n for next, p for prev, arrow keys also work)
            if c in ("n", "next") or "\x1b[C" in choice or "\x1b[B" in choice:
                page = min(page + 1, total_pages - 1)
                continue
            elif c in ("p", "prev") or "\x1b[D" in choice or "\x1b[A" in choice:
                page = max(page - 1, 0)
                continue
            elif c == "b":
                self.selected_city = None
                return None
            elif c == "q":
                return False

            # Parse multi-selection (works across all pages)
            selected_cats = self.parse_category_input(choice, all_cats)
            if not selected_cats:
                self.console.print("[red]No valid categories selected[/red]")
                continue

            self.console.print(f"\n[green]Selected {len(selected_cats)} categories:[/green]")
            for cat in selected_cats:
                self.console.print(f"  - {cat} ({cat_counts.get(cat, 0)})")

            self.preview_categories(selected_cats, cat_counts)

            action = Prompt.ask("[C]onfirm / [B]ack", default="c").lower()
            if action == "b":
                self.selected_city = None
                return None  # Back to city selection
            elif action == "c":
                self.selected_categories = selected_cats
                return True

    def display_restaurant(self, biz: dict, richness: int = 0) -> None:
        """Display restaurant info with richness score and category evidence."""
        attrs = biz.get("attributes") or {}

        content = Text()
        content.append(f"City: {biz.get('city', 'N/A')}    ", style="dim")
        cats_str = ", ".join(self.selected_categories[:3])
        if len(self.selected_categories) > 3:
            cats_str += f" +{len(self.selected_categories) - 3}"
        content.append(f"Categories: {cats_str}\n", style="dim")
        content.append(f"Richness Score: {richness:,} chars\n\n", style="bold cyan")

        content.append("Attributes:\n", style="bold")
        # Display in two columns
        attr_items = list(attrs.items())[:8]
        col_width = 40
        for i in range(0, len(attr_items), 2):
            left_key, left_val = attr_items[i]
            left_str = f"  - {left_key}: {left_val}"
            if len(left_str) > col_width:
                left_str = left_str[:col_width-3] + "..."
            if i + 1 < len(attr_items):
                right_key, right_val = attr_items[i + 1]
                right_str = f"  - {right_key}: {right_val}"
                content.append(f"{left_str:<{col_width}}{right_str}\n")
            else:
                content.append(f"{left_str}\n")
        if len(attrs) > 8:
            content.append(f"  ... and {len(attrs) - 8} more\n", style="dim")

        title = f"[bold]{biz['name']}[/bold] ({biz.get('stars', '?')}★, {biz.get('review_count', 0)} reviews)"
        self.console.print(Panel(content, title=title))

        # Category Evidence panel (first page only, 5 items)
        snippets, total, has_more = self.get_category_evidence(biz, page=0, per_page=5)
        if snippets:
            ev_content = Text()
            for i, (snippet, kw) in enumerate(snippets, 1):
                # Highlight matched keyword (case-insensitive)
                highlighted = re.sub(f"({re.escape(kw)})", r"[bold yellow]\1[/bold yellow]", snippet, flags=re.IGNORECASE)
                ev_content.append(f"{i}. ", style="bold")
                ev_content.append_text(Text.from_markup(f"{highlighted}\n\n"))
            title = f"[bold yellow]Category Evidence (1-{len(snippets)} of {total})[/bold yellow]"
            self.console.print(Panel(ev_content, title=title))
            if has_more:
                self.console.print("[dim]Press [E] to browse more evidence...[/dim]")
        else:
            self.console.print("[dim]No category keyword matches found in reviews[/dim]")

        # LLM Category Fit Estimation
        llm_estimate = self.estimate_category_fit(biz)
        self.console.print(f"[bold magenta]LLM Category Fit:[/bold magenta] {llm_estimate}")

    def display_all_attributes(self, biz: dict) -> None:
        """Display all attributes in two-column layout."""
        attrs = biz.get("attributes") or {}
        if not attrs:
            self.console.print("[dim]No attributes available[/dim]")
            return

        # Format items with truncation
        items = []
        for k, v in sorted(attrs.items()):
            v_str = str(v)
            if len(v_str) > 20:
                v_str = v_str[:17] + "..."
            items.append(f"{k}: {v_str}")

        # Split into two columns
        mid = (len(items) + 1) // 2
        left_col = items[:mid]
        right_col = items[mid:]

        # Pad shorter column
        while len(right_col) < len(left_col):
            right_col.append("")

        # Build output with fixed column width
        col_width = 38
        lines = []
        for left, right in zip(left_col, right_col):
            line = f"{left:<{col_width}} {right}"
            lines.append(line)

        content = "\n".join(lines)
        self.console.print(Panel(content, title=f"[bold]All Attributes ({len(attrs)})[/bold]"))

    def browse_evidence_loop(self, biz: dict) -> None:
        """Paginated evidence browser."""
        page = 0
        per_page = 5

        while True:
            snippets, total, has_more = self.get_category_evidence(biz, page, per_page)

            if not snippets and page == 0:
                self.console.print("[dim]No category evidence found[/dim]")
                return

            start_num = page * per_page + 1
            end_num = start_num + len(snippets) - 1

            ev_content = Text()
            for i, (snippet, kw) in enumerate(snippets, start_num):
                # Highlight matched keyword (case-insensitive)
                highlighted = re.sub(f"({re.escape(kw)})", r"[bold yellow]\1[/bold yellow]", snippet, flags=re.IGNORECASE)
                ev_content.append(f"{i}. ", style="bold")
                ev_content.append_text(Text.from_markup(f"{highlighted}\n\n"))

            self.console.print(Panel(
                ev_content,
                title=f"[bold yellow]Category Evidence ({start_num}-{end_num} of {total})[/bold yellow]"
            ))

            # Build dynamic options
            opts = []
            if has_more:
                opts.append("[N]ext")
            if page > 0:
                opts.append("[P]rev")
            opts.append("[B]ack")

            action = Prompt.ask(" / ".join(opts), default="b").lower()
            if action == "b":
                return
            elif action == "n" and has_more:
                page += 1
            elif action == "p" and page > 0:
                page -= 1

    def search_evidence_loop(self, biz: dict) -> None:
        """Search reviews for specific keyword."""
        keyword = Prompt.ask("Enter search keyword").strip().lower()
        if not keyword:
            self.console.print("[dim]No keyword entered[/dim]")
            return

        reviews = self.reviews_by_biz.get(biz["business_id"], [])
        matches = []
        for r in reviews:
            text = r.get("text", "")
            if keyword in text.lower():
                matches.append((len(text), text))

        if not matches:
            self.console.print(f"[dim]No reviews contain '{keyword}'[/dim]")
            return

        matches.sort(key=lambda x: -x[0])  # Longest first
        self.console.print(f"[green]Found {len(matches)} reviews with '{keyword}'[/green]")

        # Paginate through matches
        page = 0
        per_page = 5

        while True:
            start = page * per_page
            end = start + per_page
            page_matches = matches[start:end]

            ev_content = Text()
            for i, (_, text) in enumerate(page_matches, start + 1):
                # Find keyword and show ~400 char window
                idx = text.lower().find(keyword)
                start_pos = max(0, idx - 100)
                end_pos = min(len(text), idx + 300)
                snippet = ("..." if start_pos > 0 else "") + text[start_pos:end_pos] + ("..." if end_pos < len(text) else "")
                # Highlight matched keyword
                highlighted = re.sub(f"({re.escape(keyword)})", r"[bold yellow]\1[/bold yellow]", snippet, flags=re.IGNORECASE)
                ev_content.append(f"{i}. ", style="bold")
                ev_content.append_text(Text.from_markup(f"{highlighted}\n\n"))

            start_num = start + 1
            end_num = start + len(page_matches)
            self.console.print(Panel(
                ev_content,
                title=f"[bold yellow]Search: '{keyword}' ({start_num}-{end_num} of {len(matches)})[/bold yellow]"
            ))

            # Navigation
            opts = []
            has_more = end < len(matches)
            if has_more:
                opts.append("[N]ext")
            if page > 0:
                opts.append("[P]rev")
            opts.append("[B]ack")

            action = Prompt.ask(" / ".join(opts), default="b").lower()
            if action == "b":
                return
            elif action == "n" and has_more:
                page += 1
            elif action == "p" and page > 0:
                page -= 1

    def run_restaurant_loop(self) -> None:
        """Iterate through richness-sorted restaurants until 100 kept."""
        scored = self.compute_richness_scores()
        kept_count = 0
        self.target = 100

        self.console.print(f"\n[bold]Starting restaurant selection ({len(scored)} candidates, sorted by richness)[/bold]\n")

        idx = 0
        while idx < len(scored):
            if kept_count >= self.target:
                self.console.print(f"\n[green]Reached {self.target} restaurants. Stopping.[/green]")
                break

            biz, richness = scored[idx]
            self.console.print(f"\n[bold cyan]═══ Restaurant {idx + 1} of {len(scored)} | Kept: {kept_count}/{self.target} ═══[/bold cyan]")
            self.display_restaurant(biz, richness)

            action = Prompt.ask(
                "Keep? [Y]es / [N]o / [A]ttributes / [E]vidence / [K]eyword",
                default="y"
            ).lower()

            if action == "n":
                idx += 1
                continue
            elif action == "a":
                self.display_all_attributes(biz)
                # Stay on same restaurant
                continue
            elif action == "e":
                self.browse_evidence_loop(biz)
                # Stay on same restaurant
                continue
            elif action == "k":
                self.search_evidence_loop(biz)
                # Stay on same restaurant
                continue
            elif action == "y":
                # Get LLM estimate for saving
                llm_response = self.estimate_category_fit(biz)
                llm_percent = self.parse_percentage(llm_response)
                self.save_selection(biz, llm_percent, llm_response)
                kept_count += 1
                self.console.print(f"[green]Saved! ({kept_count}/{self.target})[/green]")
                idx += 1

        # Write JSONL file at end
        self.write_selections_file()
        self.console.print(f"\n[bold]Session complete. Kept {kept_count} restaurants.[/bold]")

    def save_selection(self, biz: dict, llm_percent: int, llm_reasoning: str) -> None:
        """Add selection to in-memory list (written at end)."""
        self.selections.append({
            "item_id": biz["business_id"],
            "llm_percent": llm_percent,
            "llm_reasoning": llm_reasoning
        })

    def write_selections_file(self) -> None:
        """Write all selections to file as JSONL (one dict per line)."""
        with open(self.output_file, "w") as f:
            for sel in self.selections:
                f.write(json.dumps(sel) + "\n")
        self.console.print(f"[green]Saved {len(self.selections)} selections to {self.output_file}[/green]")

    def run(self) -> None:
        """Main entry point for the curation tool."""
        self.console.print(Panel.fit(
            "[bold]Yelp Data Curation Tool[/bold]\n"
            "Human-in-the-loop selection for benchmark datasets",
            border_style="cyan"
        ))

        # Load business data
        self.load_business_data()

        # Choose output mode FIRST (before any selection)
        if not self.choose_output_mode():
            self.console.print("[yellow]Exiting...[/yellow]")
            return

        # City selection loop (with back navigation)
        while True:
            if not self.select_city_loop():
                self.console.print("[yellow]Exiting...[/yellow]")
                return

            # Category selection loop
            result = self.select_category_loop()
            if result is False:  # Quit
                self.console.print("[yellow]Exiting...[/yellow]")
                return
            elif result is None:  # Back to city
                continue
            else:  # Confirmed
                break

        # Generate category keywords ONCE for all restaurants
        with self.console.status("[bold blue]Generating category keywords with LLM..."):
            self.category_keywords = self.generate_category_keywords(self.selected_categories)
        self.console.print(f"[cyan]Keywords: {', '.join(self.category_keywords[:15])}{'...' if len(self.category_keywords) > 15 else ''}[/cyan]\n")

        # Load reviews for filtered businesses
        filtered = self.get_filtered_businesses()
        business_ids = {b["business_id"] for b in filtered}
        self.load_reviews_for_businesses(business_ids)

        # Ask for mode choice
        self.console.print(f"\n[bold]Mode selection:[/bold]")
        self.console.print(f"  [M]anual: Review each restaurant one by one")
        self.console.print(f"  [A]uto: LLM batch estimates, auto-keep ≥{self.threshold}%, target {self.target} restaurants")
        mode = Prompt.ask("\nChoose mode", default="m").lower()
        if mode == "a":
            asyncio.run(self.run_auto_mode())
        else:
            self.run_restaurant_loop()

        # Update metalog and show summary
        if self.selections:
            self.update_metalog()

        self.console.print("\n" + "═" * 50)
        self.console.print(f"[bold green]Session complete![/bold green]")
        self.console.print(f"Total selections: {len(self.selections)}")
        self.console.print(f"Output file: {self.output_file}")
        self.console.print(f"Metalog: {METALOG_FILE}")


def main():
    curator = YelpCurator()
    curator.run()


if __name__ == "__main__":
    main()
