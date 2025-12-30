#!/usr/bin/env python3
"""Human-in-the-Loop Yelp data curation tool for benchmark dataset creation."""

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


# File paths
RAW_DIR = Path(__file__).parent.parent / "raw"
BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
OUTPUT_FILE = Path(__file__).parent.parent / "yelp_selections.jsonl"


class YelpCurator:
    """Human-in-the-loop curation tool for Yelp data."""

    def __init__(self):
        self.console = Console()
        self.businesses: Dict[str, dict] = {}
        self.reviews_by_biz: Dict[str, List[dict]] = defaultdict(list)
        self.selected_city: Optional[str] = None
        self.selected_categories: List[str] = []  # Multi-category support
        self.selections: List[dict] = []

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

    def get_category_evidence(self, biz: dict, page: int = 0, per_page: int = 3) -> Tuple[List[str], int, bool]:
        """Find review snippets containing category keywords with pagination.

        Returns: (snippets, total_matches, has_more)
        """
        # Use category names as keywords
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
                    snippets.append(snippet)
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
        for key, val in list(attrs.items())[:8]:
            content.append(f"  - {key}: {val}\n")
        if len(attrs) > 8:
            content.append(f"  ... and {len(attrs) - 8} more\n", style="dim")

        title = f"[bold]{biz['name']}[/bold] ({biz.get('stars', '?')}★, {biz.get('review_count', 0)} reviews)"
        self.console.print(Panel(content, title=title))

        # Category Evidence panel (first page only, 5 items)
        snippets, total, has_more = self.get_category_evidence(biz, page=0, per_page=5)
        if snippets:
            ev_content = Text()
            for i, snippet in enumerate(snippets, 1):
                ev_content.append(f"{i}. ", style="bold")
                ev_content.append(f"{snippet}\n\n")
            title = f"[bold yellow]Category Evidence (1-{len(snippets)} of {total})[/bold yellow]"
            self.console.print(Panel(ev_content, title=title))
            if has_more:
                self.console.print("[dim]Press [E] to browse more evidence...[/dim]")
        else:
            self.console.print("[dim]No category keyword matches found in reviews[/dim]")

    def display_all_attributes(self, biz: dict) -> None:
        """Display all attributes in multi-column layout."""
        attrs = biz.get("attributes") or {}
        if not attrs:
            self.console.print("[dim]No attributes available[/dim]")
            return

        # Calculate columns based on terminal width
        term_width = shutil.get_terminal_size().columns
        items = [f"{k}: {v}" for k, v in sorted(attrs.items())]

        # Estimate column width from longest item
        max_item_len = max(len(str(item)) for item in items) if items else 20
        col_width = min(max_item_len + 4, 50)  # Cap at 50 chars
        num_cols = max(1, (term_width - 4) // col_width)  # -4 for panel borders

        # Build table with dynamic columns
        table = Table(show_header=False, box=None, padding=(0, 2))
        for _ in range(num_cols):
            table.add_column(width=col_width)

        # Fill rows
        for i in range(0, len(items), num_cols):
            row = items[i:i + num_cols]
            # Pad row if needed
            while len(row) < num_cols:
                row.append("")
            table.add_row(*row)

        self.console.print(Panel(table, title=f"[bold]All Attributes ({len(attrs)})[/bold]"))

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
            for i, snippet in enumerate(snippets, start_num):
                ev_content.append(f"{i}. ", style="bold")
                ev_content.append(f"{snippet}\n\n")

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

    def run_restaurant_loop(self) -> None:
        """Iterate through richness-sorted restaurants until 100 kept."""
        scored = self.compute_richness_scores()
        kept_count = 0
        target = 100

        self.console.print(f"\n[bold]Starting restaurant selection ({len(scored)} candidates, sorted by richness)[/bold]\n")

        idx = 0
        while idx < len(scored):
            if kept_count >= target:
                self.console.print(f"\n[green]Reached {target} restaurants. Stopping.[/green]")
                break

            biz, richness = scored[idx]
            self.console.print(f"\n[bold cyan]═══ Restaurant {idx + 1} of {len(scored)} | Kept: {kept_count}/{target} ═══[/bold cyan]")
            self.display_restaurant(biz, richness)

            action = Prompt.ask(
                "[K]eep / [S]kip / [A]ttributes / [E]vidence / [Q]uit",
                default="k"
            ).lower()

            if action == "q":
                break
            elif action == "s":
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
                self.process_and_save(biz, richness)
                kept_count += 1
                self.console.print(f"[green]Saved! ({kept_count}/{target})[/green]")
                idx += 1

        self.console.print(f"\n[bold]Session complete. Kept {kept_count} restaurants.[/bold]")

    def process_and_save(self, biz: dict, richness: int) -> None:
        """Automatically process and save restaurant with bucketed reviews."""
        reviews = self.reviews_by_biz.get(biz["business_id"], [])

        # Bucket by stars (1-5)
        buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
        for r in reviews:
            star = int(r.get("stars", 3))
            star = max(1, min(5, star))  # Clamp to 1-5
            buckets[star].append(r)

        # Sort each bucket by length (longest first)
        for star in buckets:
            buckets[star].sort(key=lambda r: -len(r.get("text", "")))

        # Build structured output
        reviews_by_star = {}
        for star, bucket in buckets.items():
            reviews_by_star[f"{star}_star"] = [
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

        # Save
        self.save_selection(biz, reviews_by_star, richness)

    def save_selection(self, biz: dict, reviews_by_star: dict, richness: int) -> None:
        """Save a curated selection to the output file (new format with star buckets)."""
        cats = (biz.get("categories") or "").split(", ")
        cats = [c.strip() for c in cats if c.strip()]

        selection = {
            "item_id": biz["business_id"],
            "item_name": biz["name"],
            "city": biz.get("city"),
            "categories": cats,
            "stars": biz.get("stars"),
            "review_count": biz.get("review_count"),
            "richness_score": richness,
            "reviews_by_star": reviews_by_star
        }

        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(selection) + "\n")

        self.selections.append(selection)
        self.console.print(f"[green]Saved selection to {OUTPUT_FILE}[/green]")
        self.console.print(f"[dim]Total selections this session: {len(self.selections)}[/dim]")

    def run(self) -> None:
        """Main entry point for the curation tool."""
        self.console.print(Panel.fit(
            "[bold]Yelp Data Curation Tool[/bold]\n"
            "Human-in-the-loop selection for benchmark datasets",
            border_style="cyan"
        ))

        # Load business data
        self.load_business_data()

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

        # Load reviews for filtered businesses
        filtered = self.get_filtered_businesses()
        business_ids = {b["business_id"] for b in filtered}
        self.load_reviews_for_businesses(business_ids)

        # Run main restaurant selection loop
        self.run_restaurant_loop()

        # Summary
        self.console.print("\n" + "═" * 50)
        self.console.print(f"[bold green]Session complete![/bold green]")
        self.console.print(f"Total selections: {len(self.selections)}")
        self.console.print(f"Output file: {OUTPUT_FILE}")


def main():
    curator = YelpCurator()
    curator.run()


if __name__ == "__main__":
    main()
