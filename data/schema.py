"""Data schema definitions."""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Review:
    review_id: str
    review: str
    stars: float
    date: str


@dataclass
class Item:
    item_id: str
    item_name: str
    city: str
    neighborhood: str
    price_range: str
    cuisine: List[str]
    stars: float
    item_data: List[Review]
    gold_labels: Dict[str, int]  # R0, R1, ... → {-1, 0, 1}


def print_schema():
    """Print schema structure for debugging."""
    print("Item:")
    for field in Item.__dataclass_fields__:
        print(f"  {field}: {Item.__dataclass_fields__[field].type}")
    print("\nReview:")
    for field in Review.__dataclass_fields__:
        print(f"  {field}: {Review.__dataclass_fields__[field].type}")
