#!/usr/bin/env python3
"""Quick test for Phase 1 exploration."""

import os
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["KNOT_DEBUG"] = "1"

from data.loader import Dataset
from methods.anot import AdaptiveNetworkOfThought, execute_tool

def test_execute_tool():
    """Test the execute_tool function."""
    print("=== Testing execute_tool ===\n")

    sample_data = {
        "items": [
            {"item_id": "1", "item_name": "Cafe A", "attributes": {"WiFi": True}},
            {"item_id": "2", "item_name": "Cafe B", "attributes": {"DriveThru": True, "WiFi": True}},
            {"item_id": "3", "item_name": "Cafe C", "attributes": {"NoiseLevel": "quiet"}},
        ]
    }

    tests = [
        ("keys", ""),
        ("keys", "items[0]"),
        ("keys", "items[0].attributes"),
        ("count", "items"),
        ("type", "items[0].attributes"),
        ("sample", "items[1].item_name"),
        ("union_keys", "items[*].attributes"),  # Should find DriveThru!
    ]

    for tool, path in tests:
        result = execute_tool(tool, path, sample_data)
        print(f'{tool}("{path}") → {result}')

    print()

def test_tool_paths():
    """Test specific tool paths the LLM is using."""
    print("=== Testing Tool Paths ===\n")

    query = {
        "items": [
            {"item_id": "1", "attributes": {"WiFi": True}},
            {"item_id": "2", "attributes": {"DriveThru": True, "WiFi": True}},
            {"item_id": "3", "attributes": {"NoiseLevel": "quiet"}},
        ]
    }

    paths = [
        ("union_keys", "items[*]"),           # What LLM tried in round 1
        ("keys", "items[*].attributes"),      # What LLM tried in round 2 (wrong)
        ("union_keys", "items[*].attributes"), # What it SHOULD use
        ("union_keys", "items[*].item_data.*"),
        ("union_keys", "items[*].*"),
    ]

    for tool, path in paths:
        result = execute_tool(tool, path, query)
        print(f'{tool}("{path}")')
        print(f'  → {result}\n')

def test_phase1():
    """Test Phase 1 exploration with mock data."""
    print("=== Testing Phase 1 Exploration ===\n")

    # Create realistic mock data that mimics real dataset structure
    context = "I need a cafe with a drive-thru option - I can't get my kids out of the car"

    query = {
        "items": [
            {
                "index": 0,
                "item_id": "cafe1",
                "item_name": "Tria Cafe Rittenhouse",
                "city": "Philadelphia",
                "address": "123 Main St",
                "attributes": {"WiFi": "free", "NoiseLevel": "quiet", "OutdoorSeating": True},
                "categories": ["Coffee & Tea", "Breakfast"],
                "hours": {"Monday": "7:00-17:00"},
                "item_data": [{"review": "Great coffee!"}],
            },
            {
                "index": 1,
                "item_id": "cafe2",
                "item_name": "Milkcrate Cafe",
                "city": "Philadelphia",
                "address": "456 Oak Ave",
                "attributes": {"DriveThru": True, "WiFi": "free", "GoodForKids": True},
                "categories": ["Coffee & Tea"],
                "hours": {"Monday": "6:00-18:00"},
                "item_data": [{"review": "Easy drive-thru!"}],
            },
            {
                "index": 2,
                "item_id": "cafe3",
                "item_name": "Front Street Cafe",
                "city": "Philadelphia",
                "address": "789 Elm Rd",
                "attributes": {"WiFi": "no", "HasTV": True},
                "categories": ["Cafe", "Breakfast"],
                "hours": {"Monday": "8:00-15:00"},
                "item_data": [{"review": "Nice atmosphere."}],
            },
        ]
    }

    print(f"Context: {context}\n")
    print(f"Data: {len(query['items'])} items\n")

    # Test union_keys first to verify DriveThru exists
    print("Checking if DriveThru exists in any item...")
    result = execute_tool("union_keys", "items[*].attributes", query)
    print(f"All attribute keys: {result}")
    if "DriveThru" in result:
        print("✓ DriveThru found!\n")
    else:
        print("✗ DriveThru NOT found\n")

    # Initialize method
    method = AdaptiveNetworkOfThought(defense=False, run_dir="results/dev/test_phase1")

    # Run Phase 1
    print("Running Phase 1 exploration...")
    lwt = method.phase1_explore(query, context)

    print(f"\n=== Generated LWT ===\n{lwt}")

if __name__ == "__main__":
    test_execute_tool()
    print("\n" + "="*60 + "\n")
    test_tool_paths()
    print("\n" + "="*60 + "\n")
    test_phase1()
