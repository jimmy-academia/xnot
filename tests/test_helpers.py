import pytest
from methods.anot.helpers import format_items_compact

def test_format_items_compact():
    items = [
        {"name": "Cafe A", "attributes": {"WiFi": "Free"}},
        {"name": "Cafe B", "attributes": {"WiFi": "Paid"}}
    ]
    formatted = format_items_compact(items)
    assert "Cafe A" in formatted
    assert "Cafe B" in formatted
    assert "WiFi" in formatted

def test_format_items_compact_truncation():
    long_val = "x" * 100
    items = [{"key": long_val}]
    formatted = format_items_compact(items)
    assert len(formatted) < 200  # Should be truncated
