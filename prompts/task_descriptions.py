"""
Standard task descriptions for all methods.

These descriptions are generic (items with attributes and reviews) so they
can be used across different datasets and evaluation scenarios.
"""

# Full task description for methods that need detailed context
RANKING_TASK = """[TASK TYPE]
RANKING - Given N candidate items, find the top-{k} that best match the user's criteria.

[USER REQUEST]
{context}

[AVAILABLE DATA]
Each item has:
- Structured metadata: attributes (key-value pairs like WiFi, NoiseLevel, PriceRange)
- Categories: list of category tags
- Hours: operating hours by day
- User reviews: review text with star ratings

[OUTPUT]
Return indices of top-{k} items, ranked by relevance to user request.
Scoring convention:
  1 = matches the user's criteria
  0 = unknown/uncertain
 -1 = does not match the user's criteria
"""

# Compact version for prompts that already have detailed instructions
RANKING_TASK_COMPACT = """RANKING task: Find top-{k} items matching the user request.

User request: {context}

Each item has: attributes (key-value), categories, hours, and user reviews.
Score: 1 (matches), 0 (unknown), -1 (doesn't match).
"""
