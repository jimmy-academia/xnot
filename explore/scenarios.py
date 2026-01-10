"""
Decision scenarios for the benchmark.

Each scenario defines:
- A question (purpose-based)
- Keywords to filter reviews
- Conditions to check
- Logic to compute ground truth and verdict
"""

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Condition:
    """A single condition to evaluate."""
    id: str
    description: str
    output_type: Literal["count", "percentage", "boolean", "average"]
    threshold: float = None  # For threshold checks
    compare: Literal["gte", "lte", "gt", "lt", "eq"] = None
    tolerance: float = 0  # For numeric accuracy


@dataclass
class GroundTruth:
    """Ground truth for a scenario."""
    conditions: dict[str, any]  # C1, C2, etc. -> values
    verdict: bool
    reasoning: str


@dataclass
class Scenario:
    """A decision scenario with multiple conditions."""
    name: str
    question: str
    conditions: list[Condition]

    def compute_ground_truth(self, reviews: list[dict]) -> GroundTruth:
        """Override in subclasses."""
        raise NotImplementedError


def matches_any(text: str, keywords: list[str], word_boundary: bool = True) -> bool:
    """Check if text contains any of the keywords."""
    text_lower = text.lower()
    for kw in keywords:
        if word_boundary:
            # Use word boundary to avoid partial matches
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                return True
        else:
            if kw.lower() in text_lower:
                return True
    return False


# =============================================================================
# SCENARIO 1: Business Dinner
# =============================================================================

class BusinessDinnerScenario(Scenario):
    """Is this restaurant suitable for a business dinner?"""

    def __init__(self):
        super().__init__(
            name="business_dinner",
            question="Is this restaurant suitable for a business dinner?",
            conditions=[
                Condition("C1", "Count of reviews mentioning business/meeting/client/professional", "count"),
                Condition("C2", "Is C1 >= 10?", "boolean"),
                Condition("C3", "Among C1 reviews, percentage with 4+ stars", "percentage"),
                Condition("C4", "Is C3 >= 70%?", "boolean"),
                Condition("C5", "Any negative review (1-2 stars) mentioning loud/noisy/crowded?", "boolean"),
            ]
        )
        self.topic_keywords = ["business", "meeting", "client", "professional", "work", "colleague"]
        self.veto_keywords = ["loud", "noisy", "crowded", "too loud", "couldn't hear"]

    def compute_ground_truth(self, reviews: list[dict]) -> GroundTruth:
        # C1: Count topic reviews
        topic_reviews = [r for r in reviews if matches_any(r["text"], self.topic_keywords)]
        c1 = len(topic_reviews)

        # C2: Threshold check
        c2 = c1 >= 10

        # C3: Percentage positive among topic reviews
        if c1 > 0:
            positive = sum(1 for r in topic_reviews if r["stars"] >= 4)
            c3 = round(positive / c1 * 100, 1)
        else:
            c3 = 0.0

        # C4: Percentage threshold
        c4 = c3 >= 70

        # C5: Veto check - any negative review with veto keywords?
        veto_found = False
        for r in reviews:
            if r["stars"] <= 2 and matches_any(r["text"], self.veto_keywords):
                veto_found = True
                break
        c5 = veto_found  # True means veto triggered (bad)

        # Verdict: YES if C2=True AND C4=True AND C5=False
        verdict = c2 and c4 and (not c5)

        reasoning = f"C1={c1} (need >=10), C3={c3}% (need >=70%), veto={'found' if c5 else 'none'}"

        return GroundTruth(
            conditions={"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
            verdict=verdict,
            reasoning=reasoning
        )

    def format_rules(self) -> str:
        return """RULES:
1. Count reviews mentioning: business, meeting, client, professional, work, colleague
2. This count must be >= 10 for sufficient evidence
3. Among those reviews, calculate the percentage with 4+ stars
4. This percentage must be >= 70%
5. Check if ANY negative review (1-2 stars) mentions: loud, noisy, crowded
6. If conditions 2, 4 are met AND no veto in condition 5 → YES, otherwise → NO

REQUIRED OUTPUT FORMAT:
C1: [count of business-related reviews]
C2: [YES if C1 >= 10, else NO]
C3: [percentage of C1 reviews with 4+ stars]
C4: [YES if C3 >= 70%, else NO]
C5: [YES if veto found, NO if no veto]
VERDICT: [YES or NO]"""


# =============================================================================
# SCENARIO 2: Family with Kids
# =============================================================================

class FamilyScenario(Scenario):
    """Is this restaurant good for families with young children?"""

    def __init__(self):
        super().__init__(
            name="family_kids",
            question="Is this restaurant good for families with young children?",
            conditions=[
                Condition("C1", "Count of reviews mentioning kids/children/family/child", "count"),
                Condition("C2", "Is C1 >= 5?", "boolean"),
                Condition("C3", "Among C1 reviews, percentage with 4+ stars", "percentage"),
                Condition("C4", "Is C3 >= 75%?", "boolean"),
                Condition("C5", "Any review mentioning not kid-friendly/no kids/adults only/21+?", "boolean"),
            ]
        )
        self.topic_keywords = ["kids", "children", "family", "child", "toddler", "baby"]
        self.veto_keywords = ["not kid-friendly", "no kids", "adults only", "21+", "not for children", "no children"]

    def compute_ground_truth(self, reviews: list[dict]) -> GroundTruth:
        topic_reviews = [r for r in reviews if matches_any(r["text"], self.topic_keywords)]
        c1 = len(topic_reviews)
        c2 = c1 >= 5

        if c1 > 0:
            positive = sum(1 for r in topic_reviews if r["stars"] >= 4)
            c3 = round(positive / c1 * 100, 1)
        else:
            c3 = 0.0
        c4 = c3 >= 75

        veto_found = any(matches_any(r["text"], self.veto_keywords, word_boundary=False) for r in reviews)
        c5 = veto_found

        verdict = c2 and c4 and (not c5)
        reasoning = f"C1={c1} (need >=5), C3={c3}% (need >=75%), veto={'found' if c5 else 'none'}"

        return GroundTruth(
            conditions={"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
            verdict=verdict,
            reasoning=reasoning
        )

    def format_rules(self) -> str:
        return """RULES:
1. Count reviews mentioning: kids, children, family, child, toddler, baby
2. This count must be >= 5 for sufficient evidence
3. Among those reviews, calculate the percentage with 4+ stars
4. This percentage must be >= 75%
5. Check if ANY review mentions: "not kid-friendly", "no kids", "adults only", "21+", "not for children"
6. If conditions 2, 4 are met AND no veto in condition 5 → YES, otherwise → NO

REQUIRED OUTPUT FORMAT:
C1: [count of family-related reviews]
C2: [YES if C1 >= 5, else NO]
C3: [percentage of C1 reviews with 4+ stars]
C4: [YES if C3 >= 75%, else NO]
C5: [YES if veto found, NO if no veto]
VERDICT: [YES or NO]"""


# =============================================================================
# SCENARIO 3: Quick Lunch
# =============================================================================

class QuickLunchScenario(Scenario):
    """Is this restaurant suitable for a quick weekday lunch?"""

    def __init__(self):
        super().__init__(
            name="quick_lunch",
            question="Is this restaurant suitable for a quick weekday lunch?",
            conditions=[
                Condition("C1", "Count of reviews mentioning lunch/midday/noon", "count"),
                Condition("C2", "Is C1 >= 8?", "boolean"),
                Condition("C3", "Among C1 reviews, count mentioning wait complaints", "count"),
                Condition("C4", "Wait complaint percentage (C3/C1)", "percentage"),
                Condition("C5", "Is C4 < 25%?", "boolean"),
            ]
        )
        self.topic_keywords = ["lunch", "midday", "noon", "lunchtime", "lunch break"]
        self.wait_keywords = ["waited", "wait", "slow", "forever", "took forever", "long wait", "slow service"]

    def compute_ground_truth(self, reviews: list[dict]) -> GroundTruth:
        topic_reviews = [r for r in reviews if matches_any(r["text"], self.topic_keywords)]
        c1 = len(topic_reviews)
        c2 = c1 >= 8

        # C3: Count wait complaints among topic reviews
        wait_complaints = [r for r in topic_reviews if matches_any(r["text"], self.wait_keywords)]
        c3 = len(wait_complaints)

        # C4: Percentage
        if c1 > 0:
            c4 = round(c3 / c1 * 100, 1)
        else:
            c4 = 0.0

        # C5: Is complaint rate acceptable (< 25%)?
        c5 = c4 < 25

        verdict = c2 and c5
        reasoning = f"C1={c1} (need >=8), wait complaints={c3} ({c4}%, need <25%)"

        return GroundTruth(
            conditions={"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
            verdict=verdict,
            reasoning=reasoning
        )

    def format_rules(self) -> str:
        return """RULES:
1. Count reviews mentioning: lunch, midday, noon, lunchtime, "lunch break"
2. This count must be >= 8 for sufficient evidence
3. Among those reviews, count how many mention wait complaints: waited, wait, slow, forever, "took forever", "long wait", "slow service"
4. Calculate the wait complaint percentage (C3 / C1 * 100)
5. This percentage must be < 25% (meaning most lunch experiences are quick)
6. If conditions 2 AND 5 are met → YES, otherwise → NO

REQUIRED OUTPUT FORMAT:
C1: [count of lunch-related reviews]
C2: [YES if C1 >= 8, else NO]
C3: [count of wait complaints among C1]
C4: [wait complaint percentage]
C5: [YES if C4 < 25%, else NO]
VERDICT: [YES or NO]"""


# =============================================================================
# SCENARIO 4: Romantic/Special Occasion
# =============================================================================

class RomanticScenario(Scenario):
    """Would you recommend this for an anniversary or romantic dinner?"""

    def __init__(self):
        super().__init__(
            name="romantic_occasion",
            question="Would you recommend this for an anniversary or romantic dinner?",
            conditions=[
                Condition("C1", "Count of reviews mentioning romantic/date/anniversary/special occasion/celebrate", "count"),
                Condition("C2", "Is C1 >= 5?", "boolean"),
                Condition("C3", "Average star rating of C1 reviews", "average"),
                Condition("C4", "Is C3 >= 4.0?", "boolean"),
                Condition("C5", "Any C1 review with <= 2 stars?", "boolean"),
            ]
        )
        self.topic_keywords = ["romantic", "date night", "date", "anniversary", "special occasion",
                               "celebrate", "celebration", "proposal", "valentine"]

    def compute_ground_truth(self, reviews: list[dict]) -> GroundTruth:
        topic_reviews = [r for r in reviews if matches_any(r["text"], self.topic_keywords)]
        c1 = len(topic_reviews)
        c2 = c1 >= 5

        # C3: Average rating
        if c1 > 0:
            c3 = round(sum(r["stars"] for r in topic_reviews) / c1, 2)
        else:
            c3 = 0.0
        c4 = c3 >= 4.0

        # C5: Any low-rated topic review?
        low_rated = any(r["stars"] <= 2 for r in topic_reviews)
        c5 = low_rated  # True means bad review exists (veto)

        verdict = c2 and c4 and (not c5)
        reasoning = f"C1={c1} (need >=5), avg={c3} (need >=4.0), low-rated={'found' if c5 else 'none'}"

        return GroundTruth(
            conditions={"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
            verdict=verdict,
            reasoning=reasoning
        )

    def format_rules(self) -> str:
        return """RULES:
1. Count reviews mentioning: romantic, "date night", date, anniversary, "special occasion", celebrate, celebration, proposal, valentine
2. This count must be >= 5 for sufficient evidence
3. Calculate the average star rating of those reviews
4. This average must be >= 4.0
5. Check if ANY of those reviews has <= 2 stars (a bad romantic experience)
6. If conditions 2, 4 are met AND no bad experience in condition 5 → YES, otherwise → NO

REQUIRED OUTPUT FORMAT:
C1: [count of romantic-related reviews]
C2: [YES if C1 >= 5, else NO]
C3: [average star rating of C1 reviews]
C4: [YES if C3 >= 4.0, else NO]
C5: [YES if any C1 review has <= 2 stars, else NO]
VERDICT: [YES or NO]"""


# =============================================================================
# REGISTRY
# =============================================================================

ALL_SCENARIOS = [
    BusinessDinnerScenario(),
    FamilyScenario(),
    QuickLunchScenario(),
    RomanticScenario(),
]


def get_scenario(name: str) -> Scenario:
    for s in ALL_SCENARIOS:
        if s.name == name:
            return s
    raise ValueError(f"Unknown scenario: {name}")
