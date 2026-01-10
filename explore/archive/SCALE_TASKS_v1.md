# SCALE Task Generation Matrix: 10 Groups x 10 Tasks (100 Total)

We define 10 "Failure Groups" (Reasoning Primitives) where LLMs struggle in large contexts. Each group has 10 specific task templates that can be parameterized.

## G1: Attribute Extraction (The "Search" Primitive)
*Goal: Test ability to find needle-in-haystack attributes.*
1.  **Existence**: Does the review explicitly mention [Attribute]?
2.  **Value Extraction**: What is the specific [Value] of [Attribute]? (e.g., "Wait time was 45 mins")
3.  **Boolean Check**: Is [Attribute] present? (Yes/No)
4.  **Absence Check**: Confirm [Attribute] is NOT mentioned.
5.  **List Extraction**: List all [Items] mentioned.
6.  **First Mention**: What is the FIRST attribute mentioned?
7.  **Last Mention**: What is the LAST attribute mentioned?
8.  **Frequency**: How many times is [Term] used?
9.  **Contextual Meaning**: Does "hot" refer to temperature or spice?
10. **Ambiguity Resolution**: Which "service" (dinner/valet) is referred to?

## G2: Quantitative Aggregation (The "Math" Primitive)
*Goal: Test arithmetic stability over text.*
11. **Count**: How many reviews mention [X]?
12. **Average**: Calculate average rating of reviews mentioning [X].
13. **Max**: What is the highest rating for [X]?
14. **Min**: What is the lowest rating for [X]?
15. **Sum**: Total number of "useful" votes for reviews mentioning [X].
16. **Ratio**: Percentage of reviews mentioning [X].
17. **Difference**: Difference in count between [X] and [Y].
18. **Mode**: Most common rating for [X].
19. **Threshold**: Are there more than N reviews for [X]?
20. **Weighted Avg**: Calculate weighted score based on useful counts.

## G3: Temporal Reasoning (The "Time" Primitive)
*Goal: Test sorting and time-series analysis.*
21. **Ordering**: Is the latest review positive?
22. **Trend**: Is the sentiment trend increasing or decreasing?
23. **Before/After**: Avg rating before vs. after [Date].
24. **Seasonality**: Are reviews in [Season] better?
25. **Duration**: Time span between first and last review.
26. **Recency Bias**: Avg rating of last N reviews vs rest.
27. **Specific Date**: Find review on [Date].
28. **Day of Week**: Are weekends worse than weekdays?
29. **Gap Analysis**: Longest gap between reviews.
30. **Sequence**: Did [Event A] happen before [Event B]?

## G4: Entity Resolution (The "Clustering" Primitive)
*Goal: Test ability to link variable names to single entities.*
31. **Staff ID**: Map "Server Mike", "Mike", "Michael" to one entity.
32. **Product ID**: Map "The burger", "Cheeseburger", "Mac's Burger" to one dish.
33. **Most Mentioned**: Who is the most mentioned staff?
34. **Entity Sentiment**: Specific sentiment for [Entity].
35. **Cross-Ref**: Did the user who mentioned [Entity A] also mention [Entity B]?
36. **Disambiguation**: Distinguish "The manager" (male) from "The manager" (female).
37. **Role Identification**: Identify role of [Person].
38. **Unique Entities**: Count distinct staff members mentioned.
39. **Co-Occurrence**: Who worked with [Staff X]?
40. **Entity Trajectory**: Did sentiment for [Staff X] change over time?

## G5: Conditional Logic (The "Filter" Primitive)
*Goal: Test multi-step calculated filtering.*
41. **Filter-Then-Count**: Count [X] ONLY in 5-star reviews.
42. **Filter-Then-Avg**: Avg star of reviews with >100 chars.
43. **Double Filter**: Reviews with [X] AND [Y].
44. **Exclusion**: Avg star of reviews NOT mentioning [X].
45. **Dependent Logic**: If [Condition A], count [X], else count [Y].
46. **Nested Filter**: Among [Group A], find [Subgroup B].
47. **User-Conditional**: Avg rating from "Elite" users only.
48. **Length-Conditional**: Sentinel of longest review.
49. **Date-Conditional**: Count [X] in [Year].
50. **Rating-Conditional**: Common words in 1-star reviews.

## G6: Cross-Aspect Correlation (The "Link" Primitive)
*Goal: Test ability to connect disparate signals.*
51. **Correlation**: Do [Aspect A] complaints predict [Aspect B] complaints?
52. **Causality**: Did [Aspect A] cause [Aspect B] (textual implication)?
53. **Trade-off**: "Great food but bad service" pattern count.
54. **Dominance**: Which aspect drives the rating more?
55. **Independence**: Is [Aspect A] independent of rating?
56. **Cluster**: Group reviews by dominant aspect.
57. **Signal Strength**: Which aspect has strongest sentiment?
58. **Conflict**: Positive [Aspect A] vs Negative [Aspect A] in same review.
59. **Aspect Pivot**: How does [Aspect] change across ratings?
60. **Synergy**: "Food and Service both good" count.

## G7: Persona Analysis (The "Segment" Primitive)
*Goal: Test theory-of-mind and user segmentation.*
61. **User Type**: Classify reviewer as [Type] (e.g. Family, Couple).
62. **Intent**: Was the visit for [Occasion]?
63. **Satisfaction by Type**: Do [Families] like it more?
64. **Credibility**: Weight by user review count.
65. **Style**: Formal vs Informal language impact.
66. **Demographic Proxy**: Guess context (e.g. "kids" -> Parent).
67. **Expertise**: Is user a "Foodie"?
68. **Bias Detection**: Is user unfairly biased?
69. **Response Prediction**: How would [User Type] rate this?
70. **User History**: Does user mention previous visits?

## G8: Outlier Detection (The "Anomaly" Primitive)
*Goal: Test ability to ignore noise and find signals.*
71. **Rating Outlier**: Find review with rating far from mean.
72. **Length Outlier**: Find exceptionally long/short review.
73. **Sentiment Outlier**: Positive text but low stars (Irony).
74. **Topic Outlier**: Mention of irrelevant topic.
75. **Spam Detection**: Identify "fake" sounding review.
76. **Contrarian**: User disagreeing with consensus.
77. **Burst Detection**: Sudden spike in [Topic].
78. **Vocabulary Outlier**: Use of unique/rare words.
79. **Formatting Outlier**: ALL CAPS or strange punctuation.
80. **Date Outlier**: Review from different era.

## G9: Contradiction Detection (The "Conflict" Primitive)
*Goal: Test logical consistency checking.*
81. **Self-Contradiction**: "Food was good... food was bad".
82. **Fact vs Opinion**: "Wait was 5 mins" vs "Wait was long".
83. **Review vs Metadata**: Review says "No Wi-Fi", metadata says "Has Wi-Fi".
84. **User vs User**: Direct disagreement on specific fact.
85. **Rating vs Text**: 1-star but "Loved it".
86. **Date Contradiction**: "Visited in summer" (date is Dec).
87. **Price Contradiction**: "$$" vs "Cheap".
88. **Menu Contradiction**: "Had the steak" (Steak not on menu).
89. **Policy Contradiction**: "Allowed dog" vs "No Pets".
90. **Chain-of-Though Error**: Premise does not match Conclusion.

## G10: Decision Gating (The "Gate" Primitive)
*Goal: Test final actionable decision making.*
91. **Binary Choice**: "Yes/No" final recommendation.
92. **Constraint Satisfaction**: "Fits budget AND diet?"
93. **Comparative**: "Better than [Competitor]?" (Simulated).
94. **Risk Assessment**: "Is it safe for [Allergy]?"
95. **Policy Check**: "Does it violate [Rule]?"
96. **Suitability**: "Good for [Date Night]?"
97. **Value Judgement**: "Is it worth the money?"
98. **Navigation**: "Should I go now?" (Time-sensitive).
99. **Action**: "Order [Dish]?"
100. **The SCALE Verdict**: Final holistic pass/fail.
