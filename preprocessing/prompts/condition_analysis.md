# Condition Analysis Prompts

Prompts for building and analyzing condition satisfaction matrices.

## Generic Template

### Build Condition Matrix

```
Build a condition satisfaction matrix for the {N} selected items.

**Source:**
- data/{name}/restaurants.jsonl
- data/{name}/reviews.jsonl

**Condition Categories:**
1. Boolean attributes (True/False)
2. Enum attributes (values from set)
3. Numeric attributes (ranges)
4. Hours (day/time availability)
5. Review patterns (keyword presence)
6. Review metadata (credibility thresholds)

**Output:**
For each condition:
- Condition name and specification
- Count of items satisfying it
- List of item indices that satisfy it
- Classification: unique (1), rare (2-5), common (>5)

**Format:**
| Condition | Count | Items | Class |
|-----------|-------|-------|-------|
| DriveThru=True | 1 | [0] | unique |
| WiFi=free | 35 | [0,1,2,...] | common |
```

### Identify Unique Identifiers

```
From the condition matrix, identify unique identifier paths for each item.

**Requirements:**
- Find shortest condition combination that uniquely identifies each item
- Prefer single conditions where available
- Fall back to pairs, then triples

**Output:**
| Item | Name | Shortest Unique Path |
|------|------|---------------------|
| 0 | Milkcrate Cafe | DriveThru=True |
| 1 | Tria Cafe | PriceRange=3 |
| 2 | Front Street | CoatCheck=True + FullBar=True |
```

### Analyze Feature Gaps

```
Analyze the condition matrix for coverage gaps.

**Check:**
1. Are all major attribute categories represented?
2. Are there features that no item satisfies?
3. Are there items with no unique identifier?
4. What combinations are needed for under-identified items?

**Output:**
- Missing features list
- Items without unique identifiers
- Suggested additional conditions to collect
```

---

## philly_cafes Actual Prompts

### Session 1: Initial Matrix Build

**Prompt:**
```
Build a complete condition matrix for the 20 selected restaurants.

Categories to analyze:
1. Boolean attributes: DriveThru, CoatCheck, HasTV, BikeParking, GoodForKids, etc.
2. WiFi: free, paid, no
3. Alcohol: none, beer_and_wine, full_bar
4. NoiseLevel: quiet, average, loud
5. RestaurantsPriceRange2: 1, 2, 3
6. Ambience: hipster, trendy, romantic, casual, etc.
7. Hours: open morning, afternoon, evening, late night for each day
8. Review keywords: common patterns in reviews

Output a matrix showing which restaurants satisfy each condition.
```

**Response Summary:**
```
CONDITION MATRIX EXCERPT:

Boolean Attributes:
| Condition | Count | Restaurants |
|-----------|-------|-------------|
| DriveThru=True | 1 | [0] Milkcrate |
| CoatCheck=True | 2 | [2] Front Street, [1] Tria |
| BYOB=True | 4 | [5], [11], [12], [3] |
| HasTV=False | 12 | [0], [2], [4], ... |

WiFi:
| free | 17 | All except [3], [14] |
| no | 2 | [3] MilkBoy, [14] La Colombe |

Noise:
| quiet | 4 | [7], [13], [15], [16] |
| loud | 2 | [3], [4] |
| average | 14 | Rest |

Ambience:
| hipster | 3 | [0], [2], [5] |
| trendy | 4 | [1], [5], [16], [17] |
```

### Session 2: Unique Identifier Discovery

**Prompt:**
```
From the condition matrix, find the shortest unique identifier for each restaurant.

Goal: For each of the 20 restaurants, find:
1. Single condition that uniquely identifies it (if exists)
2. Pair combination if no single unique
3. Triple combination if no pair unique

This will be used to design single-answer requests.
```

**Response Summary:**
```
UNIQUE IDENTIFIERS:

Single Unique:
[0] Milkcrate: DriveThru=True
[1] Tria: PriceRange=3

Pair Unique:
[2] Front Street: CoatCheck + FullBar
[3] MilkBoy: Loud + NoWiFi
[4] Kung Fu Tea: Loud + NoOutdoor
[5] Function: BYOB + Trendy
[7] Swiss Haus: Quiet + NoOutdoor
[13] Steap: Dogs + Quiet
[14] La Colombe: NoWiFi + NoOutdoor
[16] Elixr: Quiet + Trendy
[18] Chapterhouse: Dogs + NoOutdoor

Triple Unique (needed for remaining):
[6] Bubble House: FullBar + Kids + Outdoor
[9] Gran Caffe: FullBar + HappyHour + Outdoor
[10] Thirsty Dice: FullBar + Kids + NoOutdoor
...
```

### Session 3: Review Keyword Analysis

**Prompt:**
```
Search for unique keywords in the review text that appear in only 1-2 restaurants.

Focus on:
- Food/drink specific terms (gelato, matcha, banh mi)
- Experience terms (live music, board game, midnight)
- Descriptive terms (wine bar, dog friendly)

For each unique keyword:
- Pattern to match
- Restaurants where it appears
- Count of reviews mentioning it
```

**Response Summary:**
```
UNIQUE KEYWORDS:

Truly Unique (1 restaurant):
| Keyword | Restaurant | Count |
|---------|------------|-------|
| "live music" | [3] MilkBoy | 5 |
| "gelato" | [9] Gran Caffe | 8 |
| "wine bar" | [1] Tria | 12 |
| "challah" | [19] Sabrina's | 3 |
| "dog friendly" | [13] Steap | 4 |
| "banh mi" | [2] Front Street | 2 |

Rare (2 restaurants):
| Keyword | Restaurants | Count |
|---------|-------------|-------|
| "board game" | [10], [18] | 6, 4 |
| "bubble tea" | [4], [6] | 10, 8 |
| "latte art" | [14], [16] | 3, 5 |
```

### Session 4: Matrix Export

**Prompt:**
```
Export the condition matrix to data/philly_cafes/condition_matrix.json

Format:
{
  "conditions": {
    "drive_thru": {"kind": "item_meta", "path": [...], "true": "True"},
    ...
  },
  "matrix": {
    "drive_thru": [0],  // restaurant indices satisfying
    "wifi_free": [0, 1, 2, 4, 5, ...],
    ...
  },
  "unique_identifiers": {
    "0": ["drive_thru"],
    "1": ["price_3"],
    "2": ["coat_check", "full_bar"],
    ...
  }
}
```

**Response Summary:**
- Exported full condition matrix
- Included all 5 evidence type categories
- Documented unique identifier paths

---

## Usage Notes

### Condition Naming Convention

| Category | Naming Pattern | Example |
|----------|---------------|---------|
| Boolean true | `{attr}` | `drive_thru`, `bike_parking` |
| Boolean false | `no_{attr}` | `no_tv`, `no_dogs` |
| Enum value | `{attr}_{value}` | `wifi_free`, `noise_quiet` |
| Numeric | `price_{N}` | `price_1`, `price_3` |
| Hours | `hours_{day}_{period}` | `hours_monday_morning` |
| Review | `{pattern}_reviews` | `cozy_reviews`, `coffee_reviews` |
| Review meta | `{field}_{pattern}` | `elite_love`, `popular_work` |

### Building the Matrix

1. **Start with attributes**: Extract all Yelp attributes
2. **Add hours**: Parse hours for time availability
3. **Add review patterns**: Search for common keywords
4. **Classify**: Unique (1), rare (2-5), common (>5)
5. **Find identifiers**: Shortest path for each item

### Common Issues

- **Missing attributes**: Some items may have None for optional fields
- **Ambiguous patterns**: Review keywords may need regex refinement
- **No unique identifier**: May need to add more conditions or select different item
