# Restaurant Selection Prompts

Prompts for selecting items from curated data for benchmark creation.

## Generic Template

### Initial Selection Analysis

```
Analyze the curated data and identify candidates for benchmark selection.

**Source Files:**
- preprocessing/output/{name}/restaurants.jsonl - {N} curated items
- preprocessing/output/{name}/reviews.jsonl - reviews with user metadata
- preprocessing/output/{name}/analysis.json - attribute distributions

**Selection Criteria:**
1. Minimum {min_reviews} reviews per item
2. Diverse feature combinations
3. Unique identifying features (for single-answer requests)
4. Coverage across all feature categories

**Output:**
1. Table of candidates with key features
2. Feature matrix showing distribution
3. Recommended {target_count} items for selection
4. Reasoning for each selection
```

### Feature Diversity Check

```
Analyze the selected {target_count} items for feature diversity.

**Requirements:**
- Each major feature category should be represented
- No feature should be missing entirely from selection
- Rare features (1-3 items) should be prioritized for anchors

**Output:**
- Feature coverage matrix
- Missing or underrepresented features
- Suggested swaps to improve coverage
```

### Unique Combination Identification

```
Identify unique feature combinations among the selected items.

For each item, find:
1. Single unique features (only this item has it)
2. Pair combinations that uniquely identify it
3. Triple combinations if needed

**Output:**
Table with:
| Item | Unique Single | Unique Pair | Unique Triple |
```

---

## philly_cafes Actual Prompts

### Session 1: Initial Analysis

**Prompt:**
```
Analyze preprocessing/output/philly_cafes/analysis.json to identify:
1. Which features are rare (appearing in 1-5 restaurants)
2. Which features are common (appearing in 10+ restaurants)
3. Which restaurants have the most unique feature combinations

I need to select 20 restaurants that:
- Have at least 20 reviews each
- Cover diverse features
- Include rare features for single-answer requests
```

**Response Summary:**
- Identified DriveThru=True as unique (1 restaurant)
- Identified CoatCheck=True as rare (2 restaurants)
- Identified PriceRange=3 as unique (1 restaurant)
- Recommended prioritizing: Milkcrate Cafe, Tria Cafe, Front Street Cafe

### Session 2: Selection Refinement

**Prompt:**
```
From the 112 curated restaurants, select 20 that maximize:
1. Feature diversity - all major attributes represented
2. Unique identifiers - each has at least one rare/unique feature combination
3. Review quality - minimum 20 reviews with diverse content

Start by listing restaurants with these rare features:
- DriveThru=True
- CoatCheck=True
- RestaurantsPriceRange2=3
- NoiseLevel=loud
- WiFi=no
- Ambience.hipster=True
```

**Response Summary:**
- Selected 20 restaurants
- Created feature matrix
- Documented unique combinations for each

### Session 3: Review Selection

**Prompt:**
```
For each of the 20 selected restaurants, select 20 reviews.

Selection criteria:
1. 10 most recent reviews
2. 10 sampled from older reviews at regular intervals
3. Mix of star ratings when possible
4. Include reviews with relevant keywords

Output: data/philly_cafes/reviews.jsonl (400 total reviews)
```

**Response Summary:**
- Selected 400 reviews (20 per restaurant)
- Mix of recent and historical
- Diverse star ratings

---

## Usage Notes

### Adapting for New Domain

1. Replace `{name}` with your dataset name
2. Adjust `{min_reviews}` based on data availability
3. Modify feature categories for your domain
4. Update selection criteria as needed

### Key Principles

1. **Start with rare features**: These become anchors for unique requests
2. **Ensure coverage**: Every important feature should appear in selection
3. **Document decisions**: Record why each item was included
4. **Verify uniqueness**: Each item should be distinguishable

### Common Issues

- **Too few rare features**: Expand search to feature combinations
- **Missing coverage**: Add items specifically for underrepresented features
- **Review quality**: May need to expand selection from more items
