# Philly Cafes Reference Scripts

One-time generation scripts used to build the `philly_cafes` benchmark. Kept as reference for creating similar benchmarks on new datasets.

## Pipeline Order

Scripts were executed roughly in this order:

1. **expand_dataset.py** - Expand from initial selection to 50 restaurants
   - Input: `preprocessing/output/`
   - Output: `restaurants.jsonl`, `reviews.jsonl`

2. **harden_requests.py** - Add constraints to make requests harder
   - Input: `requests.jsonl`
   - Output: Updated `requests.jsonl` with more conditions

3. **condition_matrix.py** - Build lookup matrix of condition satisfaction
   - Input: `restaurants.jsonl`, `reviews.jsonl`
   - Output: `condition_matrix.json`

4. **generate_g05_g08.py** - Generate nested AND/OR structures (G05-G08)
   - Input: `condition_matrix.json`
   - Output: Requests R41-R80

5. **redistribute_g06_g07.py** - (Optional) Redistribute for scaling experiments
   - Ensures G06/G07 use top-20 restaurants

6. **fix_g01_requests.py** - Utility to regenerate request text from conditions
   - Useful when condition definitions change

7. **generate_social_mapping.py** - Generate social graph for G09/G10
   - Output: `user_mapping.json`

## Notes

- These scripts are **philly_cafes-specific** (hardcoded paths, attribute mappings)
- Adapt condition definitions and paths for new datasets
- Run `data/validate.py` after generation to verify 100% validation
