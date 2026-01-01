# Baseline Benchmark Results

Benchmark results on yelp dataset (selection_1, k=5).

## Results Summary

| Method       | Hits@1 | Hits@5 | Category |
|--------------|--------|--------|----------|
| ps           | 0%     | 30%    | Reasoning |
| cot          | 12%    | 27%    | Reasoning |
| decomp       | 10%    | 25%    | Reasoning |
| cotsc        | 5%     | 25%    | Reasoning |
| prp          | 0%     | 25%    | Ranking |
| listwise     | 0%     | 25%    | Ranking |
| finegrained  | 0%     | 25%    | Ranking |
| setwise      | 0%     | 20%    | Ranking |
| parade       | 0%     | 20%    | Ranking |
| pal          | 5%     | 20%    | Structured |
| pot          | 8%     | 18%    | Structured |
| cot_table    | 2%     | 18%    | Structured |
| rankgpt      | 5%     | 15%    | Ranking |
| react        | 10%    | 15%    | Reasoning |
| weaver       | 5%     | 15%    | Structured |
| selfask      | 10%    | 10%    | Reasoning |
| l2m          | 0%     | 10%    | Reasoning |

## Key Findings

- **Best performer**: ps (Plan-and-Solve) at 30% Hits@5
- **Reasoning methods** (cot, cotsc, ps, decomp): 10-30%
- **Ranking methods** (prp, listwise, setwise, parade): 20-25%
- **Structured-input methods** (pal, pot, cot_table, weaver): 15-20%

## Notes

- Model: gpt-5-nano
- Dataset: yelp (20 samples)
- Selection: selection_1
- anot excluded (0%, needs fix)
