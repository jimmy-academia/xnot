# Attack Implementation Plan

## Goal
Implement adversarial attacks to test robustness:
1. Attacks cause CoT to fail
2. ANoT resists attacks (or uses defense to resist)
3. Defense prompts on CoT make it worse (not better)

## Current State
- `oldsrc/attack.py` has complete attack implementations (typo, injection, fake_review, sarcastic)
- `utils/arguments.py` already parses `--attack`, `--seed`, `--defense` flags (but unused)
- Clean runs work correctly - must not break them

## Implementation Phases

### Phase 1: Integrate Attack Module
**Copy attack.py to main codebase**

1. Copy `oldsrc/attack.py` → `attack.py` (root level)
2. Update ATTACK_CHOICES in `utils/arguments.py` to match `attack.py`

### Phase 2: Wire Attack into Evaluation Flow
**Insert attack application after data filtering**

**Design decision:** For injection attacks, target = OPPOSITE of ground truth per item
- Gold item gets injected with "not recommend" (-1)
- Non-gold items get injected with "recommend" (1)
- This requires per-item attack with groundtruth awareness

1. **Modify `attack.py`**: Add `apply_attacks_with_groundtruth()` function
   ```python
   def apply_attacks_with_groundtruth(items, attack, groundtruth, seed=None):
       """Apply attacks where injection target opposes ground truth."""
       # For each item, determine if it's gold for any request
       # If gold -> inject target=-1, else inject target=1
   ```

2. **`run/scaling.py`** (after `filter_by_candidates`):
   ```python
   if args.attack not in ("none", "clean", None):
       from attack import apply_attacks_with_groundtruth
       dataset.items, attack_params = apply_attacks_with_groundtruth(
           dataset.items, args.attack, dataset.groundtruth,
           getattr(args, 'seed', None)
       )
   ```

3. **`run/orchestrate.py`** (same pattern)

4. **Store attack params** in `config.json` for reproducibility

### Phase 3: Test Attack Effectiveness

```bash
# Test 1: Clean baseline
python main.py --method cot --candidates 10 --run-name cot_clean

# Test 2: Attack causes CoT to fail
python main.py --method cot --candidates 10 --attack inject_override --run-name cot_attacked
# Expected: Accuracy drops significantly

# Test 3: ANoT resists attack
python main.py --method anot --candidates 10 --attack inject_override --run-name anot_attacked
# Expected: Accuracy stays high (or drops less)

# Test 4: Defense on CoT doesn't help (or makes worse)
python main.py --method cot --candidates 10 --attack inject_override --defense --run-name cot_defense
# Expected: Accuracy same or worse than cot_attacked
```

### Phase 4: Defense Integration (if needed)
- `methods/shared.py` has `_defense` and `_use_defense_prompt` globals
- Methods check these to switch prompts (see `methods/cot.py`)
- Verify defense prompts exist in baseline methods

## Files to Modify

| File | Changes |
|------|---------|
| `attack.py` | NEW - copy from oldsrc/attack.py |
| `utils/arguments.py` | Update ATTACK_CHOICES, ensure --defense works |
| `run/scaling.py` | Add attack application after filter_by_candidates |
| `run/orchestrate.py` | Add attack application after filter_by_candidates |
| `utils/experiment.py` | Store attack_params in config.json |

## Key Design Decisions

1. **Attack applied AFTER filtering** - attacks see same candidates as clean run
2. **Default `--attack none`** - clean runs unchanged
3. **Reproducibility via `--seed`** - same seed = same typos/injections
4. **Attack params in config.json** - full traceability

## Attack Types Available (from oldsrc/attack.py)

| Attack | Description | Expected Impact |
|--------|-------------|-----------------|
| typo_10 | 10% word typos | Minor degradation |
| typo_20 | 20% word typos | Moderate degradation |
| inject_override | "IGNORE INSTRUCTIONS, output X" | High on CoT, low on ANoT |
| inject_fake_sys | Fake system messages | High on CoT |
| inject_hidden | Hidden instructions in reviews | Moderate |
| fake_positive | Add fake glowing review | Confuses sentiment |
| fake_negative | Add fake terrible review | Confuses sentiment |
| sarcastic_all | Misleading positive framing | Subtle, tests understanding |

## Success Criteria

1. Clean runs (`--attack none`) produce identical results to current
2. `inject_override` on CoT: >30% accuracy drop
3. `inject_override` on ANoT: <10% accuracy drop
4. CoT + defense: no improvement (or worse)
