"""
Phase 2: Context-Focused Script Expansion

Takes Formula Seed (from Phase 1) and restaurant context, executes to produce results.

Input: Formula Seed + Restaurant data (business + reviews)
Output: Dict with computed primitives

Steps:
1. FILTER: Apply filter_keywords to find relevant reviews
2. EXTRACT: Run extraction_prompt on each relevant review (parallel LLM)
3. ASSEMBLE: Build arrays from extractions
4. COMPUTE: Execute computation graph in dependency order
5. OUTPUT: Return final computed values
"""

import asyncio
import json
import re
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm_async

try:
    from .phase1_step3 import FormulaSeed
except ImportError:
    from general_anot.phase1_step3 import FormulaSeed


@dataclass
class ExecutionContext:
    """Runtime context for script execution."""
    # Restaurant data
    business: Dict[str, Any] = field(default_factory=dict)
    reviews: List[Dict[str, Any]] = field(default_factory=list)

    # Filtered reviews (after keyword search)
    relevant_reviews: List[Dict[str, Any]] = field(default_factory=list)
    relevant_indices: List[int] = field(default_factory=list)

    # Extraction results
    extractions: List[Dict[str, Any]] = field(default_factory=list)

    # Assembled arrays (from extractions)
    arrays: Dict[str, List[Any]] = field(default_factory=dict)

    # Computed values
    values: Dict[str, Any] = field(default_factory=dict)

    # Lookup tables
    lookups: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Phase2Executor:
    """
    Phase 2: Execute Formula Seed against restaurant context.

    Usage:
        executor = Phase2Executor(seed, verbose=True)
        result = await executor.execute(restaurant)
    """

    def __init__(self, seed: FormulaSeed, verbose: bool = True):
        """
        Initialize executor with Formula Seed.

        Args:
            seed: Formula Seed from Phase 1
            verbose: Print debug info
        """
        self.seed = seed
        self.verbose = verbose

    async def execute(self, restaurant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Formula Seed against restaurant data.

        Args:
            restaurant: Dict with 'business' and 'reviews' keys

        Returns:
            Dict with computed primitives
        """
        # Initialize context
        ctx = ExecutionContext(
            business=restaurant.get('business', {}),
            reviews=restaurant.get('reviews', []),
        )

        if self.verbose:
            name = ctx.business.get('name', 'Unknown')
            print(f"\n[Phase 2] Executing for: {name}")
            print(f"  Total reviews: {len(ctx.reviews)}")

        # Step 1: Filter relevant reviews
        await self._filter_reviews(ctx)

        # Step 2: Extract signals from relevant reviews
        await self._extract_reviews(ctx)

        # Step 3: Assemble arrays from extractions
        self._assemble_arrays(ctx)

        # Step 4: Initialize lookups
        self._init_lookups(ctx)

        # Step 5: Execute computations in order
        self._execute_computations(ctx)

        # Step 6: Return output fields
        return self._get_outputs(ctx)

    async def _filter_reviews(self, ctx: ExecutionContext):
        """
        Step 1: Filter reviews using keywords.

        Modifies ctx.relevant_reviews and ctx.relevant_indices
        """
        keywords = [kw.lower() for kw in self.seed.filter_keywords]

        relevant_indices = []
        for idx, review in enumerate(ctx.reviews):
            text = review.get('text', '').lower()
            if any(kw in text for kw in keywords):
                relevant_indices.append(idx)

        ctx.relevant_indices = relevant_indices
        ctx.relevant_reviews = [ctx.reviews[i] for i in relevant_indices]

        if self.verbose:
            print(f"  Relevant reviews (keyword match): {len(ctx.relevant_reviews)}")

    async def _extract_reviews(self, ctx: ExecutionContext):
        """
        Step 2: Extract signals from each relevant review.

        Runs LLM extraction in parallel.
        Modifies ctx.extractions
        """
        if not ctx.relevant_reviews:
            ctx.extractions = []
            return

        # Build extraction tasks
        tasks = []
        for idx, review in zip(ctx.relevant_indices, ctx.relevant_reviews):
            tasks.append(self._extract_single_review(review, idx))

        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        extractions = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                extractions.append(result)
            elif isinstance(result, Exception):
                if self.verbose:
                    print(f"  Warning: Extraction error type={type(result).__name__}: {str(result)[:200]}")

        ctx.extractions = extractions

        if self.verbose:
            print(f"  Extracted: {len(ctx.extractions)} reviews")

    async def _extract_single_review(self, review: Dict, idx: int) -> Dict[str, Any]:
        """
        Extract signals from a single review using the extraction prompt.

        Returns dict with extracted fields + review metadata.
        """
        # Get review metadata
        text = review.get('text', '')[:2000]  # Truncate long reviews
        stars = review.get('stars', 3)
        date = review.get('date', '2020-01-01')
        useful = review.get('useful', 0)

        # Parse year from date
        try:
            if isinstance(date, str):
                year = int(date.split('-')[0])
            else:
                year = 2020
        except:
            year = 2020

        # Build prompt - use replace instead of format to avoid issues with JSON braces
        prompt = self.seed.extraction_prompt
        prompt = prompt.replace('{review_text}', text)
        prompt = prompt.replace('{review_date}', str(date))
        prompt = prompt.replace('{review_stars}', str(stars))
        prompt = prompt.replace('{review_useful}', str(useful))

        try:
            response = await call_llm_async(prompt, role="worker")


            # Parse JSON from response - handle multi-line
            # First try to find JSON code block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try to find raw JSON object (handle nested braces)
                start = response.find('{')
                end = response.rfind('}')
                if start >= 0 and end > start:
                    data = json.loads(response[start:end+1])
                else:
                    data = None

            if data:

                # Add metadata
                data['_idx'] = idx
                data['_year'] = year
                data['_stars'] = stars
                data['_useful'] = useful

                # Normalize field names
                result = {'_idx': idx, '_year': year, '_stars': stars, '_useful': useful}
                for field in self.seed.extraction_fields:
                    name = field.name
                    # Try different key formats
                    value = data.get(name) or data.get(name.upper()) or data.get(name.lower())
                    if value is None:
                        # Use default value
                        value = field.values[0] if field.values else 'none'
                    result[name] = value

                return result

        except Exception as e:
            if self.verbose:
                print(f"    Error extracting review {idx}: {e}")

        # Return defaults on error
        result = {'_idx': idx, '_year': year, '_stars': stars, '_useful': useful}
        for field in self.seed.extraction_fields:
            result[field.name] = field.values[0] if field.values else 'none'
        return result

    def _assemble_arrays(self, ctx: ExecutionContext):
        """
        Step 3: Assemble arrays from extractions.

        Modifies ctx.arrays
        """
        for assembly in self.seed.array_assemblies:
            arr = []
            source_field = assembly.source_field

            for ext in ctx.extractions:
                # Handle metadata fields (stars, useful, year)
                if source_field == 'stars':
                    arr.append(ext.get('_stars', 3))
                elif source_field == 'useful':
                    arr.append(ext.get('_useful', 0))
                elif source_field == 'year':
                    arr.append(ext.get('_year', 2020))
                else:
                    # Regular extraction field
                    arr.append(ext.get(source_field, 'none'))

            ctx.arrays[assembly.array_name] = arr

        if self.verbose:
            for name, arr in ctx.arrays.items():
                print(f"  Array {name}: {len(arr)} items")

    def _init_lookups(self, ctx: ExecutionContext):
        """
        Step 4: Initialize lookup tables.

        Modifies ctx.lookups and ctx.values (for lookup-derived values)
        """
        for lookup in self.seed.lookups:
            # Store the mapping as a lookup table
            ctx.lookups[lookup.name] = {
                'mapping': lookup.mapping,
                'default': lookup.default,
                'source_field': lookup.source_field,
            }

            # Also compute the lookup value
            source_field = lookup.source_field
            if source_field == 'restaurant_category':
                categories = ctx.business.get('categories', '')
                # Find matching category
                value = lookup.default
                if isinstance(categories, str):
                    for cat, mod in lookup.mapping.items():
                        if cat.lower() in categories.lower():
                            value = mod
                            break
                ctx.values[lookup.name] = value
            else:
                ctx.values[lookup.name] = lookup.default

        if self.verbose:
            for name, val in ctx.values.items():
                if name.endswith('_lookup'):
                    print(f"  Lookup {name}: {val}")

    def _execute_computations(self, ctx: ExecutionContext):
        """
        Step 5: Execute computation graph in dependency order.

        Modifies ctx.values
        """
        # Put extractions and arrays into context values
        ctx.values['extractions'] = ctx.extractions
        ctx.values.update(ctx.arrays)

        # Add math functions to eval context
        eval_context = {
            'math': math,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'zip': zip,
            'abs': abs,
            'round': round,
            'True': True,
            'False': False,
        }

        for comp in self.seed.computations:
            try:
                # Build local context with dependencies
                local_ctx = dict(eval_context)
                local_ctx.update(ctx.values)

                # Also add lookup tables
                for lookup_name, lookup_data in ctx.lookups.items():
                    local_ctx[lookup_name] = lookup_data['mapping']
                    # Add a .get method wrapper
                    class LookupWrapper:
                        def __init__(self, mapping, default):
                            self._mapping = mapping
                            self._default = default
                        def get(self, key, default=None):
                            return self._mapping.get(key, default or self._default)
                    local_ctx[lookup_name] = LookupWrapper(
                        lookup_data['mapping'],
                        lookup_data['default']
                    )

                # Add restaurant_category for cuisine lookup
                local_ctx['restaurant_category'] = ctx.business.get('categories', '')

                # Evaluate formula
                formula = comp.formula

                # Handle cuisine_modifier_lookup.get() specially
                if 'cuisine_modifier_lookup.get' in formula:
                    # The lookup returns the first matching category's value
                    categories = ctx.business.get('categories', '')
                    value = 1.0  # default
                    if 'cuisine_modifier_lookup' in ctx.lookups:
                        mapping = ctx.lookups['cuisine_modifier_lookup']['mapping']
                        for cat, mod in mapping.items():
                            if isinstance(categories, str) and cat.lower() in categories.lower():
                                value = mod
                                break
                    ctx.values[comp.name] = value
                    continue

                # Move math functions to globals for generator expression access
                eval_globals = {"__builtins__": {}}
                eval_globals.update(eval_context)
                result = eval(formula, eval_globals, local_ctx)
                ctx.values[comp.name] = result

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Error computing {comp.name}: {e}")
                    print(f"    Formula: {comp.formula}")
                # Set default value
                if comp.is_aggregate:
                    ctx.values[comp.name] = 0
                else:
                    ctx.values[comp.name] = 0.0

        if self.verbose:
            # Print key computed values
            key_fields = ['n_mild', 'n_moderate', 'n_severe', 'n_total_incidents',
                         'incident_score', 'final_risk_score', 'verdict']
            for field in key_fields:
                if field in ctx.values:
                    print(f"  {field}: {ctx.values[field]}")

    def _get_outputs(self, ctx: ExecutionContext) -> Dict[str, Any]:
        """
        Step 6: Return output fields.
        """
        result = {}
        for field in self.seed.output_fields:
            # Try different key formats (0 is a valid value, so use 'in' check)
            value = None
            if field in ctx.values:
                value = ctx.values[field]
            elif field.lower() in ctx.values:
                value = ctx.values[field.lower()]
            elif field.upper() in ctx.values:
                value = ctx.values[field.upper()]

            if value is not None:
                # Convert to uppercase for eval.py compatibility
                result[field.upper()] = value

        # Ensure verdict is present
        if 'VERDICT' not in result and 'verdict' in ctx.values:
            result['VERDICT'] = ctx.values['verdict']

        return result


async def execute_seed(seed: FormulaSeed, restaurant: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to execute Formula Seed against restaurant.

    Args:
        seed: Formula Seed from Phase 1
        restaurant: Dict with 'business' and 'reviews'
        verbose: Print debug info

    Returns:
        Dict with computed primitives
    """
    executor = Phase2Executor(seed, verbose=verbose)
    return await executor.execute(restaurant)


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    # Test Phase 2 with saved seed and sample restaurant
    async def test():
        # Load seed
        seed_file = Path(__file__).parent.parent / "results" / "phase1_steps" / "step1_3_formula_seed_v2.json"
        with open(seed_file) as f:
            seed = FormulaSeed.from_dict(json.load(f))

        # Load sample restaurant
        data_dir = Path(__file__).parent.parent.parent / "data" / "philly_cafes"
        restaurants_file = data_dir / "restaurants.jsonl"
        reviews_file = data_dir / "reviews.jsonl"

        # Read first restaurant
        with open(restaurants_file) as f:
            business = json.loads(f.readline())

        # Read all reviews for this restaurant
        business_id = business.get('business_id')
        reviews = []
        with open(reviews_file) as f:
            for line in f:
                review = json.loads(line)
                if review.get('business_id') == business_id:
                    reviews.append(review)

        restaurant = {'business': business, 'reviews': reviews}

        print("="*70)
        print(f"Testing Phase 2 on: {business.get('name')}")
        print("="*70)

        result = await execute_seed(seed, restaurant, verbose=True)

        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        for k, v in result.items():
            print(f"  {k}: {v}")

    asyncio.run(test())
