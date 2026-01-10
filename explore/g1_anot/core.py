"""
G1a-ANoT: Adaptive Network of Thought for Peanut Allergy Safety Task

3-Phase LWT Architecture:
- Phase 1: Understand formula → Create LWT seed (cached per task)
- Phase 2: Apply seed to context → Form concrete script (per restaurant)
- Phase 3: Execute script → Compute primitives (DAG execution)
"""

import asyncio
import json
import math
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm import call_llm_async

from .prompts import PHASE1_PROMPT, PHASE2_PROMPT, EXTRACTION_PROMPT, AGGREGATION_PROMPT
from .tools import (
    keyword_search, get_review_metadata, get_cuisine_modifier,
    compute_incident_weight, compute_recency_decay, compute_risk_score,
    prepare_review_data, format_review_preview, ALLERGY_KEYWORDS
)
from .helpers import (
    parse_lwt_script, build_execution_layers, substitute_variables,
    safe_eval, format_output
)


class G1aANoT:
    """
    Adaptive Network of Thought adapted for G1a Peanut Allergy Safety task.

    Uses 3-phase LWT architecture:
    - Phase 1: Create reusable LWT seed from formula
    - Phase 2: Expand seed with restaurant-specific data
    - Phase 3: Execute script via DAG parallel execution
    """

    name = "g1a_anot"

    def __init__(self, task_prompt: str = None, verbose: bool = True, run_dir: str = None):
        """
        Initialize G1a-ANoT.

        Args:
            task_prompt: The G1a formula prompt (for Phase 1)
            verbose: Print debug info
            run_dir: Directory for trace logs
        """
        self.task_prompt = task_prompt
        self.verbose = verbose
        self.run_dir = run_dir

        # Cache for LWT seed (reusable across restaurants)
        self._lwt_seed = None
        self._seed_cache_key = None

    async def evaluate(self, restaurant: Dict) -> Dict[str, Any]:
        """
        Evaluate a single restaurant for peanut allergy safety.

        Args:
            restaurant: Dict with 'business' and 'reviews' keys

        Returns:
            Dict with all computed primitives matching TaskG1GroundTruth fields
        """
        business = restaurant.get('business', {})
        reviews = restaurant.get('reviews', [])
        name = business.get('name', 'Unknown')

        if self.verbose:
            print(f"\n[G1a-ANoT] Evaluating: {name}")

        # Simplified execution: Skip LWT generation, directly extract and compute
        result = await self._direct_extraction(restaurant)

        if self.verbose:
            print(f"[G1a-ANoT] Result: {result.get('VERDICT')} (score: {result.get('FINAL_RISK_SCORE')})")

        return result

    async def _direct_extraction(self, restaurant: Dict) -> Dict[str, Any]:
        """
        Direct extraction approach without LWT script generation.

        This is a simplified but effective approach:
        1. Find relevant reviews via keyword search
        2. Extract signals from each relevant review in parallel
        3. Aggregate and compute primitives

        Args:
            restaurant: Restaurant data

        Returns:
            Dict with computed primitives
        """
        business = restaurant.get('business', {})
        reviews = restaurant.get('reviews', [])
        categories = business.get('categories', '')

        # Step 1: Find allergy-relevant reviews
        relevant_indices = keyword_search(reviews)
        n_allergy_reviews = len(relevant_indices)

        if self.verbose:
            print(f"  Found {n_allergy_reviews} allergy-relevant reviews")

        # Step 2: Extract signals from each relevant review
        extractions = []
        if relevant_indices:
            tasks = [
                self._extract_review(reviews[idx], idx)
                for idx in relevant_indices
            ]
            extractions = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out errors
            extractions = [e for e in extractions if isinstance(e, dict)]

        # Step 3: Aggregate signals
        counts = self._aggregate_signals(extractions, reviews, relevant_indices)

        # Step 4: Compute derived values
        n_mild = counts['n_mild']
        n_moderate = counts['n_moderate']
        n_severe = counts['n_severe']
        n_total = counts['n_total']
        n_positive = counts['n_positive']
        n_negative = counts['n_negative']
        n_betrayal = counts['n_betrayal']

        # Incident score
        incident_score = (n_mild * 2) + (n_moderate * 5) + (n_severe * 15)

        # Recency decay
        if counts['incident_years']:
            most_recent_year = max(counts['incident_years'])
        else:
            most_recent_year = 2020
        recency_decay = compute_recency_decay(most_recent_year)

        # Credibility factor
        if counts['incident_weights']:
            credibility_factor = sum(counts['incident_weights']) / max(n_total, 1)
        else:
            credibility_factor = 1.0

        # Cuisine modifier
        cuisine_modifier = get_cuisine_modifier(categories)

        # Step 5: Final score
        final_score, verdict = compute_risk_score(
            n_mild, n_moderate, n_severe,
            n_positive, n_negative, n_betrayal,
            recency_decay, credibility_factor, cuisine_modifier,
            n_allergy_reviews
        )

        return {
            'N_TOTAL_INCIDENTS': n_total,
            'INCIDENT_SCORE': round(incident_score, 2),
            'RECENCY_DECAY': round(recency_decay, 2),
            'CREDIBILITY_FACTOR': round(credibility_factor, 2),
            'FINAL_RISK_SCORE': final_score,
            'VERDICT': verdict,
        }

    async def _extract_review(self, review: Dict, idx: int) -> Dict[str, Any]:
        """
        Extract allergy safety signals from a single review.

        Args:
            review: Review dict
            idx: Review index

        Returns:
            Dict with severity, account, interaction
        """
        meta = get_review_metadata(review)
        text = review.get('text', '')

        prompt = EXTRACTION_PROMPT.format(
            text=text[:1500],  # Truncate very long reviews
            stars=meta['stars'],
            date=meta['date'],
            useful=meta.get('useful', 0)
        )

        try:
            response = await call_llm_async(prompt, role="worker")

            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'idx': idx,
                    'severity': data.get('severity', 'none'),
                    'account': data.get('account', 'none'),
                    'interaction': data.get('interaction', 'none'),
                    'year': meta['year'],
                    'stars': meta['stars'],
                    'useful': meta.get('useful', 0),
                }
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to extract review {idx}: {e}")

        # Default to none
        return {
            'idx': idx,
            'severity': 'none',
            'account': 'none',
            'interaction': 'none',
            'year': meta['year'],
            'stars': meta['stars'],
            'useful': meta.get('useful', 0),
        }

    def _aggregate_signals(
        self,
        extractions: List[Dict],
        reviews: List[Dict],
        relevant_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Aggregate extraction results into counts.

        Args:
            extractions: List of extraction results
            reviews: All reviews
            relevant_indices: Indices of relevant reviews

        Returns:
            Dict with counts and metadata for computation
        """
        n_mild = n_moderate = n_severe = 0
        n_positive = n_negative = n_betrayal = 0
        incident_years = []
        incident_weights = []

        for ext in extractions:
            severity = ext.get('severity', 'none')
            account = ext.get('account', 'none')
            interaction = ext.get('interaction', 'none')

            # Count incidents (firsthand only)
            if account == 'firsthand' and severity != 'none':
                if severity == 'mild':
                    n_mild += 1
                elif severity == 'moderate':
                    n_moderate += 1
                elif severity == 'severe':
                    n_severe += 1

                # Track for recency and credibility
                incident_years.append(ext.get('year', 2020))
                weight = compute_incident_weight(
                    ext.get('stars', 3),
                    ext.get('useful', 0)
                )
                incident_weights.append(weight)

            # Count safety interactions (all account types)
            if interaction == 'positive':
                n_positive += 1
            elif interaction == 'negative':
                n_negative += 1
            elif interaction == 'betrayal':
                n_betrayal += 1

        n_total = n_mild + n_moderate + n_severe

        return {
            'n_mild': n_mild,
            'n_moderate': n_moderate,
            'n_severe': n_severe,
            'n_total': n_total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_betrayal': n_betrayal,
            'incident_years': incident_years,
            'incident_weights': incident_weights,
        }

    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format result as response string matching eval.py expectations.

        Args:
            result: Dict with computed primitives

        Returns:
            Formatted string with ===FINAL ANSWERS=== block
        """
        lines = ['===FINAL ANSWERS===']
        lines.append(f"N_TOTAL_INCIDENTS: {result.get('N_TOTAL_INCIDENTS', 0)}")
        lines.append(f"INCIDENT_SCORE: {result.get('INCIDENT_SCORE', 0.0)}")
        lines.append(f"RECENCY_DECAY: {result.get('RECENCY_DECAY', 0.3)}")
        lines.append(f"CREDIBILITY_FACTOR: {result.get('CREDIBILITY_FACTOR', 1.0)}")
        lines.append(f"FINAL_RISK_SCORE: {result.get('FINAL_RISK_SCORE', 2.75)}")
        lines.append(f"VERDICT: {result.get('VERDICT', 'Low Risk')}")
        lines.append('===END===')
        return '\n'.join(lines)


# Convenience function for standalone testing
async def evaluate_restaurant(restaurant: Dict, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate a restaurant using G1a-ANoT.

    Args:
        restaurant: Dict with 'business' and 'reviews' keys
        verbose: Print debug info

    Returns:
        Dict with computed primitives
    """
    anot = G1aANoT(verbose=verbose)
    return await anot.evaluate(restaurant)
