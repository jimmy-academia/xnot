"""
Phase 1, Step 1.3: Generate Formula Seed

Combines outputs from Step 1.1 and 1.2 into a complete Formula Seed.

Key responsibilities:
1. Normalize variable names (fix case inconsistency)
2. Specify extraction→array assembly (the missing link)
3. Generate complete Formula Seed with Compute DAG

Input: ExtractionConditions (1.1) + ComputationGraph (1.2)
Output: FormulaSeed (complete executable specification)
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .phase1_step1 import ExtractionConditions, ExtractionField, AggregationCondition
    from .phase1_step2 import ComputationGraph, ComputationStep, LookupTable
except ImportError:
    from general_anot.phase1_step1 import ExtractionConditions, ExtractionField, AggregationCondition
    from general_anot.phase1_step2 import ComputationGraph, ComputationStep, LookupTable


@dataclass
class ArrayAssembly:
    """Specification for assembling an array from extractions."""
    array_name: str              # e.g., "incident_severities"
    source_field: str            # e.g., "incident_severity" (from extraction)
    filter_condition: str = ""   # Optional filter, e.g., "account_type == 'firsthand'"
    include_metadata: bool = False  # If True, also collect review metadata


@dataclass
class FormulaSeed:
    """
    Complete Formula Seed for Phase 2 execution.

    This is the output of Phase 1 - a template that Phase 2 adapts per restaurant.
    """
    # Task info
    task_name: str

    # Filter: How to find relevant reviews
    filter_keywords: List[str] = field(default_factory=list)

    # Extraction: What to extract from each review
    extraction_fields: List[ExtractionField] = field(default_factory=list)
    extraction_prompt: str = ""  # Complete prompt template for LLM extraction

    # Array Assembly: How extractions become computation inputs
    array_assemblies: List[ArrayAssembly] = field(default_factory=list)

    # Lookups: Static tables
    lookups: List[LookupTable] = field(default_factory=list)

    # Computation: Ordered COMPUTE steps (using normalized names)
    computations: List[ComputationStep] = field(default_factory=list)

    # Output
    output_fields: List[str] = field(default_factory=list)

    # Variable name mapping (original -> normalized)
    name_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormulaSeed':
        """Create FormulaSeed from dictionary."""
        # Reconstruct nested dataclasses
        extraction_fields = [
            ExtractionField(**f) for f in data.get('extraction_fields', [])
        ]
        array_assemblies = [
            ArrayAssembly(**a) for a in data.get('array_assemblies', [])
        ]
        lookups = [
            LookupTable(**l) for l in data.get('lookups', [])
        ]
        computations = [
            ComputationStep(**c) for c in data.get('computations', [])
        ]

        return cls(
            task_name=data.get('task_name', ''),
            filter_keywords=data.get('filter_keywords', []),
            extraction_fields=extraction_fields,
            extraction_prompt=data.get('extraction_prompt', ''),
            array_assemblies=array_assemblies,
            lookups=lookups,
            computations=computations,
            output_fields=data.get('output_fields', []),
            name_map=data.get('name_map', {}),
        )

    def to_script(self) -> str:
        """Generate a script representation of the Formula Seed."""
        lines = []
        lines.append("# === FORMULA SEED SCRIPT ===")
        lines.append(f"# Task: {self.task_name}")
        lines.append("")

        # Section 1: Filter
        lines.append("# --- FILTER RELEVANT REVIEWS ---")
        lines.append(f"(filter_keywords)=CONST({json.dumps(self.filter_keywords)})")
        lines.append("(relevant_reviews)=TOOL('keyword_filter', reviews={(context)}[reviews], keywords={(filter_keywords)})")
        lines.append("")

        # Section 2: Extraction (template - Phase 2 expands)
        lines.append("# --- EXTRACTION (Phase 2 expands per review) ---")
        lines.append("# @FOREACH review_idx IN range(len(relevant_reviews)):")
        lines.append("#   (extract_{review_idx})=LLM(extraction_prompt.format(review=relevant_reviews[review_idx]))")
        lines.append("# @END_FOREACH")
        lines.append(f"# Extraction prompt stored in: extraction_prompt")
        lines.append("")

        # Section 3: Array Assembly
        lines.append("# --- ARRAY ASSEMBLY ---")
        for aa in self.array_assemblies:
            if aa.filter_condition:
                lines.append(f"({aa.array_name})=ASSEMBLE(extractions, field='{aa.source_field}', filter='{aa.filter_condition}')")
            else:
                lines.append(f"({aa.array_name})=ASSEMBLE(extractions, field='{aa.source_field}')")
        lines.append("")

        # Section 4: Lookups
        if self.lookups:
            lines.append("# --- LOOKUPS ---")
            for lookup in self.lookups:
                lines.append(f"({lookup.name})=CONST({json.dumps(lookup.mapping)})")
            lines.append("")

        # Section 5: Computations
        lines.append("# --- COMPUTATIONS ---")
        for comp in self.computations:
            lines.append(f"({comp.name})=COMPUTE('{comp.formula}')")
        lines.append("")

        # Section 6: Output
        lines.append("# --- OUTPUT ---")
        lines.append(f"# Output fields: {self.output_fields}")

        return "\n".join(lines)


class Step3GenerateSeed:
    """
    Step 1.3: Generate Formula Seed from extraction conditions and computation graph.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(
        self,
        task_name: str,
        extraction_conditions: ExtractionConditions,
        computation_graph: ComputationGraph
    ) -> FormulaSeed:
        """
        Generate Formula Seed.

        Args:
            task_name: Name of the task
            extraction_conditions: Output from Step 1.1
            computation_graph: Output from Step 1.2

        Returns:
            FormulaSeed ready for Phase 2 execution
        """
        if self.verbose:
            print("[Step 1.3] Generating Formula Seed...")

        # Step 1: Identify true extraction fields (filter out metadata)
        true_extraction_fields = self._filter_extraction_fields(extraction_conditions)

        # Step 2: Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(true_extraction_fields)

        # Step 3: Build array assemblies from extraction fields
        array_assemblies = self._build_array_assemblies(true_extraction_fields)

        # Step 4: Generate aggregation formulas from Step 1.1 conditions
        aggregation_computations = self._build_aggregation_computations(
            extraction_conditions.aggregation_conditions
        )

        # Step 5: Normalize and merge computations
        # - Use aggregation_computations for aggregates
        # - Use computation_graph for derived values
        name_map, final_computations = self._merge_computations(
            aggregation_computations,
            computation_graph
        )

        # Step 6: Build the seed
        seed = FormulaSeed(
            task_name=task_name,
            filter_keywords=extraction_conditions.filter_condition.keywords if extraction_conditions.filter_condition else [],
            extraction_fields=true_extraction_fields,
            extraction_prompt=extraction_prompt,
            array_assemblies=array_assemblies,
            lookups=computation_graph.lookups,
            computations=final_computations,
            output_fields=computation_graph.output_fields,
            name_map=name_map,
        )

        if self.verbose:
            print(f"[Step 1.3] Generated seed:")
            print(f"  - {len(seed.filter_keywords)} filter keywords")
            print(f"  - {len(seed.extraction_fields)} extraction fields")
            print(f"  - {len(seed.array_assemblies)} array assemblies")
            print(f"  - {len(seed.lookups)} lookups")
            print(f"  - {len(seed.computations)} computations")
            print(f"  - {len(seed.output_fields)} output fields")

        return seed

    def _filter_extraction_fields(self, conditions: ExtractionConditions) -> List[ExtractionField]:
        """
        Filter out metadata fields, keep only fields requiring LLM extraction.

        Metadata fields (already in review data): stars, date, useful, year
        """
        metadata_names = {'stars', 'date', 'useful', 'year', 'review_date', 'restaurant_category'}

        true_fields = []
        for ef in conditions.extraction_fields:
            if ef.name.lower() not in metadata_names:
                # Also filter out redundant boolean fields derived from filtering
                if ef.name != 'mentions_allergy' and ef.field_type != 'string':
                    true_fields.append(ef)

        return true_fields

    def _build_extraction_prompt(self, fields: List[ExtractionField]) -> str:
        """Build the extraction prompt template for LLM."""
        lines = []
        lines.append("Analyze this restaurant review and extract the following signals.")
        lines.append("")
        lines.append("REVIEW:")
        lines.append("{review_text}")
        lines.append("")
        lines.append("REVIEW METADATA:")
        lines.append("- Date: {review_date}")
        lines.append("- Stars: {review_stars}")
        lines.append("- Useful votes: {review_useful}")
        lines.append("")
        lines.append("EXTRACT THESE FIELDS:")
        lines.append("")

        for ef in fields:
            lines.append(f"**{ef.name}**")
            if ef.field_type == "enum":
                lines.append(f"  Type: Choose ONE of {ef.values}")
            else:
                lines.append(f"  Type: {ef.field_type}")
            if ef.description:
                lines.append(f"  Description: {ef.description}")
            if ef.extraction_hint:
                lines.append(f"  Hints: {ef.extraction_hint}")
            lines.append("")

        lines.append("OUTPUT FORMAT:")
        lines.append("Return a JSON object with these exact keys:")
        lines.append("{")
        for i, ef in enumerate(fields):
            comma = "," if i < len(fields) - 1 else ""
            if ef.field_type == "enum":
                lines.append(f'  "{ef.name}": "<one of {ef.values}>"{comma}')
            else:
                lines.append(f'  "{ef.name}": <{ef.field_type}>{comma}')
        lines.append("}")
        lines.append("")
        lines.append("If a field cannot be determined from the review, use the default:")
        for ef in fields:
            if ef.field_type == "enum" and "none" in [v.lower() for v in ef.values]:
                lines.append(f'  {ef.name}: "none"')
            elif ef.field_type == "boolean":
                lines.append(f'  {ef.name}: false')
            elif ef.field_type == "number":
                lines.append(f'  {ef.name}: 0')
            else:
                lines.append(f'  {ef.name}: ""')

        return "\n".join(lines)

    def _normalize_names(self, graph: ComputationGraph) -> tuple:
        """
        Normalize all variable names to lowercase_snake_case.

        Returns:
            (name_map, normalized_computations)
        """
        name_map = {}

        # Collect all variable names
        all_names: Set[str] = set()
        for comp in graph.computations:
            all_names.add(comp.name)
            all_names.update(comp.depends_on)
        for lookup in graph.lookups:
            all_names.add(lookup.name)

        # Build normalization map
        for name in all_names:
            normalized = self._to_snake_case(name)
            if normalized != name:
                name_map[name] = normalized

        # Apply normalization to computations
        normalized_computations = []
        for comp in graph.computations:
            new_formula = comp.formula
            new_depends = []

            # Replace variable names in formula and depends_on
            for old_name, new_name in name_map.items():
                # Use word boundary replacement
                new_formula = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, new_formula)

            for dep in comp.depends_on:
                new_depends.append(name_map.get(dep, dep))

            normalized_computations.append(ComputationStep(
                name=name_map.get(comp.name, comp.name),
                formula=new_formula,
                depends_on=new_depends,
                description=comp.description,
                is_aggregate=comp.is_aggregate,
                is_output=comp.is_output,
            ))

        return name_map, normalized_computations

    def _to_snake_case(self, name: str) -> str:
        """Convert name to lowercase_snake_case."""
        # Insert underscore before uppercase letters (except at start)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def _build_array_assemblies(self, extraction_fields: List[ExtractionField]) -> List[ArrayAssembly]:
        """
        Build array assemblies from extraction fields.

        Each extraction field becomes an array that collects values from all extractions.
        """
        assemblies = []

        for ef in extraction_fields:
            # Create array name (plural form)
            if ef.name.endswith('y'):
                array_name = ef.name[:-1] + 'ies'
            elif ef.name.endswith('s'):
                array_name = ef.name + 'es'
            else:
                array_name = ef.name + 's'

            assemblies.append(ArrayAssembly(
                array_name=array_name,
                source_field=ef.name,
            ))

        # Also add metadata arrays needed for computation
        metadata_arrays = [
            ArrayAssembly(array_name='review_stars', source_field='stars'),
            ArrayAssembly(array_name='review_useful', source_field='useful'),
            ArrayAssembly(array_name='review_years', source_field='year'),
        ]
        assemblies.extend(metadata_arrays)

        return assemblies

    def _build_aggregation_computations(
        self,
        aggregation_conditions: List
    ) -> List[ComputationStep]:
        """
        Build aggregation computations from Step 1.1 conditions.

        This generates the actual aggregation formulas that count/sum extractions.
        """
        computations = []

        for ac in aggregation_conditions:
            name = ac.name.lower()

            # Build the aggregation formula based on conditions
            if ac.agg_type == 'count':
                if ac.conditions:
                    # Build condition string
                    cond_parts = []
                    for field, value in ac.conditions.items():
                        cond_parts.append(f'e["{field}"] == "{value}"')
                    cond_str = ' and '.join(cond_parts)
                    formula = f'sum(1 for e in extractions if {cond_str})'
                else:
                    formula = 'len(extractions)'
            elif ac.agg_type == 'sum':
                # For n_total_incidents, it's a derived sum, not aggregation
                # Skip it here, let it come from computation graph
                continue
            elif ac.agg_type == 'max':
                formula = f'max((e["{list(ac.conditions.keys())[0]}"] for e in extractions), default=0)'
            elif ac.agg_type == 'min':
                formula = f'min((e["{list(ac.conditions.keys())[0]}"] for e in extractions), default=0)'
            else:
                formula = '0'  # Fallback

            computations.append(ComputationStep(
                name=name,
                formula=formula,
                depends_on=['extractions'],
                description=ac.description,
                is_aggregate=True,
                is_output=False,
            ))

        return computations

    def _merge_computations(
        self,
        aggregation_computations: List[ComputationStep],
        computation_graph: ComputationGraph
    ) -> tuple:
        """
        Merge aggregation computations with computation graph.

        - Use aggregation_computations for aggregate values
        - Use computation_graph for derived values
        - Normalize all names
        """
        name_map = {}
        final_computations = []

        # Create lookup of aggregation computations by name
        agg_by_name = {c.name: c for c in aggregation_computations}

        # First pass: collect all variable names and build normalization map
        all_names = set()
        for comp in computation_graph.computations:
            all_names.add(comp.name)
            all_names.update(comp.depends_on)
            # Also find uppercase words in formulas
            uppercase_vars = re.findall(r'\b[A-Z][A-Z_]+\b', comp.formula)
            all_names.update(uppercase_vars)

        for name in all_names:
            normalized = self._to_snake_case(name)
            if normalized != name:
                name_map[name] = normalized

        # Also add array name fixes (incident_* -> review_* for metadata)
        array_fixes = {
            'incident_years': 'review_years',
            'incident_stars': 'review_stars',
            'incident_useful': 'review_useful',
        }
        name_map.update(array_fixes)

        # Process computation graph
        for comp in computation_graph.computations:
            normalized_name = self._to_snake_case(comp.name)

            # If this is an aggregate and we have a proper formula from Step 1.1, use it
            if comp.is_aggregate and normalized_name in agg_by_name:
                agg_comp = agg_by_name[normalized_name]
                # Special case: n_allergy_reviews should count all extractions (filtered = relevant)
                formula = agg_comp.formula
                if normalized_name == 'n_allergy_reviews':
                    formula = 'len(extractions)'

                final_computations.append(ComputationStep(
                    name=normalized_name,
                    formula=formula,
                    depends_on=['extractions'],
                    description=agg_comp.description or comp.description,
                    is_aggregate=True,
                    is_output=comp.is_output,
                ))
            else:
                # Use computation graph formula, but normalize ALL variable names
                formula = comp.formula

                # Apply all name normalizations to formula
                for old_name, new_name in name_map.items():
                    formula = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, formula)

                # Normalize dependencies
                depends_on = [name_map.get(dep, self._to_snake_case(dep)) for dep in comp.depends_on]

                final_computations.append(ComputationStep(
                    name=normalized_name,
                    formula=formula,
                    depends_on=depends_on,
                    description=comp.description,
                    is_aggregate=comp.is_aggregate,
                    is_output=comp.is_output,
                ))

        return name_map, final_computations

    def _infer_array_assemblies(
        self,
        extraction_fields: List[ExtractionField],
        computations: List[ComputationStep]
    ) -> List[ArrayAssembly]:
        """
        Infer what arrays need to be assembled from extractions.

        Look at aggregation formulas to determine what arrays they expect.
        """
        assemblies = []

        # Common patterns: field_name -> array_name (plural form)
        field_to_array = {}
        for ef in extraction_fields:
            # incident_severity -> incident_severities
            array_name = ef.name + ('es' if ef.name.endswith('y') else 's')
            if ef.name.endswith('y'):
                array_name = ef.name[:-1] + 'ies'
            field_to_array[ef.name] = array_name

        # Check which arrays are actually used in computations
        used_arrays = set()
        for comp in computations:
            if comp.is_aggregate:
                # Look for array references in formula
                for field_name, array_name in field_to_array.items():
                    if array_name in comp.formula:
                        used_arrays.add((field_name, array_name))

                # Also check for common patterns like "safety_interactions"
                if 'safety_interactions' in comp.formula:
                    used_arrays.add(('safety_interaction', 'safety_interactions'))
                if 'account_types' in comp.formula:
                    used_arrays.add(('account_type', 'account_types'))
                if 'incident_severities' in comp.formula:
                    used_arrays.add(('incident_severity', 'incident_severities'))

        # Also need metadata arrays for some computations
        metadata_arrays = [
            ('incident_stars', 'stars', "has_incident == True"),
            ('incident_useful', 'useful', "has_incident == True"),
            ('incident_years', 'year', "has_incident == True"),
        ]

        for comp in computations:
            for array_name, source, filter_cond in metadata_arrays:
                if array_name in comp.formula:
                    assemblies.append(ArrayAssembly(
                        array_name=array_name,
                        source_field=source,
                        filter_condition=filter_cond,
                    ))

        # Add extraction field arrays
        for field_name, array_name in used_arrays:
            assemblies.append(ArrayAssembly(
                array_name=array_name,
                source_field=field_name,
            ))

        return assemblies


def generate_formula_seed(
    task_name: str,
    extraction_conditions: ExtractionConditions,
    computation_graph: ComputationGraph,
    verbose: bool = True
) -> FormulaSeed:
    """Convenience function to run Step 1.3."""
    step = Step3GenerateSeed(verbose=verbose)
    return step.run(task_name, extraction_conditions, computation_graph)


# Test
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from g1_allergy import TASK_G1_PROMPT
    from utils.llm import configure
    from general_anot.phase1_step1 import extract_conditions
    from general_anot.phase1_step2 import build_computation_graph

    async def test():
        configure(temperature=0.0)

        print("="*70)
        print("STEP 1.1: EXTRACT CONDITIONS")
        print("="*70)
        conditions = await extract_conditions(TASK_G1_PROMPT, verbose=True)

        print("\n" + "="*70)
        print("STEP 1.2: BUILD COMPUTATION GRAPH")
        print("="*70)
        graph = await build_computation_graph(TASK_G1_PROMPT, conditions, verbose=True)

        print("\n" + "="*70)
        print("STEP 1.3: GENERATE FORMULA SEED")
        print("="*70)
        seed = generate_formula_seed("G1a", conditions, graph, verbose=True)

        # Save outputs
        output_dir = Path(__file__).parent.parent / "results" / "phase1_steps"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save seed JSON
        with open(output_dir / "step1_3_formula_seed.json", 'w') as f:
            f.write(seed.to_json())

        # Save script representation
        with open(output_dir / "step1_3_formula_script.txt", 'w') as f:
            f.write(seed.to_script())

        print(f"\nSaved to: {output_dir}")

        # Print key outputs
        print("\n" + "="*70)
        print("EXTRACTION FIELDS (filtered to true LLM extractions):")
        print("="*70)
        for ef in seed.extraction_fields:
            print(f"  {ef.name}: {ef.field_type} = {ef.values}")

        print("\n" + "="*70)
        print("ARRAY ASSEMBLIES:")
        print("="*70)
        for aa in seed.array_assemblies:
            filter_str = f" WHERE {aa.filter_condition}" if aa.filter_condition else ""
            print(f"  {aa.array_name} <- extraction['{aa.source_field}']{filter_str}")

        print("\n" + "="*70)
        print("NAME NORMALIZATIONS:")
        print("="*70)
        for old, new in seed.name_map.items():
            print(f"  {old} -> {new}")

        print("\n" + "="*70)
        print("EXTRACTION PROMPT:")
        print("="*70)
        print(seed.extraction_prompt[:1000] + "..." if len(seed.extraction_prompt) > 1000 else seed.extraction_prompt)

        print("\n" + "="*70)
        print("FORMULA SCRIPT:")
        print("="*70)
        print(seed.to_script())

        return seed

    asyncio.run(test())
