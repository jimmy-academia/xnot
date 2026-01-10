"""
Helpers for G1a-ANoT LWT parsing and DAG execution.

LWT Format:
    (step_id)=INSTRUCTION_TYPE('parameters')

Instruction Types:
    - TOOL('name', args)    - Execute tool function
    - LLM('prompt')         - Call LLM
    - COMPUTE('expression') - Execute Python expression

Variable References:
    - {(step_id)} - Reference output from previous step
    - {(context)} - Reference restaurant data
"""

import re
import json
import math
from typing import List, Dict, Any, Tuple, Set


# Pattern for parsing LWT lines
LWT_PATTERN = re.compile(r'^\((\w+)\)=(\w+)\((.+)\)$')

# Pattern for variable references in strings
VAR_PATTERN = re.compile(r'\{?\(\((\w+)\)\)\}?|\{\((\w+)\)\}')


def parse_lwt_line(line: str) -> Tuple[str, str, str]:
    """
    Parse a single LWT line.

    Args:
        line: LWT line like "(step_id)=INSTRUCTION('params')"

    Returns:
        (step_id, instruction_type, parameters)

    Raises:
        ValueError if line doesn't match expected format
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None, None

    match = LWT_PATTERN.match(line)
    if not match:
        # Try more lenient parsing
        if '=' in line and line.startswith('('):
            parts = line.split('=', 1)
            step_id = parts[0].strip('() ')
            rest = parts[1].strip()
            # Extract instruction type
            paren_idx = rest.find('(')
            if paren_idx > 0:
                instr_type = rest[:paren_idx].strip()
                params = rest[paren_idx+1:].rstrip(')')
                return step_id, instr_type, params
        return None, None, None

    return match.group(1), match.group(2), match.group(3)


def parse_lwt_script(script: str) -> List[Tuple[str, str, str]]:
    """
    Parse complete LWT script.

    Args:
        script: Multi-line LWT script

    Returns:
        List of (step_id, instruction_type, parameters)
    """
    steps = []
    for line in script.split('\n'):
        step_id, instr_type, params = parse_lwt_line(line)
        if step_id:
            steps.append((step_id, instr_type, params))
    return steps


def find_dependencies(params: str) -> Set[str]:
    """
    Find variable references in a parameter string.

    Args:
        params: Parameter string potentially containing {(var)} references

    Returns:
        Set of referenced step_ids
    """
    deps = set()
    for match in VAR_PATTERN.finditer(params):
        var_name = match.group(1) or match.group(2)
        if var_name and var_name != 'context':
            deps.add(var_name)
    return deps


def build_dependency_graph(steps: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    """
    Build dependency graph from LWT steps.

    Args:
        steps: List of parsed LWT steps

    Returns:
        Dict mapping step_id to set of dependencies
    """
    graph = {}
    for step_id, instr_type, params in steps:
        graph[step_id] = find_dependencies(params)
    return graph


def topological_sort(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    """
    Sort steps into execution layers (parallelizable groups).

    Args:
        graph: Dependency graph from build_dependency_graph

    Returns:
        List of sets, each set contains steps that can run in parallel
    """
    # Track completed steps
    completed = set()
    layers = []

    remaining = set(graph.keys())

    while remaining:
        # Find steps whose dependencies are all completed
        ready = set()
        for step in remaining:
            deps = graph.get(step, set())
            if deps.issubset(completed):
                ready.add(step)

        if not ready:
            # Circular dependency or missing step
            # Just add remaining steps as final layer
            layers.append(remaining)
            break

        layers.append(ready)
        completed.update(ready)
        remaining -= ready

    return layers


def build_execution_layers(steps: List[Tuple[str, str, str]]) -> List[List[Tuple[str, str, str]]]:
    """
    Organize LWT steps into execution layers for DAG execution.

    Args:
        steps: List of parsed LWT steps

    Returns:
        List of layers, each layer is a list of steps that can run in parallel
    """
    graph = build_dependency_graph(steps)
    layer_order = topological_sort(graph)

    # Map step_id to full step tuple
    step_map = {step_id: (step_id, instr_type, params) for step_id, instr_type, params in steps}

    result = []
    for layer_steps in layer_order:
        layer = [step_map[step_id] for step_id in layer_steps if step_id in step_map]
        if layer:
            result.append(layer)

    return result


def substitute_variables(text: str, cache: Dict[str, Any], context: Dict = None) -> str:
    """
    Substitute variable references in text.

    Args:
        text: Text containing {(var)} references
        cache: Cache of step outputs
        context: Restaurant context data

    Returns:
        Text with variables substituted
    """
    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        if var_name == 'context':
            return json.dumps(context) if context else '{}'
        elif var_name in cache:
            val = cache[var_name]
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            return str(val)
        return match.group(0)  # Keep original if not found

    return VAR_PATTERN.sub(replace_var, text)


def resolve_path(data: Any, path: str) -> Any:
    """
    Resolve a path like "[reviews][0][text]" on data.

    Args:
        data: Data to traverse
        path: Path string with bracket notation

    Returns:
        Value at path or None
    """
    # Parse path segments
    segments = re.findall(r'\[([^\]]+)\]', path)

    current = data
    for seg in segments:
        if current is None:
            return None

        # Try numeric index
        try:
            idx = int(seg)
            if isinstance(current, (list, tuple)) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
            continue
        except ValueError:
            pass

        # Try dict key (remove quotes)
        key = seg.strip('\'"')
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def safe_eval(expr: str, context: Dict[str, Any]) -> Any:
    """
    Safely evaluate a Python expression with given context.

    Args:
        expr: Python expression string
        context: Variables available for evaluation

    Returns:
        Result of evaluation
    """
    # Allowed names for safe eval
    safe_names = {
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'len': len,
        'sum': sum,
        'int': int,
        'float': float,
        'str': str,
        'log': math.log,
        'True': True,
        'False': False,
        'None': None,
    }

    # Add context variables
    eval_context = {**safe_names, **context}

    try:
        return eval(expr, {"__builtins__": {}}, eval_context)
    except Exception as e:
        return f"ERROR: {e}"


def format_output(cache: Dict[str, Any]) -> str:
    """
    Format cache values into expected output format.

    Args:
        cache: Cache with computed primitives

    Returns:
        Formatted output string matching eval.py expectations
    """
    # Map cache keys to output field names
    field_map = {
        'n_total': 'N_TOTAL_INCIDENTS',
        'n_total_incidents': 'N_TOTAL_INCIDENTS',
        'incident_score': 'INCIDENT_SCORE',
        'inc_score': 'INCIDENT_SCORE',
        'recency_decay': 'RECENCY_DECAY',
        'recency': 'RECENCY_DECAY',
        'credibility_factor': 'CREDIBILITY_FACTOR',
        'credibility': 'CREDIBILITY_FACTOR',
        'final_risk_score': 'FINAL_RISK_SCORE',
        'final_score': 'FINAL_RISK_SCORE',
        'verdict': 'VERDICT',
    }

    lines = ['===FINAL ANSWERS===']

    # Output in expected order
    output_order = [
        'N_TOTAL_INCIDENTS',
        'INCIDENT_SCORE',
        'RECENCY_DECAY',
        'CREDIBILITY_FACTOR',
        'FINAL_RISK_SCORE',
        'VERDICT',
    ]

    values = {}
    for cache_key, output_key in field_map.items():
        if cache_key in cache and output_key not in values:
            values[output_key] = cache[cache_key]

    for key in output_order:
        if key in values:
            val = values[key]
            if isinstance(val, float):
                lines.append(f'{key}: {val:.2f}')
            else:
                lines.append(f'{key}: {val}')

    lines.append('===END===')
    return '\n'.join(lines)
