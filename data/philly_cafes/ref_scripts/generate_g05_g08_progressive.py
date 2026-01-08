import json
import random
from analyze_progressive import restaurants, get_flat_attrs, gold_components, global_nulls

# ATTR_MAP for overrides
ATTR_MAP = {
    "Alcohol": {"text": "serves alcohol", "path": ["attributes", "Alcohol"], "value": "True", "op": "true"},
    # "WiFi": {"text": "has wifi", "path": ["attributes", "WiFi"], "value": "free", "op": "contains"}, # Example
}

def get_attr_spec(key):
    spec = {}
    is_neg = False
    
    # Handle negation prefix
    if key.startswith("no_"):
        is_neg = True
        key = key[3:]

    # 1. Manual Overrides
    if key in ATTR_MAP:
        spec = ATTR_MAP[key].copy()
    
    # 2. Dynamic Parsing
    elif key.startswith("cat_"):
        raw = key[4:]
        # Fix: Ensure single space around ampersand, single space elsewhere
        val = raw.replace("_&_", " AMPERSAND ") 
        val = val.replace("_", " ")
        val = val.replace(" AMPERSAND ", " & ")
        val = val.title()
        val = val.replace("(Traditional)", "(Traditional)").replace("(New)", "(New)")
        # Clean up any double spaces
        val = " ".join(val.split())
        spec = {"text": f"is a {val}", "path": ["categories"], "value": val, "op": "contains"}
        
    elif key.startswith("wifi_"):
        val = key.split("_")[1]
        # Strip quotes if present in key
        if val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        spec = {"text": f"has {val} wifi", "path": ["attributes", "WiFi"], "value": val, "op": "contains"}

    elif key.startswith("ambience_"):
        trait = key.split("_")[1]
        spec = {"text": f"has {trait} ambience", "path": ["attributes", "Ambience"], "value": f"'{trait}': True", "op": "contains"}
        
    elif key.startswith("goodformeal_"):
        meal = key.split("_")[1]
        spec = {"text": f"is good for {meal}", "path": ["attributes", "GoodForMeal"], "value": f"'{meal}': True", "op": "contains"}
    
    elif key.startswith("price_"):
        rng = key.split("_")[1]
        spec = {"text": f"price range {rng}", "path": ["attributes", "RestaurantsPriceRange2"], "value": rng, "op": "true"}
    
    elif key.startswith("alcohol_"):
        val = key.replace("alcohol_", "")
        spec = {"text": f"alcohol: {val}", "path": ["attributes", "Alcohol"], "value": val, "op": "contains"}
        
    elif key.startswith("music_"):
        trait = key.split("_")[1]
        spec = {"text": f"has {trait} music", "path": ["attributes", "Music"], "value": f"'{trait}': True", "op": "contains"}
        
    elif key.startswith("bestnights_"):
        day = key.split("_")[1]
        spec = {"text": f"is good on {day}", "path": ["attributes", "BestNights"], "value": f"'{day}': True", "op": "contains"}

    elif key.startswith("restaurantsattire_"):
        val = key.replace("restaurantsattire_", "")
        spec = {"text": f"attire is {val}", "path": ["attributes", "RestaurantsAttire"], "value": val, "op": "contains"}
        
    elif key.startswith("parking_") or key.startswith("businessparking_"):
        parts = key.split("_")
        val = parts[1]
        spec = {"text": f"has {val} parking", "path": ["attributes", "BusinessParking"], "value": f"'{val}': True", "op": "contains"}

    elif key.startswith("dietary_"):
        val = key.split("_")[1]
        spec = {"text": f"has {val} dietary option", "path": ["attributes", "DietaryRestrictions"], "value": f"'{val}': True", "op": "contains"}

    elif key.startswith("smoking_"):
        val = key.split("_")[1]
        spec = {"text": f"smoking: {val}", "path": ["attributes", "Smoking"], "value": val, "op": "contains"}
        
    elif key.startswith("hair_"):
         val = key.split("_")[1]
         spec = {"text": f"specializes in {val}", "path": ["attributes", "HairSpecializesIn"], "value": f"'{val}': True", "op": "contains"}

    elif key.startswith("noise_") or key.startswith("noiselevel_"):
        parts = key.split("_")
        val = parts[1] if len(parts) > 1 else "average"
        spec = {"text": f"is {val}", "path": ["attributes", "NoiseLevel"], "value": val, "op": "true"}

    else:
        # Default Boolean Attribute
        spec = {"text": f"has {key}", "path": ["attributes", key], "value": "True", "op": "true"}
        
    # Apply Negation Logic
    if is_neg:
        spec["text"] = "no " + spec["text"]
        spec["negate"] = True
    else:
        spec["negate"] = False
        
    return spec

def make_node(op, args):
    children = []
    for arg in args:
        if isinstance(arg, str): # Condition name
            spec = get_attr_spec(arg)
            
            # Create base condition
            cond = {
                "aspect": arg,
                "evidence": {
                    "kind": "item_meta",
                    "path": spec["path"],
                    "op": spec.get("op", "true"),
                    "value": spec["value"] # CRITICAL: Validate.py needs 'value'
                }
            }
            # Copy semantic keys for completeness (optional but good for debugging)
            if spec["op"] == "contains": cond["evidence"]["contains"] = spec["value"]
            else: cond["evidence"]["true"] = spec["value"]
            
            # Handle Negation
            if spec["negate"]:
                # Wrap in NOT structure
                if arg.startswith("no_"):
                     cond["aspect"] = arg[3:]
                req = {"op": "NOT", "args": [cond]}
            else:
                req = cond
            
            children.append(req)
        elif isinstance(arg, dict): # Nested structure
            children.append(arg)
    return {"op": op, "args": children}

# Target Pools
targets_with_U = [gid for gid, c in gold_components.items() if "U" in c]
targets_with_Pair = [gid for gid, c in gold_components.items() if "U_pair" in c and "T_pair" in c]
targets_with_Local = [gid for gid, c in gold_components.items() if "A" in c and "B" in c and "U_local" in c and "T_local" in c]

if not targets_with_U:
    print("WARNING: No U targets found. G05 will be skipped!")
    
new_requests = []
req_id_counter = 41

def get_safe_distractors(n, exclude_attrs):
    ds = [d for d in global_nulls if d not in exclude_attrs]
    if len(ds) < n: return ds
    return random.sample(ds, n)

# --- G05 (R41-R50): OR(U, D1, D2) ---
for i in range(10):
    if not targets_with_U: break
    rid = f"R{req_id_counter}"
    req_id_counter += 1
    gid = targets_with_U[i % len(targets_with_U)]
    comps = gold_components[gid]
    U = comps["U"]
    Ds = get_safe_distractors(2, {U})
    D1, D2 = Ds[0], Ds[1]
    
    structure = make_node("OR", [U, D1, D2])
    text = f"Looking for a place that {get_attr_spec(U)['text']}, or {get_attr_spec(D1)['text']}, or {get_attr_spec(D2)['text']}."
    
    new_requests.append({
        "id": rid, "group": "G05", "gold_restaurant": gid, "text": text,
        "shorthand": f"OR({U}, {D1}, {D2})", "structure": structure
    })

# --- G06 (R51-R60): OR(AND(U, T2), AND(D1, D2)) ---
for i in range(10):
    if not targets_with_Pair: break
    rid = f"R{req_id_counter}"
    req_id_counter += 1
    gid = targets_with_Pair[i % len(targets_with_Pair)]
    comps = gold_components[gid]
    U = comps["U_pair"]; T2 = comps["T_pair"]
    Ds = get_safe_distractors(2, {U, T2})
    D1, D2 = Ds[0], Ds[1]
    
    term1 = make_node("AND", [U, T2]); term2 = make_node("AND", [D1, D2])
    structure = make_node("OR", [term1, term2])
    text = f"Looking for a place that {get_attr_spec(U)['text']} and {get_attr_spec(T2)['text']}, or has impossible features."
    
    new_requests.append({
        "id": rid, "group": "G06", "gold_restaurant": gid, "text": text,
        "shorthand": f"OR(AND({U}, {T2}), AND({D1}, {D2}))", "structure": structure
    })

# --- G07 (R61-R70): AND(OR(U, D1), OR(T2, D2)) ---
for i in range(10):
    if not targets_with_Pair: break
    rid = f"R{req_id_counter}"
    req_id_counter += 1
    gid = targets_with_Pair[(i+5) % len(targets_with_Pair)]
    comps = gold_components[gid]
    U = comps["U_pair"]; T2 = comps["T_pair"]
    Ds = get_safe_distractors(2, {U, T2})
    D1, D2 = Ds[0], Ds[1]
    
    term1 = make_node("OR", [U, D1]); term2 = make_node("OR", [T2, D2])
    structure = make_node("AND", [term1, term2])
    text = f"Looking for {get_attr_spec(U)['text']} (or dummy) and {get_attr_spec(T2)['text']} (or dummy)."
    
    new_requests.append({
        "id": rid, "group": "G07", "gold_restaurant": gid, "text": text,
        "shorthand": f"AND(OR({U}, {D1}), OR({T2}, {D2}))", "structure": structure
    })

# --- G08 (R71-R80): AND(A, B, OR(D1, AND(U, T2))) ---
for i in range(10):
    if not targets_with_Local: break
    rid = f"R{req_id_counter}"
    req_id_counter += 1
    gid = targets_with_Local[i % len(targets_with_Local)]
    comps = gold_components[gid]
    A = comps["A"]; B = comps["B"]; U = comps["U_local"]; T2 = comps["T_local"]
    Ds = get_safe_distractors(1, {A, B, U, T2})
    D1 = Ds[0]
    
    term_unique = make_node("AND", [U, T2])
    term_or = make_node("OR", [D1, term_unique])
    structure = make_node("AND", [A, B, term_or])
    text = f"Looking for {get_attr_spec(A)['text']} that {get_attr_spec(B)['text']}, and either has {D1} or ({get_attr_spec(U)['text']} and {get_attr_spec(T2)['text']})."
    
    new_requests.append({
        "id": rid, "group": "G08", "gold_restaurant": gid, "text": text,
        "shorthand": f"AND({A}, {B}, OR({D1}, AND({U}, {T2})))", "structure": structure
    })

print(f"Generated {len(new_requests)} new requests.")
with open("g05_g08_generated.jsonl", "w") as f:
    for req in new_requests:
        f.write(json.dumps(req) + "\n")
