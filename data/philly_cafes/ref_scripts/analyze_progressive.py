import json
from data.loader import load_dataset

# Use Loader to get restaurants
print("Loading via Loader...")

ds = load_dataset("philly_cafes")
restaurants = ds.items

# Normalize attributes exactly like Loader does
def get_flat_attrs(r):
    flat = {}
    
    # 1. Attributes
    attrs = r.get("attributes", {})
    
    for k, v in attrs.items():
        if v is None:
            flat[f"no_{k}"] = True
            continue
            
        if isinstance(v, bool):
            if v: flat[k] = True
            else: flat[f"no_{k}"] = True
            continue
            
        if isinstance(v, str):
            clean_val = v.lower()
            if clean_val == "true": flat[k] = True
            elif clean_val == "false": flat[f"no_{k}"] = True
            elif clean_val in ["none", ""]: flat[f"no_{k}"] = True
            else: 
                flat[f"{k.lower()}_{clean_val}"] = True
                # CRITICAL FIX: Emit base key for truthy strings (matches validate op:true)
                flat[k] = True
            continue
            
        if isinstance(v, dict):
            # CRITICAL FIX: Emit base key for non-empty dicts (matches validate op:true)
            if v: flat[k] = True
            
            prefix = k.lower()
            for subk, subv in v.items():
                if subv: # Only add if True
                    flat[f"{prefix}_{subk}"] = True
                else:
                    flat[f"no_{prefix}_{subk}"] = True
            continue

    # 2. Categories
    cats = r.get("categories", [])
    if cats:
        for c in cats:
            clean_c = c.replace(' ', '_').replace('&', '_&_').lower()
            flat[f"cat_{clean_c}"] = True

    return flat

# Extract unique gold restaurants from requests
gold_ids = set()
with open("data/philly_cafes/requests.jsonl", "r") as f:
    for line in f:
        req = json.loads(line)
        gold_ids.add(req["gold_restaurant"])

print(f"Loaded {len(restaurants)} restaurants.")
print(f"Found {len(gold_ids)} unique gold restaurants.")

# Re-index
attr_index = {} # attr -> set(ids)
gold_components = {} # id -> {A, B, U, ...}
global_nulls = []

biz_attrs = {} # id -> set(attrs)
all_attrs = set()

# Process
for r in restaurants:
    rid = r["business_id"]
    flat = get_flat_attrs(r)
    biz_attrs[rid] = set(flat.keys())
    all_attrs.update(flat.keys())
    
    for a in flat:
        if a not in attr_index: attr_index[a] = set()
        attr_index[a].add(rid)

# Global Nulls - NOT USED ANYMORE by generator (Uses Safe List), but kept for structure
for a in all_attrs:
    if a.startswith("no_"):
        positive = a[3:]
        if positive not in all_attrs: 
             global_nulls.append(positive)
        elif len(attr_index.get(positive, [])) == 0:
             global_nulls.append(positive)

def find_global_components(target_id):
    target_attrs = list(biz_attrs.get(target_id, []))
    if not target_attrs: return None
    
    components = {}
    
    # 1. Global Unique (U)
    # Freq(U) == 1
    uniques = [a for a in target_attrs if len(attr_index[a]) == 1]
    # Filter out boolean "True" flags if they are common but slipped through?
    # No, strict count=1.
    if uniques:
        # Prefer non-boolean if possible? No, Unique is Unique.
        components["U"] = uniques[0]
        
    # 2. Global Pair (U_pair, T_pair)
    # Freq(U_pair) > 1, Freq(T_pair) > 1, Intersection == {target}
    pair_cands = sorted([a for a in target_attrs if 1 < len(attr_index[a]) <= 15], key=lambda x: len(attr_index[x]))
    
    for i in range(len(pair_cands)):
        for j in range(i+1, len(pair_cands)):
            p1, p2 = pair_cands[i], pair_cands[j]
            s1, s2 = set(attr_index[p1]), set(attr_index[p2])
            intersect = s1.intersection(s2)
            if len(intersect) == 1 and list(intersect)[0] == target_id:
                components["U_pair"] = p1
                components["T_pair"] = p2
                break
        if "U_pair" in components: break
    
    # 3. Local Context (A, B) + Local Unique
    broad = [a for a in target_attrs if 5 <= len(attr_index[a]) <= 25]
    for b in broad:
        s_b = set(attr_index[b])
        # Find B intersecting
        others = [a for a in target_attrs if a != b and 2 <= len(attr_index[a]) <= 15]
        for o in others:
            s_o = set(attr_index[o])
            subset = s_b.intersection(s_o)
            if len(subset) <= 6 and len(subset) >= 2:
                # subset valid, look for unique in subset
                valid_subset_ids = subset
                 
                # Filter down to target
                for u in target_attrs:
                    if u in [b, o]: continue
                    s_u = set(attr_index[u])
                    if subset.intersection(s_u) == {target_id}:
                         components["A"] = b
                         components["B"] = o
                         components["U_local"] = u
                         # Find T2 local (redundant true)
                         for t in target_attrs:
                             if t not in [b, o, u]:
                                 components["T_local"] = t
                                 break
                         break
        if "A" in components: break

    return components

print("\n--- Components for Gold Restaurants ---")
for gid in gold_ids:
    comps = find_global_components(gid)
    if comps:
        gold_components[gid] = comps
        print(f"ID {gid[-5:]}: Has {list(comps.keys())}")
    else:
        print(f"ID {gid[-5:]}: No components found")
