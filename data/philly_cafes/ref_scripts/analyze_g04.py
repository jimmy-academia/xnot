
import json
import collections

# Target restaurants
targets = {
    "R38": {"name": "Front Street Cafe", "id": "JrG4NINLspXPNhSXg7Q07Q"},
    "R39": {"name": "Cafe La Maude", "id": "K7KHmHzxNwzqiijSJeKe_A"},
    "R40": {"name": "United By Blue", "id": "ZpgVL2z1kgRi954c9m9INw"}
}

data = []
with open("data/philly_cafes/restaurants.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Helper to flatten attributes
def get_attrs(r):
    attrs = r.get("attributes", {})
    categories = r.get("categories", "").split(", ")
    
    flat = {}
    
    # Categories
    if "Shopping" in categories: flat["cat_shopping"] = True
    if "Fashion" in categories: flat["cat_fashion"] = True
    
    # Attributes
    if "CoatCheck" in attrs and attrs["CoatCheck"] == "True": flat["coat_check"] = True
    if "HappyHour" in attrs and attrs["HappyHour"] == "True": flat["happy_hour"] = True
    if "Alcohol" in attrs: 
        alc = attrs['Alcohol'].replace("u'", "").replace("'", "")
        flat[f"alcohol_{alc}"] = True
    if "DogsAllowed" in attrs and attrs["DogsAllowed"] == "True": flat["dogs_allowed"] = True
    if "WheelchairAccessible" in attrs and attrs["WheelchairAccessible"] == "True": flat["wheelchair"] = True
    if "BYOB" in attrs and attrs["BYOB"] == "True": flat["byob"] = True
    if "RestaurantsPriceRange2" in attrs: flat[f"price_{attrs['RestaurantsPriceRange2']}"] = True
    if "OutdoorSeating" in attrs and attrs["OutdoorSeating"] == "True": flat["outdoor"] = True
    
    # Ambience
    ambience_str = attrs.get("Ambience", "{}")
    # simplified parsing of python dict string
    if "'hipster': True" in ambience_str: flat["ambience_hipster"] = True
    if "'trendy': True" in ambience_str: flat["ambience_trendy"] = True
    if "'classy': True" in ambience_str: flat["ambience_classy"] = True
    if "'casual': True" in ambience_str: flat["ambience_casual"] = True

    return flat

# Find unique combos
for key, target in targets.items():
    print(f"\nAnalyzing {key}: {target['name']}")
    target_rest = next((r for r in data if r["business_id"] == target["id"]), None)
    if not target_rest:
        print("Target not found!")
        continue
    
    t_attrs = get_attrs(target_rest)
    print(f"Attributes: {t_attrs}")
    
    # check if 'shopping' is unique to United By Blue
    if target["name"] == "United By Blue":
        others_with_shopping = [r["name"] for r in data if r["business_id"] != target["id"] and "Shopping" in r.get("categories", "")]
        print(f"Others with Shopping: {others_with_shopping}")
        
    # Hypothesis Checks
    if target["name"] == "Front Street Cafe":
        matches = [r["name"] for r in data if 
                   r.get("attributes", {}).get("Alcohol") == "u'full_bar'" and
                   r.get("attributes", {}).get("CoatCheck") == "True"]
        print(f"Hypothesis R38 (FullBar + CoatCheck): {matches}")

    if target["name"] == "Cafe La Maude":
        matches = [r["name"] for r in data if 
                   r.get("attributes", {}).get("BYOB") == "True" and
                   r.get("attributes", {}).get("DogsAllowed") == "True" and
                   "'classy': True" in r.get("attributes", {}).get("Ambience", "")]
        print(f"Hypothesis R39 (BYOB + DogsAllowed + Classy): {matches}")

    if target["name"] == "United By Blue":
        matches = [r["name"] for r in data if 
                   r.get("attributes", {}).get("DogsAllowed") == "True" and
                   r.get("attributes", {}).get("RestaurantsPriceRange2") == "2" and
                   r.get("attributes", {}).get("Alcohol") == "u'none'" and
                   r.get("attributes", {}).get("WheelchairAccessible") == "True"]
        print(f"Hypothesis R40 (Dogs + Price2 + NoAlcohol + Wheelchair): {matches}")

