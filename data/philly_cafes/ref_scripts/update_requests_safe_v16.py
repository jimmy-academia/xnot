import json

new_requests = {}
with open("g05_g08_generated.jsonl", "r") as f:
    for line in f:
        req = json.loads(line)
        new_requests[req["id"]] = line.strip()

updated_lines = []
with open("data/philly_cafes/requests.jsonl", "r") as f:
    for line in f:
        try:
            req = json.loads(line)
            if req["id"] in new_requests:
                updated_lines.append(new_requests[req["id"]])
            else:
                updated_lines.append(line.strip())
        except:
             updated_lines.append(line.strip())

with open("data/philly_cafes/requests.jsonl", "w") as f:
    for line in updated_lines:
        f.write(line + "\n")
