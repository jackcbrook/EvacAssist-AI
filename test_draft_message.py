import csv
import json

from bedrock_client import draft_message

# Load fictional residents.
with open("data/residents.csv", newline="", encoding="utf-8") as f:
    residents = list(csv.DictReader(f))

# Load incident scenarios.
with open("data/scenarios.json", encoding="utf-8") as f:
    incidents = json.load(f)["incidents"]

# Select Minh Hoang.
resident = next(
    r for r in residents
    if r["resident_id"] == "R016"
)

# Select confirmed West Ridge wildfire.
incident = next(
    i for i in incidents
    if i["incident_id"] == "INC-2026-001"
)

# Adapt the scenario's trusted metadata to the
# function's expected input.
incident_input = {
    **incident,
    "evacuation_order_reported": True
}

result = draft_message(resident, incident_input)

print(json.dumps(result, indent=2, ensure_ascii=False))
