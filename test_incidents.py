"""
Send example incidents from data/scenarios.json to DeepSeek V3.2 on Bedrock
and compare the extracted facts with the expected values in the file.

Run with: python test_incidents.py                  (tests INC-2026-001 and INC-2026-002)
          python test_incidents.py INC-2026-003     (tests any incident IDs you pass)
"""

import json
import pathlib
import sys

import boto3

SCENARIOS_JSON = pathlib.Path(__file__).parent / "data" / "scenarios.json"
DEFAULT_IDS = ["INC-2026-001", "INC-2026-002"]
ZONES = ["North Ridge", "South Ridge", "East Ridge", "West Ridge"]

PROMPT = """You read wildfire reports for a county emergency coordinator.
Extract the facts from the report below. Reply with ONLY a JSON object, no other text, with these keys:
- "location": the specific place mentioned (streets, landmarks)
- "zone": exactly one of {zones}, or "Unknown"
- "severity": "low", "medium", "high", or "unknown" if the report does not establish an actual fire
- "verified": "yes" only if officials or fire crews confirm the fire, otherwise "no"
- "evidence": the exact words from the report that support severity and verified
- "summary": one sentence

Report:
{report}"""

session = boto3.Session(profile_name="hackathon", region_name="us-east-1")
client = session.client("bedrock-runtime")


def ask_deepseek(report_text):
    response = client.converse(
        modelId="deepseek.v3.2",
        messages=[{
            "role": "user",
            "content": [{"text": PROMPT.format(zones=", ".join(ZONES), report=report_text)}],
        }],
        inferenceConfig={"maxTokens": 400, "temperature": 0},
    )
    text = response["output"]["message"]["content"][0]["text"].strip()
    # The model sometimes wraps JSON in a ```json fence
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def check(incident, result):
    expected = {
        "zone": incident["zone"],
        "severity": incident["severity"],
        "verified": "yes" if incident["verified"] else "no",
    }
    passed = True
    for key, want in expected.items():
        got = str(result.get(key, "")).lower()
        ok = got == want.lower()
        passed &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {key:<9} expected={want!r:<14} got={result.get(key)!r}")
    return passed


def main():
    ids = sys.argv[1:] or DEFAULT_IDS
    incidents = {i["incident_id"]: i for i in json.loads(SCENARIOS_JSON.read_text())["incidents"]}

    all_passed = True
    for incident_id in ids:
        incident = incidents[incident_id]
        print(f"\n=== {incident_id} ({incident['status']}, from {incident['reported_by']})")
        print(f"Report: {incident['description']}\n")
        result = ask_deepseek(incident["description"])
        print("DeepSeek output:")
        print(json.dumps(result, indent=2))
        print()
        all_passed &= check(incident, result)

    print("\nALL PASSED" if all_passed else "\nSOME CHECKS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
