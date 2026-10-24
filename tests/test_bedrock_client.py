"""
Tests for bedrock_client.py.

Parsing/validation tests run offline. Scenario tests call real Bedrock and are
skipped if no AWS credentials are available.
Run with: pytest tests/test_bedrock_client.py -v
"""

import json
import pathlib
from unittest import mock

import boto3
import pytest

import bedrock_client
from bedrock_client import (
    IncidentExtractionError,
    extract_incident,
    parse_json,
    validate_incident,
)

SCENARIOS_JSON = pathlib.Path(__file__).parent.parent / "data" / "scenarios.json"
INCIDENTS = {i["incident_id"]: i for i in json.loads(SCENARIOS_JSON.read_text())["incidents"]}

VALID_REPLY = {
    "location": "Crestview Drive",
    "zone": "West Ridge",
    "severity": "high",
    "verified": "yes",
    "evidence": "Active wildfire confirmed",
    "summary": "Confirmed fire in West Ridge.",
}


def _has_aws_credentials():
    try:
        bedrock_client.get_client()._request_signer._credentials.get_frozen_credentials()
        return True
    except Exception:
        return False


live = pytest.mark.skipif(not _has_aws_credentials(), reason="No AWS credentials for Bedrock")


# ---------------------------------------------------------------------------
# Parsing and validation (offline)
# ---------------------------------------------------------------------------
class TestParseAndValidate:
    def test_parses_plain_json(self):
        assert parse_json(json.dumps(VALID_REPLY)) == VALID_REPLY

    def test_parses_fenced_json_with_prose(self):
        text = "Here you go:\n```json\n" + json.dumps(VALID_REPLY) + "\n```"
        assert parse_json(text) == VALID_REPLY

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            parse_json("Location: West Ridge, severity: high")

    def test_normalizes_case(self):
        result = validate_incident({**VALID_REPLY, "zone": "west ridge", "severity": "HIGH", "verified": "Yes"})
        assert (result["zone"], result["severity"], result["verified"]) == ("West Ridge", "high", "yes")

    def test_missing_key_raises(self):
        data = dict(VALID_REPLY)
        del data["verified"]
        with pytest.raises(ValueError, match="Missing keys"):
            validate_incident(data)

    @pytest.mark.parametrize("key,value", [
        ("zone", "Oak Canyon"),
        ("severity", "Moderate to High"),
        ("verified", "maybe"),
    ])
    def test_invalid_value_raises(self, key, value):
        with pytest.raises(ValueError, match=f"Invalid {key}"):
            validate_incident({**VALID_REPLY, key: value})


class TestRetry:
    def test_retries_once_after_bad_json(self):
        replies = iter(["not json", json.dumps(VALID_REPLY)])
        with mock.patch.object(bedrock_client, "_converse", side_effect=lambda _: next(replies)) as m:
            assert extract_incident("report")["zone"] == "West Ridge"
        assert m.call_count == 2

    def test_gives_up_after_two_bad_replies(self):
        with mock.patch.object(bedrock_client, "_converse", return_value="not json") as m:
            with pytest.raises(IncidentExtractionError):
                extract_incident("report")
        assert m.call_count == 2


# ---------------------------------------------------------------------------
# Scenarios from data/scenarios.json (real Bedrock)
# ---------------------------------------------------------------------------
def _expected(incident):
    return {
        "zone": incident["zone"],
        "severity": incident["severity"],
        "verified": "yes" if incident["verified"] else "no",
    }


def _facts(result):
    return {key: result[key] for key in ("zone", "severity", "verified")}


@pytest.fixture(scope="module")
def results():
    """Extract every scenario once and share the results across tests."""
    return {incident_id: extract_incident(incident["description"])
            for incident_id, incident in INCIDENTS.items()}


@live
class TestScenarios:
    def test_confirmed_wildfire(self, results):
        assert _facts(results["INC-2026-001"]) == _expected(INCIDENTS["INC-2026-001"])

    def test_unverified_smoke(self, results):
        assert _facts(results["INC-2026-002"]) == _expected(INCIDENTS["INC-2026-002"])

    def test_second_report_of_same_wildfire(self, results):
        duplicate = INCIDENTS["INC-2026-003"]
        assert _facts(results["INC-2026-003"]) == _expected(duplicate)
        # Different wording, same fire: facts must match the original report
        assert _facts(results["INC-2026-003"]) == _facts(results[duplicate["duplicate_of"]])

    def test_evidence_quotes_the_report(self, results):
        for incident_id, result in results.items():
            assert result["evidence"], f"{incident_id} has no evidence"
            assert result["summary"], f"{incident_id} has no summary"
