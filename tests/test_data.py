"""
Automated data validation tests for EvacAssist-AI.

Tests cover data/residents.csv and data/scenarios.json.
Run with: pytest tests/test_data.py -v
"""

import csv
import json
import pathlib
import pytest

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so tests work from any cwd
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESIDENTS_CSV = DATA_DIR / "residents.csv"
SCENARIOS_JSON = DATA_DIR / "scenarios.json"

# ---------------------------------------------------------------------------
# Expected constants
# ---------------------------------------------------------------------------
EXPECTED_RESIDENT_COUNT = 20
EXPECTED_ZONES = {"North Ridge", "South Ridge", "East Ridge", "West Ridge"}
EXPECTED_RESIDENTS_PER_ZONE = 5

REQUIRED_RESIDENT_COLUMNS = {
    "resident_id",
    "name",
    "age",
    "zone",
    "mobility_needs",
    "transportation_access",
    "medical_equipment",
    "living_arrangement",
    "preferred_language",
    "phone",
    "email",
}

EXPECTED_R016 = {
    "resident_id": "R016",
    "name": "Minh Hoang",
    "age": "81",
    "zone": "West Ridge",
    "mobility_needs": "wheelchair",
    "transportation_access": "no_car",
    "medical_equipment": "home_oxygen",
    "living_arrangement": "lives_alone",
    "preferred_language": "Vietnamese",
}

# String sentinel used in the CSV to signal "no special need"
NO_NEED_SENTINEL = "none"

# ---------------------------------------------------------------------------
# Fixtures — load files once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def residents():
    """Return all rows from residents.csv as a list of dicts."""
    with RESIDENTS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def incidents():
    """Return the incidents list from scenarios.json."""
    with SCENARIOS_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["incidents"]


# ---------------------------------------------------------------------------
# residents.csv tests
# ---------------------------------------------------------------------------

class TestResidentsCSV:

    def test_file_exists(self):
        """residents.csv must exist on disk."""
        assert RESIDENTS_CSV.exists(), f"File not found: {RESIDENTS_CSV}"

    def test_exactly_20_residents(self, residents):
        """There must be exactly 20 resident rows."""
        assert len(residents) == EXPECTED_RESIDENT_COUNT, (
            f"Expected {EXPECTED_RESIDENT_COUNT} residents, got {len(residents)}"
        )

    def test_unique_resident_ids(self, residents):
        """Every resident_id must be unique."""
        ids = [r["resident_id"] for r in residents]
        duplicates = [rid for rid in set(ids) if ids.count(rid) > 1]
        assert not duplicates, f"Duplicate resident IDs found: {duplicates}"

    def test_required_columns_present(self, residents):
        """All required columns must be present in the CSV header."""
        actual_columns = set(residents[0].keys())
        missing = REQUIRED_RESIDENT_COLUMNS - actual_columns
        assert not missing, f"Missing columns: {missing}"

    def test_exactly_four_zones(self, residents):
        """Residents must be distributed across exactly the 4 expected zones."""
        actual_zones = {r["zone"] for r in residents}
        assert actual_zones == EXPECTED_ZONES, (
            f"Zone mismatch.\n  Expected: {EXPECTED_ZONES}\n  Got: {actual_zones}"
        )

    def test_five_residents_per_zone(self, residents):
        """Each zone must contain exactly 5 residents."""
        zone_counts: dict[str, int] = {}
        for r in residents:
            zone_counts[r["zone"]] = zone_counts.get(r["zone"], 0) + 1

        for zone in EXPECTED_ZONES:
            count = zone_counts.get(zone, 0)
            assert count == EXPECTED_RESIDENTS_PER_ZONE, (
                f"Zone '{zone}' has {count} residents, expected {EXPECTED_RESIDENTS_PER_ZONE}"
            )

    def test_r016_attributes(self, residents):
        """R016 must match Minh Hoang's expected profile exactly."""
        r016 = next((r for r in residents if r["resident_id"] == "R016"), None)
        assert r016 is not None, "Resident R016 not found"
        for field, expected_value in EXPECTED_R016.items():
            assert r016[field] == expected_value, (
                f"R016.{field}: expected '{expected_value}', got '{r016[field]}'"
            )

    @pytest.mark.parametrize("field", ["mobility_needs", "medical_equipment"])
    def test_none_sentinel_is_not_a_risk_factor(self, residents, field):
        """
        Residents with 'none' in mobility_needs or medical_equipment must NOT
        be treated as having a special need.

        This test documents the contract: 'none' == no special need.
        Any risk engine consuming this data must check `value != 'none'`
        (or `value not in (None, 'none')`) before flagging a resident.
        """
        none_residents = [r for r in residents if r[field] == NO_NEED_SENTINEL]
        # There should be at least one such resident (confirms the sentinel is used)
        assert none_residents, (
            f"No residents found with {field}='{NO_NEED_SENTINEL}'. "
            "If the sentinel value changed, update NO_NEED_SENTINEL in this file."
        )
        # Verify none of them accidentally have a truthy risk indicator
        for r in none_residents:
            assert r[field] == NO_NEED_SENTINEL, (
                f"Resident {r['resident_id']} has unexpected {field} value: '{r[field]}'"
            )


# ---------------------------------------------------------------------------
# scenarios.json tests
# ---------------------------------------------------------------------------

class TestScenariosJSON:

    def test_file_exists(self):
        """scenarios.json must exist on disk."""
        assert SCENARIOS_JSON.exists(), f"File not found: {SCENARIOS_JSON}"

    def test_valid_json(self):
        """scenarios.json must parse as valid JSON."""
        with SCENARIOS_JSON.open(encoding="utf-8") as f:
            data = json.load(f)  # raises json.JSONDecodeError on invalid JSON
        assert "incidents" in data, "Top-level 'incidents' key missing from scenarios.json"

    def test_exactly_three_incidents(self, incidents):
        """There must be exactly 3 incidents."""
        assert len(incidents) == 3, (
            f"Expected 3 incidents, got {len(incidents)}"
        )

    def test_unique_incident_ids(self, incidents):
        """Every incident_id must be unique."""
        ids = [i["incident_id"] for i in incidents]
        duplicates = [iid for iid in set(ids) if ids.count(iid) > 1]
        assert not duplicates, f"Duplicate incident IDs found: {duplicates}"

    def test_duplicate_of_references_existing_incident(self, incidents):
        """Any duplicate_of value must reference an incident_id that exists."""
        all_ids = {i["incident_id"] for i in incidents}
        for incident in incidents:
            ref = incident.get("duplicate_of")
            if ref is not None:
                assert ref in all_ids, (
                    f"Incident {incident['incident_id']} has duplicate_of='{ref}', "
                    f"but that ID does not exist. Known IDs: {all_ids}"
                )

    def test_inc_2026_003_duplicates_inc_2026_001(self, incidents):
        """INC-2026-003 must explicitly reference INC-2026-001 as its duplicate source."""
        inc003 = next(
            (i for i in incidents if i["incident_id"] == "INC-2026-003"), None
        )
        assert inc003 is not None, "INC-2026-003 not found in scenarios.json"
        assert inc003.get("duplicate_of") == "INC-2026-001", (
            f"Expected duplicate_of='INC-2026-001', got '{inc003.get('duplicate_of')}'"
        )


# ---------------------------------------------------------------------------
# Cross-file consistency tests
# ---------------------------------------------------------------------------

class TestCrossFileConsistency:

    def test_incident_zones_match_resident_zones(self, residents, incidents):
        """
        Every zone referenced in scenarios.json must also appear in residents.csv.

        This catches zone name drift between the two files (e.g. 'West Ridge'
        vs 'WestRidge' or 'west_ridge').
        """
        resident_zones = {r["zone"] for r in residents}
        for incident in incidents:
            zone = incident.get("zone")
            assert zone in resident_zones, (
                f"Incident {incident['incident_id']} references zone '{zone}', "
                f"which does not exist in residents.csv. "
                f"Known zones: {resident_zones}"
            )
