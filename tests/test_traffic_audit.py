"""End-to-end tests for POST /api/analyses/{id}/traffic-audit."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-test-ta-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _save(client: TestClient, rules: list[dict], name: str | None = None) -> str:
    body: dict = {"rules": rules}
    if name is not None:
        body["name"] = name
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["persisted"] is True
    return r.json()["id"]


NESTED_RULES = [
    {"rule_id": "core", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100, "source_tag": "hq"},
    {"rule_id": "edge", "cidr": "10.1.0.0/16", "action": "deny", "priority": 500, "source_tag": "dc1"},
    {"rule_id": "lab", "cidr": "10.1.1.0/24", "action": "permit", "priority": 900, "source_tag": "lab"},
    {"rule_id": "v6", "cidr": "2001:db8::/32", "action": "deny", "source_tag": "v6"},
]


def _audit(client: TestClient, aid: str, records: list[dict]) -> dict:
    r = client.post(f"/api/analyses/{aid}/traffic-audit", json={"records": records})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Attribution semantics
# ---------------------------------------------------------------------------
def test_lpm_attribution_and_consistency(client: TestClient) -> None:
    aid = _save(client, NESTED_RULES, name="nested")

    payload = _audit(
        client,
        aid,
        [
            {"ip": "10.2.3.4", "action": "permit"},                 # core
            {"ip": "10.1.2.3", "action": "permit"},                 # edge, but observed permit
            {"ip": "10.1.1.9", "action": "permit", "count": 5},     # lab
            {"ip": "192.0.2.1", "action": "deny"},                  # no rule covers it
            {"ip": "2001:db8::1", "action": "deny", "count": 2},    # v6
        ],
    )

    assert payload["analysis_id"] == aid
    assert payload["ruleset_hash"]

    records = payload["records"]
    assert [r["index"] for r in records] == [0, 1, 2, 3, 4]

    assert records[0] == {
        "index": 0, "ip": "10.2.3.4", "observed_action": "permit", "count": 1,
        "status": "covered", "rule_id": "core", "expected_action": "permit",
        "consistent": True,
    }
    assert records[1]["rule_id"] == "edge"
    assert records[1]["expected_action"] == "deny"
    assert records[1]["consistent"] is False
    assert records[2]["rule_id"] == "lab"
    assert records[2]["consistent"] is True
    assert records[2]["count"] == 5
    # Uncovered is explicit: never inferred as permit or deny.
    assert records[3] == {
        "index": 3, "ip": "192.0.2.1", "observed_action": "deny", "count": 1,
        "status": "uncovered", "rule_id": None, "expected_action": None,
        "consistent": None,
    }
    assert records[4]["rule_id"] == "v6"
    assert records[4]["consistent"] is True

    summary = payload["summary"]
    assert summary == {
        "record_count": 5,
        "total_hits": 10,
        "covered_records": 4,
        "covered_hits": 9,
        "uncovered_records": 1,
        "uncovered_hits": 1,
        "consistent_records": 3,
        "consistent_hits": 8,
        "inconsistent_records": 1,
        "inconsistent_hits": 1,
    }

    # Weighted aggregation per winning rule, in original submission order.
    assert payload["by_rule"] == [
        {"rule_id": "core", "cidr": "10.0.0.0/8", "action": "permit", "source_tag": "hq",
         "records": 1, "hits": 1, "consistent_hits": 1, "inconsistent_hits": 0},
        {"rule_id": "edge", "cidr": "10.1.0.0/16", "action": "deny", "source_tag": "dc1",
         "records": 1, "hits": 1, "consistent_hits": 0, "inconsistent_hits": 1},
        {"rule_id": "lab", "cidr": "10.1.1.0/24", "action": "permit", "source_tag": "lab",
         "records": 1, "hits": 5, "consistent_hits": 5, "inconsistent_hits": 0},
        {"rule_id": "v6", "cidr": "2001:db8::/32", "action": "deny", "source_tag": "v6",
         "records": 1, "hits": 2, "consistent_hits": 2, "inconsistent_hits": 0},
    ]

    assert payload["mismatches"] == [
        {"rule_id": "edge", "expected_action": "deny", "observed_action": "permit",
         "records": 1, "hits": 1}
    ]
    assert payload["uncovered"] == {"records": 1, "hits": 1}


def test_longest_prefix_beats_priority_and_submission_order(client: TestClient) -> None:
    """Attribution is LPM, never first-match or best-priority-number."""
    # The broad permit rule is submitted first AND has the best (smallest)
    # priority number; the longer-prefix deny must still win inside /16.
    aid = _save(
        client,
        [
            {"rule_id": "wide", "cidr": "10.0.0.0/8", "action": "permit", "priority": 1},
            {"rule_id": "narrow", "cidr": "10.1.0.0/16", "action": "deny", "priority": 999},
        ],
    )

    payload = _audit(client, aid, [{"ip": "10.1.0.7", "action": "deny"}])
    record = payload["records"][0]
    assert record["rule_id"] == "narrow"
    assert record["expected_action"] == "deny"
    assert record["consistent"] is True


def test_same_cidr_tie_breaks_by_priority_then_submission_order(client: TestClient) -> None:
    aid = _save(
        client,
        [
            # Same CIDR: the smaller priority number wins (b over a).
            {"rule_id": "a", "cidr": "10.9.0.0/16", "action": "deny", "priority": 10},
            {"rule_id": "b", "cidr": "10.9.0.0/16", "action": "permit", "priority": 5},
            # Same CIDR and same priority: the earlier submission wins (c over d).
            {"rule_id": "c", "cidr": "10.8.0.0/16", "action": "permit", "priority": 7},
            {"rule_id": "d", "cidr": "10.8.0.0/16", "action": "deny", "priority": 7},
        ],
    )

    payload = _audit(
        client,
        aid,
        [
            {"ip": "10.9.0.1", "action": "deny"},   # winner is b (permit) -> inconsistent
            {"ip": "10.8.0.1", "action": "permit"},  # winner is c (permit) -> consistent
        ],
    )
    first, second = payload["records"]
    assert first["rule_id"] == "b"
    assert first["expected_action"] == "permit"
    assert first["consistent"] is False
    assert second["rule_id"] == "c"
    assert second["expected_action"] == "permit"
    assert second["consistent"] is True


def test_record_order_does_not_change_attribution(client: TestClient) -> None:
    aid = _save(client, NESTED_RULES)
    records = [
        {"ip": "10.1.1.9", "action": "permit", "count": 2},
        {"ip": "10.1.2.3", "action": "deny"},
        {"ip": "10.2.3.4", "action": "deny", "count": 4},
        {"ip": "203.0.113.7", "action": "permit"},
    ]

    forward = _audit(client, aid, records)
    backward = _audit(client, aid, list(reversed(records)))

    # Per-record verdicts are attached to their own record (by ip), and every
    # aggregate is identical regardless of the request's record order.
    def verdicts_by_ip(payload: dict) -> dict:
        return {r["ip"]: {k: v for k, v in r.items() if k != "index"} for r in payload["records"]}

    assert verdicts_by_ip(forward) == verdicts_by_ip(backward)
    for key in ("summary", "by_rule", "mismatches", "uncovered"):
        assert forward[key] == backward[key]


def test_ipv6_only_ruleset_leaves_ipv4_uncovered(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "v6", "cidr": "2001:db8::/32", "action": "deny"}])

    payload = _audit(
        client,
        aid,
        [
            {"ip": "2001:0DB8:0000:0000:0000:0000:0000:0001", "action": "deny"},
            {"ip": "10.0.0.1", "action": "deny"},
        ],
    )
    v6_rec, v4_rec = payload["records"]
    # The IPv6 address is canonicalized in the response.
    assert v6_rec["ip"] == "2001:db8::1"
    assert v6_rec["status"] == "covered"
    assert v6_rec["rule_id"] == "v6"
    # A different family never attributes.
    assert v4_rec["status"] == "uncovered"
    assert v4_rec["rule_id"] is None


def test_weighted_counts_drive_all_aggregates(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "r1", "cidr": "10.0.0.0/8", "action": "permit"}])

    payload = _audit(
        client,
        aid,
        [
            {"ip": "10.0.0.1", "action": "permit", "count": 3},
            {"ip": "10.0.0.2", "action": "deny", "count": 2},
            {"ip": "10.0.0.2", "action": "deny"},  # default count = 1
            {"ip": "11.0.0.1", "action": "permit", "count": 4},
        ],
    )

    assert payload["summary"] == {
        "record_count": 4,
        "total_hits": 10,
        "covered_records": 3,
        "covered_hits": 6,
        "uncovered_records": 1,
        "uncovered_hits": 4,
        "consistent_records": 1,
        "consistent_hits": 3,
        "inconsistent_records": 2,
        "inconsistent_hits": 3,
    }
    assert payload["by_rule"] == [
        {"rule_id": "r1", "cidr": "10.0.0.0/8", "action": "permit", "source_tag": "default",
         "records": 3, "hits": 6, "consistent_hits": 3, "inconsistent_hits": 3}
    ]
    assert payload["mismatches"] == [
        {"rule_id": "r1", "expected_action": "permit", "observed_action": "deny",
         "records": 2, "hits": 3}
    ]
    assert payload["uncovered"] == {"records": 1, "hits": 4}


def test_action_aliases_are_normalized(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "r1", "cidr": "10.0.0.0/8", "action": "permit"}])

    payload = _audit(client, aid, [{"ip": "10.0.0.1", "action": "ALLOW"}])
    record = payload["records"][0]
    assert record["observed_action"] == "permit"
    assert record["consistent"] is True


# ---------------------------------------------------------------------------
# Batch validation (all-or-nothing)
# ---------------------------------------------------------------------------
def test_invalid_ips_reject_the_whole_batch(client: TestClient) -> None:
    aid = _save(client, NESTED_RULES)

    r = client.post(
        f"/api/analyses/{aid}/traffic-audit",
        json={
            "records": [
                {"ip": "10.0.0.1", "action": "permit"},
                {"ip": "not-an-ip", "action": "permit"},
                {"ip": "10.0.0.1/32", "action": "deny"},  # prefix length not allowed
            ]
        },
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"] == "invalid_traffic_audit"
    # Every offending record is listed at once; the valid record is not
    # processed either (all-or-nothing).
    assert len(body["issues"]) == 2
    assert "record #2" in body["issues"][0]
    assert "record #3" in body["issues"][1]


def test_invalid_ip_is_checked_before_version_existence(client: TestClient) -> None:
    r = client.post(
        "/api/analyses/an_0000000000000000/traffic-audit",
        json={"records": [{"ip": "not-an-ip", "action": "permit"}]},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_traffic_audit"


@pytest.mark.parametrize("action", ["permit-ish", "", 42])
def test_invalid_action_rejected(client: TestClient, action: object) -> None:
    aid = _save(client, NESTED_RULES)
    r = client.post(
        f"/api/analyses/{aid}/traffic-audit",
        json={"records": [{"ip": "10.0.0.1", "action": action}]},
    )
    assert r.status_code == 422


@pytest.mark.parametrize("count", [0, -2, 1.5, "3", True])
def test_invalid_count_rejected(client: TestClient, count: object) -> None:
    aid = _save(client, NESTED_RULES)
    r = client.post(
        f"/api/analyses/{aid}/traffic-audit",
        json={"records": [{"ip": "10.0.0.1", "action": "permit", "count": count}]},
    )
    assert r.status_code == 422


def test_malformed_batches_rejected(client: TestClient) -> None:
    aid = _save(client, NESTED_RULES)
    url = f"/api/analyses/{aid}/traffic-audit"

    # Empty batch.
    assert client.post(url, json={"records": []}).status_code == 422
    # Not a list.
    assert client.post(url, json={"records": {"ip": "10.0.0.1"}}).status_code == 422
    # Missing required fields.
    assert client.post(url, json={"records": [{"action": "permit"}]}).status_code == 422
    assert client.post(url, json={"records": [{"ip": "10.0.0.1"}]}).status_code == 422
    # Blank ip.
    assert client.post(url, json={"records": [{"ip": "  ", "action": "permit"}]}).status_code == 422
    # Unknown top-level / record fields.
    assert client.post(url, json={"records": [], "extra": 1}).status_code == 422
    r = client.post(url, json={"records": [{"ip": "10.0.0.1", "action": "permit", "x": 1}]})
    assert r.status_code == 422


def test_unknown_analysis_is_404(client: TestClient) -> None:
    r = client.post(
        "/api/analyses/an_0000000000000000/traffic-audit",
        json={"records": [{"ip": "10.0.0.1", "action": "permit"}]},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Read-only behaviour
# ---------------------------------------------------------------------------
def test_audit_is_read_only_and_deterministic(client: TestClient) -> None:
    aid = _save(client, NESTED_RULES)
    before = client.get("/api/analyses").json()
    version_before = client.get(f"/api/analyses/{aid}").json()

    records = [
        {"ip": "10.1.1.9", "action": "permit"},
        {"ip": "192.0.2.1", "action": "deny"},
    ]
    first = _audit(client, aid, records)
    second = _audit(client, aid, records)
    assert first == second

    # No new analysis version was created and the base version is untouched.
    assert client.get("/api/analyses").json() == before
    assert client.get(f"/api/analyses/{aid}").json() == version_before
