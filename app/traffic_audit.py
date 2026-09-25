"""Traffic observation attribution ("traffic audit") on a saved analysis version.

Given a batch of *access observations* — each a bare IPv4/IPv6 address, the
action actually observed for it (``permit`` / ``deny``) and an optional
positive integer weight — this module attributes every observation to the
rule of the saved version that would decide it, and reconciles the observed
action with the expected one.

Attribution semantics
---------------------
Every address is adjudicated with the engine's own rule semantics — never by
the request's record order and never by a naive first-match scan:

* **longest prefix match** decides: among all rules whose CIDR contains the
  address, the longest prefix wins, regardless of priority numbers and of
  where the rule sits in the stored (submission) order;
* **same-CIDR ties** go to the smallest ``priority`` number, then to the
  original submission order — the same tie-break the analyzer uses to elect
  a node's winner (losing same-CIDR duplicates never attribute traffic);
* an address contained in no rule's CIDR is **uncovered**: it is reported
  explicitly with ``expected_action: null`` and is never *inferred* to be a
  permit or a deny.

Per record the response carries the winning ``rule_id``, the expected action
and whether the observed action is consistent with it.  On top of the
per-record verdicts, the weighted counts (each record's ``count``) are
aggregated three ways: per winning rule, per action mismatch (expected vs
observed transition), and for the uncovered surface.

The batch is all-or-nothing: one invalid ip/action/count rejects the whole
request with :class:`TrafficAuditError`, listing every offending record at
once.  The audit is read-only — no analysis version is created or modified.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any

from .analyzer import ValidationError, normalize_action

Address = ipaddress._BaseAddress
Network = ipaddress._BaseNetwork


class TrafficAuditError(Exception):
    """The observation batch is invalid as a whole.

    ``issues`` lists every offending record (bad ip, action or count) in
    request order, so a client can fix all problems at once.
    """

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class Observation:
    """One validated access observation."""

    index: int  # position in the request batch
    address: Address
    action: str  # normalized: permit | deny
    count: int  # positive weight


@dataclass(frozen=True)
class _Rule:
    """One stored rule of the saved version (``index`` = submission order)."""

    index: int
    rule_id: str
    network: Network
    action: str
    priority: int
    source_tag: str


# ---------------------------------------------------------------------------
# Observation parsing / validation
# ---------------------------------------------------------------------------
def parse_observations(raw_records: Any) -> list[Observation]:
    """Validate raw observation records and return them in request order.

    Every record must carry a bare IPv4/IPv6 address in ``ip`` (no prefix
    length), an observed ``action`` (the analyzer's permit/deny spellings,
    aliases accepted) and, optionally, a positive integer ``count`` (default
    1).  All problems are collected and raised as one
    :class:`TrafficAuditError` — the batch is never partially accepted.
    """
    if not isinstance(raw_records, list) or not raw_records:
        raise TrafficAuditError(["records must be a non-empty list"])

    issues: list[str] = []
    observations: list[Observation] = []
    for index, raw in enumerate(raw_records):
        label = f"record #{index + 1}"
        if not isinstance(raw, dict):
            issues.append(f"{label} must be an object")
            continue

        address: Address | None = None
        ip = raw.get("ip")
        if not isinstance(ip, str) or not ip.strip():
            issues.append(f"{label}: ip must be a non-empty string")
        else:
            try:
                # ip_address rejects anything with a prefix length, e.g.
                # '10.0.0.1/32' — a record carries a bare address.
                address = ipaddress.ip_address(ip.strip())
            except ValueError as exc:
                issues.append(f"{label}: invalid ip {ip!r} ({exc})")

        action: str | None = None
        try:
            action = normalize_action(raw.get("action"))
        except ValidationError as exc:
            issues.append(f"{label}: {exc}")

        count: int | None = raw.get("count", 1)
        # Reject booleans (bool is a subclass of int) and non-integers.
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            issues.append(f"{label}: count must be a positive integer, got {count!r}")
            count = None

        if address is not None and action is not None and count is not None:
            observations.append(
                Observation(index=index, address=address, action=action, count=count)
            )

    if issues:
        raise TrafficAuditError(issues)
    return observations


# ---------------------------------------------------------------------------
# Rule loading / adjudication (from the stored version, read-only)
# ---------------------------------------------------------------------------
def _load_rules(base_rules: list[dict[str, Any]]) -> list[_Rule]:
    records = sorted(base_rules, key=lambda r: r["index"])
    return [
        _Rule(
            index=raw["index"],
            rule_id=raw["rule_id"],
            network=ipaddress.ip_network(raw["cidr"]),
            action=raw["action"],
            priority=raw["priority"],
            source_tag=raw["source_tag"],
        )
        for raw in records
    ]


def _lpm_winner(rules: list[_Rule], address: Address) -> _Rule | None:
    """Winning rule for ``address`` under the engine's own semantics.

    Longest matching prefix wins; same-CIDR candidates are ordered by
    (priority, original submission index) and the first one wins.  ``None``
    means no rule's CIDR contains the address (``uncovered``).
    """
    longest = -1
    candidates: list[_Rule] = []
    for rule in rules:
        net = rule.network
        if net.version != address.version or address not in net:
            continue
        if net.prefixlen > longest:
            longest = net.prefixlen
            candidates = [rule]
        elif net.prefixlen == longest:
            candidates.append(rule)
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r.priority, r.index))
    return candidates[0]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit_traffic(base: dict[str, Any], observations: list[Observation]) -> dict[str, Any]:
    """Attribute ``observations`` to the winning rules of ``base``.

    Returns a response-shaped dict (see module docstring).  Per-record
    verdicts keep the request order; the aggregates are deterministic
    regardless of it.
    """
    rules = _load_rules(base.get("rules", []))
    rule_index = {rule.rule_id: rule.index for rule in rules}

    records: list[dict[str, Any]] = []
    by_rule: dict[str, dict[str, Any]] = {}
    mismatch_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    uncovered_records = 0
    uncovered_hits = 0
    consistent_records = 0
    consistent_hits = 0
    inconsistent_records = 0
    inconsistent_hits = 0

    for obs in observations:
        winner = _lpm_winner(rules, obs.address)
        if winner is None:
            records.append(
                {
                    "index": obs.index,
                    "ip": str(obs.address),
                    "observed_action": obs.action,
                    "count": obs.count,
                    "status": "uncovered",
                    "rule_id": None,
                    "expected_action": None,
                    "consistent": None,
                }
            )
            uncovered_records += 1
            uncovered_hits += obs.count
            continue

        consistent = obs.action == winner.action
        records.append(
            {
                "index": obs.index,
                "ip": str(obs.address),
                "observed_action": obs.action,
                "count": obs.count,
                "status": "covered",
                "rule_id": winner.rule_id,
                "expected_action": winner.action,
                "consistent": consistent,
            }
        )

        agg = by_rule.setdefault(
            winner.rule_id,
            {
                "rule_id": winner.rule_id,
                "cidr": str(winner.network),
                "action": winner.action,
                "source_tag": winner.source_tag,
                "records": 0,
                "hits": 0,
                "consistent_hits": 0,
                "inconsistent_hits": 0,
            },
        )
        agg["records"] += 1
        agg["hits"] += obs.count
        if consistent:
            agg["consistent_hits"] += obs.count
            consistent_records += 1
            consistent_hits += obs.count
        else:
            agg["inconsistent_hits"] += obs.count
            inconsistent_records += 1
            inconsistent_hits += obs.count
            key = (winner.rule_id, winner.action, obs.action)
            group = mismatch_groups.setdefault(
                key,
                {
                    "rule_id": winner.rule_id,
                    "expected_action": winner.action,
                    "observed_action": obs.action,
                    "records": 0,
                    "hits": 0,
                },
            )
            group["records"] += 1
            group["hits"] += obs.count

    # Deterministic aggregate order: the winning rules' original submission
    # order, then (for mismatches of one rule) the transition spelling.
    by_rule_list = sorted(by_rule.values(), key=lambda e: rule_index[e["rule_id"]])
    mismatch_list = sorted(
        mismatch_groups.values(),
        key=lambda e: (rule_index[e["rule_id"]], e["expected_action"], e["observed_action"]),
    )

    covered_records = consistent_records + inconsistent_records
    covered_hits = consistent_hits + inconsistent_hits
    return {
        "analysis_id": base.get("id"),
        "ruleset_hash": base.get("ruleset_hash"),
        "analyzer_version": base.get("analyzer_version"),
        "summary": {
            "record_count": len(observations),
            "total_hits": covered_hits + uncovered_hits,
            "covered_records": covered_records,
            "covered_hits": covered_hits,
            "uncovered_records": uncovered_records,
            "uncovered_hits": uncovered_hits,
            "consistent_records": consistent_records,
            "consistent_hits": consistent_hits,
            "inconsistent_records": inconsistent_records,
            "inconsistent_hits": inconsistent_hits,
        },
        "by_rule": by_rule_list,
        "mismatches": mismatch_list,
        "uncovered": {"records": uncovered_records, "hits": uncovered_hits},
        "records": records,
    }
