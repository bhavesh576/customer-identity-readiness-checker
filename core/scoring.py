"""Transparent, documented 0-100 data readiness scoring."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.matching import _text

CONSISTENCY_ISSUES = {
    "email_typo", "email_casing", "second_email", "name_variant",
    "phone_format_variant", "duplicate_crm_customer", "unknown_customer_id",
    "status_purchase_conflict", "purchase_before_signup",
}
VALIDITY_ISSUES = {"invalid_date", "unknown_event_type", "zero_amount", "negative_amount"}


def _row_complete(frame: pd.DataFrame, fields: tuple[str, ...]) -> float:
    """Return the share of rows with all required fields populated."""
    if frame.empty:
        return 100.0
    present_fields = [field for field in fields if field in frame.columns]
    if not present_fields:
        return 0.0
    complete = frame[present_fields].apply(lambda column: column.map(lambda value: bool(_text(value))))
    return float(complete.all(axis=1).mean() * 100.0)


def compute_readiness(
    crm: pd.DataFrame,
    payments: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate the documented weighted data-readiness score and its evidence.

    Formula: 30% identity coverage + 25% completeness + 25% consistency + 20% validity.
    Component scores are bounded to 0..100. An empty source is treated as unavailable,
    not as perfect: the completeness component then averages the source scores present.
    """
    source_rows = len(crm) + len(payments) + len(events)
    downstream_rows = len(payments) + len(events)
    if source_rows == 0:
        weights = {"identity_coverage": 0.30, "completeness": 0.25, "consistency": 0.25, "validity": 0.20}
        components = {
            "identity_coverage": {"score": 0.0, "weight": weights["identity_coverage"], "description": "No source rows were supplied."},
            "completeness": {"score": 0.0, "weight": weights["completeness"], "description": "No source rows were supplied."},
            "consistency": {"score": 0.0, "weight": weights["consistency"], "description": "No source rows were supplied."},
            "validity": {"score": 0.0, "weight": weights["validity"], "description": "No source rows were supplied."},
        }
        return {"score": 0.0, "components": components, "weights": weights, "total_source_rows": 0, "downstream_rows": 0, "identity_coverage_count": 0, "consistency_affected_rows": 0, "validity_affected_rows": 0, "formula": "0.30 × identity coverage + 0.25 × completeness + 0.25 × consistency + 0.20 × validity"}
    if downstream_rows and not matches.empty and "band" in matches.columns:
        linked = int(matches["band"].isin(["auto-link", "needs review"]).sum())
        identity_score = 100.0 * linked / downstream_rows
    else:
        identity_score = 0.0 if downstream_rows else 100.0

    source_completeness: list[tuple[str, float]] = []
    if not crm.empty:
        source_completeness.append(("CRM", _row_complete(crm, ("customer_id", "name", "email", "phone", "status", "signup_date"))))
    if not payments.empty:
        source_completeness.append(("Payments", _row_complete(payments, ("customer_id", "email", "name", "phone", "transaction_date", "amount", "payment_status"))))
    if not events.empty:
        has_identity = events.apply(
            lambda row: any(_text(row.get(field, "")) for field in ("anonymous_id", "customer_id", "email", "name", "phone")),
            axis=1,
        ) if len(events) else pd.Series(dtype=bool)
        event_complete = 100.0 * float((has_identity & events.get("event_type", pd.Series("", index=events.index)).map(lambda x: bool(_text(x))) & events.get("event_timestamp", pd.Series("", index=events.index)).map(lambda x: bool(_text(x)))).mean()) if len(events) else 100.0
        source_completeness.append(("Events", event_complete))
    completeness_score = sum(score for _, score in source_completeness) / len(source_completeness) if source_completeness else 0.0

    conflicts = conflicts if conflicts is not None else pd.DataFrame()
    if conflicts.empty or source_rows == 0:
        consistency_score = 100.0 if source_rows == 0 else 100.0
        validity_score = 100.0 if source_rows == 0 else 100.0
        consistency_affected = 0
        validity_affected = 0
    else:
        consistency_rows = conflicts[conflicts["issue_type"].isin(CONSISTENCY_ISSUES)]
        validity_rows = conflicts[conflicts["issue_type"].isin(VALIDITY_ISSUES)]
        consistency_affected = int(consistency_rows[["source", "record_id"]].drop_duplicates().shape[0]) if not consistency_rows.empty else 0
        validity_affected = int(validity_rows[["source", "record_id"]].drop_duplicates().shape[0]) if not validity_rows.empty else 0
        consistency_score = max(0.0, 100.0 * (1.0 - consistency_affected / max(source_rows, 1)))
        validity_score = max(0.0, 100.0 * (1.0 - validity_affected / max(source_rows, 1)))

    weights = {"identity_coverage": 0.30, "completeness": 0.25, "consistency": 0.25, "validity": 0.20}
    components = {
        "identity_coverage": {"score": round(identity_score, 2), "weight": weights["identity_coverage"], "description": f"Downstream rows assigned to a CRM candidate at confidence 60 or higher ({int(round(identity_score * downstream_rows / 100)):,}/{downstream_rows:,})."},
        "completeness": {"score": round(completeness_score, 2), "weight": weights["completeness"], "description": "Average complete-row rate across the uploaded sources; event rows need a timestamp, event type, and at least one identity field."},
        "consistency": {"score": round(consistency_score, 2), "weight": weights["consistency"], "description": f"100 minus the share of distinct source records affected by identity/cross-system consistency issues ({consistency_affected:,} records)."},
        "validity": {"score": round(validity_score, 2), "weight": weights["validity"], "description": f"100 minus the share of distinct source records affected by invalid dates, unknown event types, or zero/negative amounts ({validity_affected:,} records)."},
    }
    overall = sum(item["score"] * item["weight"] for item in components.values())
    return {
        "score": round(max(0.0, min(100.0, overall)), 1),
        "components": components,
        "weights": weights,
        "total_source_rows": source_rows,
        "downstream_rows": downstream_rows,
        "identity_coverage_count": int(round(identity_score * downstream_rows / 100.0)),
        "consistency_affected_rows": consistency_affected,
        "validity_affected_rows": validity_affected,
        "formula": "0.30 × identity coverage + 0.25 × completeness + 0.25 × consistency + 0.20 × validity",
    }
