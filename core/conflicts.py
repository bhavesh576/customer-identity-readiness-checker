"""Data-quality and cross-system conflict detectors with suggested remediations."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from core.matching import normalize_email, normalize_phone, _text

VALID_EVENT_TYPES = {
    "page_view", "product_view", "search", "add_to_cart", "checkout_started",
    "purchase", "login", "signup", "support_opened",
}

ISSUE_GUIDANCE: dict[str, dict[str, str]] = {
    "email_typo": {"severity": "medium", "explanation": "The email is close to a known CRM email but does not normalize to it. Exact joins may split one person into multiple identities.", "suggested_fix": "Verify against the customer or source system; correct only after confirmation."},
    "email_casing": {"severity": "low", "explanation": "The email contains uppercase letters. Case differences can cause avoidable mismatches in case-sensitive downstream joins.", "suggested_fix": "Normalize email casing in a derived clean field; retain the original value in the audit log."},
    "second_email": {"severity": "medium", "explanation": "The payment uses an alternate email that is already listed for the matched CRM customer. Joining only on the primary email would miss the link.", "suggested_fix": "Keep primary and alternate emails as explicit aliases; do not overwrite the primary address."},
    "name_variant": {"severity": "low", "explanation": "The name appears abbreviated, reordered, or shortened compared with the matched CRM profile. Exact-name joins may fail.", "suggested_fix": "Retain the source name and use a reviewed normalized-name field for matching."},
    "phone_format_variant": {"severity": "low", "explanation": "The phone appears to represent the same number as CRM but uses a different national or international format.", "suggested_fix": "Normalize to a derived E.164 field using known country context; verify ambiguous local numbers."},
    "duplicate_crm_customer": {"severity": "high", "explanation": "This CRM row appears to repeat a profile already present in the CRM dataset, potentially splitting activity across customer IDs.", "suggested_fix": "Review identifiers and source history, select a survivor ID, and merge only with an approved process."},
    "unknown_customer_id": {"severity": "high", "explanation": "The referenced customer ID does not exist in the uploaded CRM ID set. Joins and customer-level totals may be incomplete.", "suggested_fix": "Check upstream ID generation and reconcile against the authoritative customer registry."},
    "missing_id": {"severity": "high", "explanation": "The record lacks its expected identity key, so deterministic joins are weaker or impossible.", "suggested_fix": "Recover the source key where possible; otherwise keep the record unresolved rather than guessing."},
    "mixed_date_format": {"severity": "medium", "explanation": "The date is parseable but uses a different format from the standard ISO date/time representation. Locale settings can change its interpretation.", "suggested_fix": "Convert valid values to ISO-8601 in a derived column and record the original value."},
    "invalid_date": {"severity": "high", "explanation": "The date cannot be parsed as a valid calendar date/time. Time-series analysis and ordering may be wrong.", "suggested_fix": "Quarantine the value and confirm it with the upstream source; do not invent a replacement date."},
    "status_purchase_conflict": {"severity": "high", "explanation": "CRM says the customer is churned, but a completed purchase appears in the recent period. Churn and retention reporting may be inconsistent.", "suggested_fix": "Confirm the churn definition and event timing; reconcile lifecycle status using an agreed business rule."},
    "purchase_before_signup": {"severity": "high", "explanation": "A payment date precedes the available CRM signup date. If signup is invalid, the record-created timestamp may be only a review proxy; this conflict can distort cohort and lifetime calculations.", "suggested_fix": "Verify signup history, record-created time, timezone, and identity linkage before correcting either record."},
    "zero_amount": {"severity": "high", "explanation": "The payment amount is zero. It may be a test, free order, or incomplete transaction and can distort revenue and average order value.", "suggested_fix": "Confirm transaction semantics and exclude only according to the documented revenue definition."},
    "negative_amount": {"severity": "high", "explanation": "The payment amount is negative. Without an explicit refund/adjustment model, revenue and order metrics may be misleading.", "suggested_fix": "Classify refunds and adjustments separately; do not automatically replace the amount with a positive value."},
    "unknown_event_type": {"severity": "medium", "explanation": "The event type is outside the known event vocabulary. Aggregations may silently omit or misclassify it.", "suggested_fix": "Map it to an approved event taxonomy or quarantine it until the producer contract is updated."},
}


def _parse_datetime(value: Any) -> datetime | None:
    """Parse the common supported source formats without guessing invalid dates."""
    text = _text(value)
    if not text:
        return None
    formats = (
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y", "%b %d, %Y %I:%M %p",
        "%Y/%m/%d", "%Y/%m/%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_iso(value: Any, includes_time: bool = False) -> bool:
    """Check if a value begins with the project's standard ISO representation."""
    text = _text(value)
    if includes_time:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", text))
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))


def _emit(
    findings: list[dict[str, Any]],
    *, source: str, record_id: str, issue_type: str, field: str,
    observed_value: Any, affected_rows: str | None = None, related_record_id: str = "",
) -> None:
    """Append one standard finding while keeping original values untouched."""
    guidance = ISSUE_GUIDANCE[issue_type]
    findings.append({
        "source": source,
        "record_id": record_id,
        "issue_type": issue_type,
        "severity": guidance["severity"],
        "field": field,
        "observed_value": _text(observed_value),
        "affected_rows": affected_rows or f"{source}:{record_id}",
        "related_record_id": related_record_id,
        "explanation": guidance["explanation"],
        "suggested_fix": guidance["suggested_fix"],
    })


def _profile_for_record(
    record: pd.Series,
    source: str,
    matches_by_key: dict[tuple[str, str], str],
    crm_by_record: dict[str, pd.Series],
    crm_by_id: dict[str, pd.Series],
) -> pd.Series | None:
    """Resolve the most reliable CRM profile for a source record for diagnostics."""
    match_id = matches_by_key.get((source, _text(record.get("record_id", ""))), "")
    if match_id in crm_by_record:
        return crm_by_record[match_id]
    customer_id = _text(record.get("customer_id", "")).lower()
    if customer_id in crm_by_id:
        return crm_by_id[customer_id]
    context_id = _text(record.get("context_customer_id", "")).lower()
    if context_id in crm_by_id:
        return crm_by_id[context_id]
    observed_email = normalize_email(record.get("email", ""))
    if observed_email:
        for profile in crm_by_record.values():
            if observed_email in {normalize_email(profile.get("email", "")), normalize_email(profile.get("secondary_email", ""))}:
                return profile
    return None


def detect_conflicts(
    crm: pd.DataFrame,
    payments: pd.DataFrame,
    events: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Flag planted and general data issues without modifying the uploaded data."""
    as_of = as_of or date.today()
    crm_data = crm.reset_index(drop=True).copy()
    payment_data = payments.reset_index(drop=True).copy()
    event_data = events.reset_index(drop=True).copy()
    for frame in (crm_data, payment_data, event_data):
        for field in ("record_id", "customer_id", "anonymous_id", "name", "email", "secondary_email", "phone"):
            if field not in frame.columns:
                frame[field] = ""
    if "country" not in payment_data.columns:
        payment_data["country"] = ""
    if "currency" not in payment_data.columns:
        payment_data["currency"] = ""
    currency_country = {"AED": "United Arab Emirates", "SAR": "Saudi Arabia", "QAR": "Qatar", "BHD": "Bahrain", "KWD": "Kuwait", "OMR": "Oman"}
    payment_data.loc[payment_data["country"].map(_text).eq(""), "country"] = payment_data.loc[payment_data["country"].map(_text).eq(""), "currency"].map(lambda x: currency_country.get(_text(x).upper(), ""))

    crm_by_record = {_text(row.get("record_id", "")): row for _, row in crm_data.iterrows()}
    crm_by_id: dict[str, pd.Series] = {}
    for _, row in crm_data.iterrows():
        cid = _text(row.get("customer_id", "")).lower()
        if cid and not cid.startswith("cust-unknown"):
            crm_by_id.setdefault(cid, row)
    matches_by_key: dict[tuple[str, str], str] = {}
    if matches is not None and not matches.empty:
        for _, item in matches.iterrows():
            matched_id = _text(item.get("matched_crm_record_id", ""))
            if matched_id:
                matches_by_key[(_text(item.get("source", "")), _text(item.get("record_id", "")))] = matched_id

    findings: list[dict[str, Any]] = []

    # Duplicate CRM profiles: report later records that reuse a primary email or normalized phone.
    seen_profile_keys: dict[tuple[str, str], str] = {}
    for _, row in crm_data.iterrows():
        email = normalize_email(row.get("email", ""))
        phone = normalize_phone(row.get("phone", ""), _text(row.get("country", "")))
        key = (email, phone)
        if any(key) and key in seen_profile_keys:
            _emit(findings, source="crm", record_id=_text(row.get("record_id", "")), issue_type="duplicate_crm_customer", field="customer_id", observed_value=row.get("customer_id", ""), related_record_id=seen_profile_keys[key])
        elif any(key):
            seen_profile_keys[key] = _text(row.get("record_id", ""))

    # Missing CRM IDs are serious because the row cannot be joined deterministically.
    for _, row in crm_data.iterrows():
        rid = _text(row.get("record_id", ""))
        if not _text(row.get("customer_id", "")):
            _emit(findings, source="crm", record_id=rid, issue_type="missing_id", field="customer_id", observed_value="")
        raw_signup = _text(row.get("signup_date", ""))
        if raw_signup:
            parsed_signup = _parse_datetime(raw_signup)
            if parsed_signup is None:
                _emit(findings, source="crm", record_id=rid, issue_type="invalid_date", field="signup_date", observed_value=raw_signup)
            elif not _is_iso(raw_signup, includes_time=False):
                _emit(findings, source="crm", record_id=rid, issue_type="mixed_date_format", field="signup_date", observed_value=raw_signup)

    # Source-specific base checks that require no identity inference.
    for source, frame in (("payments", payment_data), ("events", event_data)):
        for _, row in frame.iterrows():
            rid = _text(row.get("record_id", ""))
            customer_id = _text(row.get("customer_id", ""))
            if customer_id and customer_id.lower() not in crm_by_id:
                _emit(findings, source=source, record_id=rid, issue_type="unknown_customer_id", field="customer_id", observed_value=customer_id)
            if source == "payments" and not customer_id:
                _emit(findings, source=source, record_id=rid, issue_type="missing_id", field="customer_id", observed_value="")
            if source == "events":
                has_identity = any(_text(row.get(field, "")) for field in ("anonymous_id", "customer_id", "email", "name", "phone"))
                if not has_identity:
                    _emit(findings, source=source, record_id=rid, issue_type="missing_id", field="anonymous_id / customer_id", observed_value="")
                event_type = _text(row.get("event_type", ""))
                if event_type and event_type not in VALID_EVENT_TYPES:
                    _emit(findings, source=source, record_id=rid, issue_type="unknown_event_type", field="event_type", observed_value=event_type)

    # Preserve source distinctions while running consistency checks against the likely CRM profile.
    for source, frame in (("payments", payment_data), ("events", event_data)):
        date_field = "transaction_date" if source == "payments" else "event_timestamp"
        expected_date_format_has_time = source == "events"
        for _, row in frame.iterrows():
            rid = _text(row.get("record_id", ""))
            raw_email = _text(row.get("email", ""))
            raw_name = _text(row.get("name", ""))
            raw_phone = _text(row.get("phone", ""))
            raw_date = _text(row.get(date_field, ""))
            profile = _profile_for_record(row, source, matches_by_key, crm_by_record, crm_by_id)

            if raw_email and raw_email != raw_email.lower():
                _emit(findings, source=source, record_id=rid, issue_type="email_casing", field="email", observed_value=raw_email)

            if raw_email and profile is not None:
                primary = normalize_email(profile.get("email", ""))
                secondary = normalize_email(profile.get("secondary_email", ""))
                normalized_email = normalize_email(raw_email)
                if normalized_email and secondary and normalized_email == secondary and normalized_email != primary:
                    _emit(findings, source=source, record_id=rid, issue_type="second_email", field="email", observed_value=raw_email, related_record_id=_text(profile.get("record_id", "")))
                elif normalized_email and normalized_email not in {primary, secondary}:
                    candidates = [candidate for candidate in (primary, secondary) if candidate]
                    similarity = max((fuzz.ratio(normalized_email, candidate) for candidate in candidates), default=0)
                    if 65 <= similarity < 100:
                        _emit(findings, source=source, record_id=rid, issue_type="email_typo", field="email", observed_value=raw_email, related_record_id=_text(profile.get("record_id", "")))

            if raw_name and profile is not None:
                expected_name = _text(profile.get("name", ""))
                if expected_name and raw_name.strip() != expected_name.strip() and fuzz.token_set_ratio(raw_name, expected_name) >= 60:
                    _emit(findings, source=source, record_id=rid, issue_type="name_variant", field="name", observed_value=raw_name, related_record_id=_text(profile.get("record_id", "")))

            if raw_phone and profile is not None:
                profile_phone = _text(profile.get("phone", ""))
                observed_normalized = normalize_phone(raw_phone, _text(row.get("country", "")))
                expected_normalized = normalize_phone(profile_phone, _text(profile.get("country", row.get("country", ""))))
                if observed_normalized and expected_normalized and observed_normalized == expected_normalized and _text(raw_phone) != _text(profile_phone):
                    _emit(findings, source=source, record_id=rid, issue_type="phone_format_variant", field="phone", observed_value=raw_phone, related_record_id=_text(profile.get("record_id", "")))

            if raw_date:
                parsed = _parse_datetime(raw_date)
                if parsed is None:
                    _emit(findings, source=source, record_id=rid, issue_type="invalid_date", field=date_field, observed_value=raw_date)
                elif not _is_iso(raw_date, includes_time=expected_date_format_has_time):
                    _emit(findings, source=source, record_id=rid, issue_type="mixed_date_format", field=date_field, observed_value=raw_date)

            if source == "payments":
                try:
                    amount = float(row.get("amount", ""))
                except (TypeError, ValueError):
                    amount = float("nan")
                if pd.notna(amount) and amount == 0:
                    _emit(findings, source=source, record_id=rid, issue_type="zero_amount", field="amount", observed_value=amount)
                elif pd.notna(amount) and amount < 0:
                    _emit(findings, source=source, record_id=rid, issue_type="negative_amount", field="amount", observed_value=amount)

    # Purchase-before-signup and lifecycle conflicts are cross-source checks.
    for _, row in payment_data.iterrows():
        rid = _text(row.get("record_id", ""))
        transaction_date = _parse_datetime(row.get("transaction_date", ""))
        profile = _profile_for_record(row, "payments", matches_by_key, crm_by_record, crm_by_id)
        if profile is None or transaction_date is None:
            continue
        signup_date = _parse_datetime(profile.get("signup_date", ""))
        if signup_date is None:
            # A valid CRM record-created timestamp can be a useful review proxy when signup_date is corrupted.
            signup_date = _parse_datetime(profile.get("created_at", ""))
        if signup_date and transaction_date.date() < signup_date.date():
            _emit(findings, source="payments", record_id=rid, issue_type="purchase_before_signup", field="transaction_date", observed_value=row.get("transaction_date", ""), related_record_id=_text(profile.get("record_id", "")))
        status = _text(profile.get("status", "")).lower()
        payment_status = _text(row.get("payment_status", "")).lower()
        days_old = (as_of - transaction_date.date()).days
        if status == "churned" and payment_status == "completed" and 0 <= days_old <= 30:
            _emit(findings, source="payments", record_id=rid, issue_type="status_purchase_conflict", field="status / transaction_date", observed_value=f"status={status}; completed transaction={_text(row.get('transaction_date', ''))}", related_record_id=_text(profile.get("record_id", "")))

    if not findings:
        return pd.DataFrame(columns=["source", "record_id", "issue_type", "severity", "field", "observed_value", "affected_rows", "related_record_id", "explanation", "suggested_fix"])
    result = pd.DataFrame(findings)
    # One detector may rediscover the same issue through multiple context routes; keep one finding per source/record/type/field.
    return result.drop_duplicates(subset=["source", "record_id", "issue_type", "field"], keep="first").reset_index(drop=True)
