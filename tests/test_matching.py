"""Tests for deterministic identity normalization and scored matching."""
from __future__ import annotations

import pandas as pd
import pytest

from core.matching import match_records, normalize_email, normalize_phone
from core.schema import standardize_source


def test_email_normalization_applies_gmail_alias_rules_only_to_gmail():
    assert normalize_email("First.Last+shop@gmail.com") == "firstlast@gmail.com"
    assert normalize_email("User+tag@example.com") == "user+tag@example.com"


def test_phone_normalization_handles_common_uae_formats():
    assert normalize_phone("+971 50 123 4567", "United Arab Emirates") == "+971501234567"
    assert normalize_phone("050 123 4567", "United Arab Emirates") == "+971501234567"
    assert normalize_phone("971501234567", "United Arab Emirates") == "+971501234567"


def test_exact_id_match_is_explainable_and_auto_linked():
    crm_raw = pd.DataFrame([{"crm_record_id": "C1", "customer_id": "CUST-1", "full_name": "Ahmed Khan", "email": "ahmed@example.com", "phone": "+971501234567", "signup_date": "2025-01-01", "status": "active", "country": "United Arab Emirates"}])
    payment_raw = pd.DataFrame([{"transaction_id": "P1", "customer_id": "CUST-1", "customer_name": "Ahmed Khan", "customer_email": "AHMED@example.com", "customer_phone": "+971 50 123 4567", "transaction_date": "2025-02-01", "amount": "10", "currency": "AED", "payment_status": "completed"}])
    events_raw = pd.DataFrame(columns=["event_id", "anonymous_id", "customer_id", "user_email", "user_name", "user_phone", "event_type", "event_timestamp", "country"])
    crm = standardize_source(crm_raw, "crm", {"record_id": "crm_record_id", "customer_id": "customer_id", "name": "full_name", "email": "email", "phone": "phone", "signup_date": "signup_date", "status": "status", "country": "country"})
    payments = standardize_source(payment_raw, "payments", {"record_id": "transaction_id", "customer_id": "customer_id", "name": "customer_name", "email": "customer_email", "phone": "customer_phone", "transaction_date": "transaction_date", "amount": "amount", "currency": "currency", "payment_status": "payment_status"})
    events = standardize_source(events_raw, "events", {"record_id": "event_id"})
    result = match_records(crm, payments, events)
    assert result.iloc[0]["matched_crm_record_id"] == "C1"
    assert result.iloc[0]["band"] == "auto-link"
    assert "Customer ID agrees" in result.iloc[0]["reasons"]


def test_name_only_candidate_never_auto_links():
    crm = pd.DataFrame([{"record_id": "C1", "customer_id": "C1", "name": "Ahmed Khan", "email": "", "secondary_email": "", "phone": "", "country": ""}])
    payments = pd.DataFrame([{"record_id": "P1", "customer_id": "", "name": "Ahmed Khan", "email": "", "phone": "", "country": ""}])
    events = pd.DataFrame(columns=["record_id", "customer_id", "anonymous_id", "name", "email", "phone"])
    result = match_records(crm, payments, events)
    if not result.empty:
        assert result.iloc[0]["confidence"] < 60
        assert result.iloc[0]["band"] == "no match"


def test_thresholds_must_be_ordered():
    crm = pd.DataFrame(columns=["record_id", "customer_id", "name", "email", "phone"])
    with pytest.raises(ValueError):
        match_records(crm, crm, crm, auto_link_threshold=60, review_threshold=60)


def test_empty_inputs_do_not_crash():
    empty = pd.DataFrame(columns=["record_id", "customer_id", "name", "email", "phone"])
    result = match_records(empty, empty, empty)
    assert result.empty
