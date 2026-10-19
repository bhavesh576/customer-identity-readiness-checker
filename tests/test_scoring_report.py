"""Tests for the readiness formula and safe-cleaning audit trail."""
from __future__ import annotations

import pandas as pd

from core.report import clean_source_data, build_report_markdown
from core.scoring import compute_readiness


def test_readiness_score_is_bounded_and_formula_documented(canonical_sample, sample_analysis):
    result = sample_analysis["readiness"]
    assert 0 <= result["score"] <= 100
    assert "0.30" in result["formula"]
    assert set(result["components"]) == {"identity_coverage", "completeness", "consistency", "validity"}
    assert all(0 <= item["score"] <= 100 for item in result["components"].values())


def test_safe_cleanup_preserves_invalid_date_and_logs_changes():
    raw = pd.DataFrame({
        "transaction_id": ["P1", "P2"],
        "customer_email": [" USER+tag@GMAIL.COM ", "valid@example.com"],
        "customer_phone": ["050 123 4567", "0507654321"],
        "transaction_date": ["17/10/2026", "not-a-date"],
        "currency": ["AED", "AED"],
        "amount": [10, 20],
    })
    mapping = {"record_id": "transaction_id", "email": "customer_email", "phone": "customer_phone", "transaction_date": "transaction_date", "currency": "currency"}
    cleaned, log = clean_source_data(raw, "payments", mapping)
    assert cleaned.loc[0, "customer_email"] == "user@gmail.com"
    assert cleaned.loc[0, "transaction_date"] == "2026-10-17"
    assert cleaned.loc[1, "transaction_date"] == "not-a-date"
    assert not log.empty
    assert {"old_value", "new_value", "reason"}.issubset(log.columns)
    assert raw.loc[0, "customer_email"] == " USER+tag@GMAIL.COM "  # raw remains untouched


def test_empty_input_score_is_zero_and_report_renders():
    empty = pd.DataFrame(columns=["record_id", "customer_id", "name", "email", "phone"])
    readiness = compute_readiness(empty, empty, empty, pd.DataFrame(), pd.DataFrame())
    assert readiness["score"] == 0.0
    report = build_report_markdown(readiness, pd.DataFrame(), pd.DataFrame())
    assert "Customer Identity & Data Readiness Report" in report
    assert "Scope and limitations" in report
