"""Tests that all planted issue categories are detected and described."""
from __future__ import annotations

from datetime import date

import pandas as pd

from core.conflicts import ISSUE_GUIDANCE, detect_conflicts


def test_sample_detection_rates_are_printed_and_measured(raw_sample, sample_analysis):
    truth = raw_sample["ground_truth"]
    planted = truth[truth["truth_type"] == "error"]
    detected = sample_analysis["conflicts"]
    truth_keys = set(zip(planted["source"], planted["record_id"], planted["issue_type"]))
    detected_keys = set(zip(detected["source"], detected["record_id"], detected["issue_type"]))
    print("\nDETECTION RATES BY PLANTED ERROR TYPE")
    for issue_type, group in planted.groupby("issue_type", sort=True):
        expected = set(zip(group["source"], group["record_id"], group["issue_type"]))
        hits = expected & detected_keys
        rate = len(hits) / len(expected) if expected else 1.0
        print(f"{issue_type}: {len(hits)}/{len(expected)} = {rate:.1%}")
        assert rate == 1.0, f"Expected to detect every planted {issue_type} label: {len(hits)}/{len(expected)}"
    assert truth_keys
    assert set(planted["issue_type"].unique()).issubset(set(ISSUE_GUIDANCE))
    if not detected.empty:
        assert detected["severity"].isin(["low", "medium", "high"]).all()
        assert detected["affected_rows"].map(bool).all()
        assert detected["explanation"].map(bool).all()
        assert detected["suggested_fix"].map(bool).all()


def test_empty_and_missing_columns_are_safe():
    empty = pd.DataFrame(columns=["record_id", "customer_id", "anonymous_id", "name", "email", "phone", "signup_date", "transaction_date", "event_timestamp", "amount", "status", "event_type", "country", "currency", "payment_status"])
    result = detect_conflicts(empty, empty, empty, as_of=date(2026, 10, 19))
    assert result.empty
    # Bare frames are allowed; absent optional columns are treated as unavailable.
    result_missing = detect_conflicts(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), as_of=date(2026, 10, 19))
    assert result_missing.empty
