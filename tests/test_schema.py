"""Tests for column mapping, validation, and canonicalization."""
from __future__ import annotations

import pandas as pd
import yaml

from core.schema import auto_suggest_mapping, load_mapping, read_csv_bytes, save_mapping, standardize_source, validate_mapping


def test_fuzzy_mapping_suggests_common_alternate_headers():
    columns = ["source_row_id", "client_name", "email_address", "mobile_number", "joined_date"]
    mapping = auto_suggest_mapping(columns, "crm")
    assert mapping["record_id"] == "source_row_id"
    assert mapping["name"] == "client_name"
    assert mapping["email"] == "email_address"
    assert mapping["phone"] == "mobile_number"
    assert mapping["signup_date"] == "joined_date"


def test_mapping_validation_finds_missing_required_field():
    mapping = {"record_id": "", "customer_id": "client_key"}
    errors = validate_mapping(mapping, "crm", ["client_key"])
    assert any("record_id" in error for error in errors)


def test_mapping_yaml_round_trip(tmp_path):
    mapping = {"crm": {"record_id": "id", "email": "mail"}, "events": {"record_id": "event_id"}}
    path = tmp_path / "mapping.yaml"
    save_mapping(mapping, path)
    assert load_mapping(path) == mapping
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == mapping


def test_standardization_preserves_rows_and_handles_missing_columns():
    frame = pd.DataFrame({"id": ["x-1", "x-2"], "mail": ["A@EXAMPLE.COM", ""]})
    standardized = standardize_source(frame, "payments", {"record_id": "id", "email": "mail"})
    assert len(standardized) == 2
    assert standardized.loc[0, "record_id"] == "x-1"
    assert standardized.loc[0, "email"] == "A@EXAMPLE.COM"
    assert standardized.loc[1, "customer_id"] == ""


def test_empty_csv_upload_is_reported_and_missing_columns_are_detected():
    import pytest

    with pytest.raises(ValueError, match="empty or has no header"):
        read_csv_bytes(b"", "empty upload")
    frame = read_csv_bytes(b"odd_a,odd_b\n1,2\n", "minimal upload")
    mapping = auto_suggest_mapping(list(frame.columns), "events")
    errors = validate_mapping(mapping, "events", list(frame.columns))
    assert any("record_id" in error for error in errors)
    assert any("event_type" in error for error in errors)
