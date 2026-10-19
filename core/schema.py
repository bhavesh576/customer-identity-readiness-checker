"""Column mapping and canonical schema utilities for source data."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rapidfuzz import fuzz

SOURCES = ("crm", "payments", "events")
CANONICAL_FIELDS = (
    "record_id", "customer_id", "anonymous_id", "name", "email",
    "secondary_email", "phone", "signup_date", "transaction_date",
    "event_timestamp", "amount", "status", "event_type", "country",
    "currency", "payment_status", "created_at",
)

SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "crm": (
        "record_id", "customer_id", "name", "email", "secondary_email",
        "phone", "signup_date", "status", "country", "created_at",
    ),
    "payments": (
        "record_id", "customer_id", "email", "name", "phone",
        "transaction_date", "amount", "currency", "payment_status",
    ),
    "events": (
        "record_id", "anonymous_id", "customer_id", "email", "name",
        "phone", "event_type", "event_timestamp", "country",
    ),
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "crm": ("record_id",),
    "payments": ("record_id",),
    "events": ("record_id", "event_type", "event_timestamp"),
}
IDENTITY_FIELD_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "crm": ("customer_id", "email", "phone", "name"),
    "payments": ("customer_id", "email", "phone", "name"),
    "events": ("anonymous_id", "customer_id", "email", "phone", "name"),
}

# Aliases are ordered by general usefulness; fuzzy suggestions are still reviewable.
ALIASES: dict[str, tuple[str, ...]] = {
    "record_id": ("record_id", "crm_record_id", "transaction_id", "event_id", "row_id", "id", "source_record_id", "source_row_id", "row_key"),
    "customer_id": ("customer_id", "customerid", "client_id", "user_id", "account_id", "person_id"),
    "anonymous_id": ("anonymous_id", "anon_id", "visitor_id", "device_user_id", "anonymous_user_id"),
    "name": ("name", "full_name", "customer_name", "user_name", "client_name", "person_name"),
    "email": ("email", "email_address", "customer_email", "user_email", "mail"),
    "secondary_email": ("secondary_email", "alternate_email", "alt_email", "backup_email"),
    "phone": ("phone", "phone_number", "mobile", "mobile_number", "customer_phone", "user_phone", "telephone"),
    "signup_date": ("signup_date", "sign_up_date", "registration_date", "joined_date", "join_date", "created_signup_date"),
    "transaction_date": ("transaction_date", "payment_date", "purchase_date", "order_date", "paid_at"),
    "event_timestamp": ("event_timestamp", "timestamp", "event_time", "occurred_at", "event_date"),
    "amount": ("amount", "transaction_amount", "payment_amount", "total", "value"),
    "status": ("status", "customer_status", "lifecycle_status", "account_status"),
    "event_type": ("event_type", "event_name", "action", "event", "activity_type"),
    "country": ("country", "country_name", "market", "region_country"),
    "currency": ("currency", "currency_code", "transaction_currency"),
    "payment_status": ("payment_status", "transaction_status", "payment_state"),
    "created_at": ("created_at", "created_on", "record_created_at"),
}


def _key(value: Any) -> str:
    """Normalize a header or alias for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def read_csv_bytes(payload: bytes, label: str = "CSV file") -> pd.DataFrame:
    """Read an uploaded CSV and raise a concise error for empty/malformed payloads."""
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=object, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty or has no header row.") from exc
    except Exception as exc:
        raise ValueError(f"Could not read {label} as CSV: {exc}") from exc
    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"{label} does not contain readable columns.")
    return frame


def auto_suggest_mapping(columns: list[str] | tuple[str, ...], source: str) -> dict[str, str]:
    """Suggest canonical-field to original-column mapping from header names.

    Exact alias matches are reserved before fuzzy matching so a header such as
    ``client_name`` cannot be consumed prematurely as a vague customer-ID match.
    """
    if source not in SOURCE_FIELDS:
        raise ValueError(f"Unknown source: {source}")
    available = [str(column) for column in columns]
    normalized = {_key(column): column for column in available}
    mapping: dict[str, str] = {field: "" for field in SOURCE_FIELDS[source]}
    used: set[str] = set()

    # Pass one: reserve all clear semantic matches.
    for field in SOURCE_FIELDS[source]:
        for alias in ALIASES.get(field, (field,)):
            direct = normalized.get(_key(alias))
            if direct and direct not in used:
                mapping[field] = direct
                used.add(direct)
                break

    # Pass two: fuzzy suggestion for fields still unmapped.
    for field in SOURCE_FIELDS[source]:
        if mapping[field]:
            continue
        candidates: list[tuple[int, int, str]] = []
        aliases = ALIASES.get(field, (field,))
        for alias_index, alias in enumerate(aliases):
            for column in available:
                if column in used:
                    continue
                left, right = _key(column), _key(alias)
                score = max(
                    fuzz.ratio(left, right),
                    fuzz.partial_ratio(left, right) if min(len(left), len(right)) >= 5 else 0,
                )
                candidates.append((int(score), -alias_index, column))
        if candidates:
            score, _, column = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            if score >= 72 and column not in used:
                mapping[field] = column
                used.add(column)
    return mapping


def validate_mapping(mapping: dict[str, str], source: str, columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return mapping validation errors; an empty list means required fields pass."""
    if source not in SOURCE_FIELDS:
        return [f"Unknown source '{source}'."]
    errors: list[str] = []
    colset = set(str(c) for c in columns)
    for field in REQUIRED_FIELDS[source]:
        selected = mapping.get(field, "")
        if not selected:
            errors.append(f"{source}: map the required field '{field}'.")
        elif selected not in colset:
            errors.append(f"{source}: selected column '{selected}' for '{field}' is not present in the file.")
    identity_fields = IDENTITY_FIELD_ALTERNATIVES[source]
    if not any(mapping.get(field, "") in colset for field in identity_fields):
        errors.append(f"{source}: map at least one identity field: {', '.join(identity_fields)}.")
    selected_values = [value for key, value in mapping.items() if key in SOURCE_FIELDS[source] and value]
    duplicates = sorted({value for value in selected_values if selected_values.count(value) > 1})
    if duplicates:
        errors.append(f"{source}: column(s) mapped more than once: {', '.join(duplicates)}.")
    return errors


def save_mapping(mapping: dict[str, dict[str, str]], path: str | Path) -> None:
    """Save the user-approved mappings as readable YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_mapping(path: str | Path) -> dict[str, dict[str, str]]:
    """Load mapping YAML and reject malformed top-level structures."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Mapping YAML must contain a dictionary of sources.")
    result: dict[str, dict[str, str]] = {}
    for source, values in payload.items():
        if source not in SOURCES or not isinstance(values, dict):
            raise ValueError(f"Invalid mapping section: {source!r}")
        result[source] = {str(key): str(value or "") for key, value in values.items()}
    return result


def standardize_source(
    frame: pd.DataFrame,
    source: str,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a canonical view without modifying source values or rows."""
    if source not in SOURCE_FIELDS:
        raise ValueError(f"Unknown source: {source}")
    chosen = mapping if mapping is not None else auto_suggest_mapping(list(frame.columns), source)
    output = pd.DataFrame(index=frame.index)
    for field in CANONICAL_FIELDS:
        column = chosen.get(field, "")
        if column and column in frame.columns:
            output[field] = frame[column]
        else:
            output[field] = ""
    output["source"] = source
    # Keep values as strings where they are identifiers/contact fields; do not lose leading zeroes.
    for field in ("record_id", "customer_id", "anonymous_id", "name", "email", "secondary_email", "phone"):
        output[field] = output[field].map(_empty_to_string)
    if source == "payments":
        currency_country = {"AED": "United Arab Emirates", "SAR": "Saudi Arabia", "QAR": "Qatar", "BHD": "Bahrain", "KWD": "Kuwait", "OMR": "Oman"}
        missing_country = output["country"].astype(str).str.strip().eq("")
        output.loc[missing_country, "country"] = output.loc[missing_country, "currency"].astype(str).str.upper().map(currency_country).fillna("")
    blank_record_ids = output["record_id"].astype(str).str.strip().eq("")
    if blank_record_ids.any():
        fallback = [f"{source.upper()}-ROW-{i:07d}" for i in range(1, len(output) + 1)]
        output.loc[blank_record_ids, "record_id"] = pd.Series(fallback, index=output.index).loc[blank_record_ids]
    return output.reset_index(drop=True)


def _empty_to_string(value: Any) -> str:
    """Convert missing identifier-like values to empty strings without 'nan' text."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text
