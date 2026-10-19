"""Safe derived-data cleanup, change logs, and downloadable Markdown reports."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

from core.matching import normalize_email, normalize_phone, _text

CURRENCY_COUNTRY = {"AED": "United Arab Emirates", "SAR": "Saudi Arabia", "QAR": "Qatar", "BHD": "Bahrain", "KWD": "Kuwait", "OMR": "Oman"}


def _parse_date(value: Any) -> datetime | None:
    """Parse known common date formats used by source applications."""
    value_text = _text(value)
    if not value_text:
        return None
    formats = (
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y", "%b %d, %Y %I:%M %p", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value_text, fmt)
        except ValueError:
            continue
    return None


def _format_date(value: Any, includes_time: bool) -> str:
    """Standardize a parseable value and leave invalid values unchanged."""
    parsed = _parse_date(value)
    if parsed is None:
        return _text(value)
    if includes_time:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.strftime("%Y-%m-%d")


def clean_source_data(
    raw_frame: pd.DataFrame,
    source: str,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply safe normalizations to a copy and return row-level change log entries.

    Invalid date tokens, IDs, amounts, statuses, and conflicts are not repaired.
    Original uploaded data remains unchanged in the session.
    """
    cleaned = raw_frame.copy()
    changes: list[dict[str, str]] = []
    record_column = mapping.get("record_id", "")
    record_ids = cleaned[record_column].map(_text) if record_column in cleaned.columns else pd.Series([f"{source.upper()}-ROW-{i:07d}" for i in range(1, len(cleaned) + 1)], index=cleaned.index)

    def update(column: str, transform, reason: str) -> None:
        if not column or column not in cleaned.columns:
            return
        for idx in cleaned.index:
            old = cleaned.at[idx, column]
            if pd.isna(old):
                continue
            new = transform(old, idx)
            if _text(old) != _text(new):
                cleaned.at[idx, column] = new
                changes.append({
                    "source": source,
                    "record_id": _text(record_ids.loc[idx]),
                    "column": column,
                    "old_value": _text(old),
                    "new_value": _text(new),
                    "reason": reason,
                })

    # Trimming applies to text columns only and is intentionally logged.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        update(column, lambda value, _idx: _text(value), "Trim leading/trailing whitespace")

    for field in ("email", "secondary_email"):
        column = mapping.get(field, "")
        if column in cleaned.columns:
            update(column, lambda value, _idx: normalize_email(value), "Normalize email case and Gmail aliases where applicable")

    phone_column = mapping.get("phone", "")
    country_column = mapping.get("country", "")
    currency_column = mapping.get("currency", "")
    if phone_column in cleaned.columns:
        def phone_transform(value: Any, idx: Any) -> str:
            country = _text(cleaned.at[idx, country_column]) if country_column in cleaned.columns else ""
            if not country and currency_column in cleaned.columns:
                country = CURRENCY_COUNTRY.get(_text(cleaned.at[idx, currency_column]).upper(), "")
            return normalize_phone(value, country) or _text(value)
        update(phone_column, phone_transform, "Normalize phone to best-effort E.164 using country/currency context")

    for field, includes_time in (("signup_date", False), ("transaction_date", False), ("event_timestamp", True)):
        column = mapping.get(field, "")
        if column in cleaned.columns:
            update(column, lambda value, _idx, has_time=includes_time: _format_date(value, has_time), "Standardize a parseable date/time to ISO-8601; leave invalid tokens untouched")

    columns = ["source", "record_id", "column", "old_value", "new_value", "reason"]
    log = pd.DataFrame(changes, columns=columns)
    return cleaned, log


def build_report_markdown(
    readiness: dict[str, Any],
    matches: pd.DataFrame,
    conflicts: pd.DataFrame,
    change_log: pd.DataFrame | None = None,
    *,
    sources: dict[str, int] | None = None,
) -> str:
    """Render a portable Markdown snapshot of results for download."""
    lines = [
        "# Customer Identity & Data Readiness Report",
        "",
        "## Readiness score",
        f"**Overall score: {readiness.get('score', 0):.1f}/100**",
        "",
        f"Formula: `{readiness.get('formula', '')}`",
        "",
        "| Component | Score | Weight | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for name, component in readiness.get("components", {}).items():
        lines.append(f"| {name.replace('_', ' ').title()} | {component['score']:.1f} | {component['weight']:.0%} | {component['description']} |")
    lines.extend(["", "## Source row counts", "", "| Source | Rows |", "|---|---:|"])
    for source, count in (sources or {}).items():
        lines.append(f"| {source.title()} | {count:,} |")
    lines.extend(["", "## Identity matching", ""])
    if matches is None or matches.empty:
        lines.append("No identity matches were produced.")
    else:
        band_counts = matches["band"].value_counts().to_dict()
        lines.append("| Band | Count |\n|---|---:|")
        for band in ("auto-link", "needs review", "no match"):
            lines.append(f"| {band.title()} | {int(band_counts.get(band, 0)):,} |")
        lines.append("")
        lines.append("Review-band candidates are suggestions only and are never merged automatically.")
    lines.extend(["", "## Conflicts and data quality", ""])
    if conflicts is None or conflicts.empty:
        lines.append("No conflicts were detected by the configured checks.")
    else:
        counts = conflicts.groupby(["issue_type", "severity"]).size().reset_index(name="count").sort_values("count", ascending=False)
        lines.append("| Issue | Severity | Findings |\n|---|---|---:|")
        for _, row in counts.iterrows():
            lines.append(f"| {row['issue_type']} | {row['severity']} | {int(row['count']):,} |")
    lines.extend(["", "## Safe-cleaning audit", ""])
    change_count = 0 if change_log is None else len(change_log)
    lines.append(f"Logged cell changes: **{change_count:,}**. Normalization, trimming, and parseable date formatting are recorded. Unknown IDs, invalid dates, amounts, statuses, and identity conflicts are not silently repaired.")
    lines.extend([
        "", "## Scope and limitations", "",
        "This report is generated by an independent prototype using deterministic rules and synthetic sample data when sample files are loaded. Similarity scores are prioritization signals, not proof of identity. Human review is required for ambiguous pairs and any production merge.",
        "",
    ])
    return "\n".join(lines)
