"""Shared fixtures for deterministic end-to-end tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.schema import load_mapping, standardize_source
from core.matching import match_records
from core.conflicts import detect_conflicts
from core.scoring import compute_readiness

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def raw_sample() -> dict[str, pd.DataFrame]:
    """Load the committed generated source tables and their evaluation labels."""
    return {
        "crm": pd.read_csv(SAMPLE_DIR / "crm.csv", dtype=object, keep_default_na=False),
        "payments": pd.read_csv(SAMPLE_DIR / "payments.csv", dtype=object, keep_default_na=False),
        "events": pd.read_csv(SAMPLE_DIR / "events.csv", dtype=object, keep_default_na=False),
        "ground_truth": pd.read_csv(SAMPLE_DIR / "ground_truth.csv", dtype=object, keep_default_na=False),
    }


@pytest.fixture(scope="session")
def canonical_sample(raw_sample):
    """Standardize sample tables with the checked-in default mapping."""
    mapping = load_mapping(ROOT / "config" / "default_mapping.yaml")
    return {
        source: standardize_source(raw_sample[source], source, mapping[source])
        for source in ("crm", "payments", "events")
    }


@pytest.fixture(scope="session")
def sample_analysis(canonical_sample):
    """Run the real matcher and detector once for shared measured metrics."""
    matches = match_records(
        canonical_sample["crm"], canonical_sample["payments"], canonical_sample["events"],
        auto_link_threshold=85, review_threshold=60,
    )
    conflicts = detect_conflicts(
        canonical_sample["crm"], canonical_sample["payments"], canonical_sample["events"],
        matches, as_of=__import__("datetime").date(2026, 10, 19),
    )
    readiness = compute_readiness(
        canonical_sample["crm"], canonical_sample["payments"], canonical_sample["events"],
        matches, conflicts,
    )
    return {"matches": matches, "conflicts": conflicts, "readiness": readiness}
