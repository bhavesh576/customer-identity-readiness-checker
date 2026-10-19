"""Streamlit UI for the Customer Identity & Data Readiness Checker."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conflicts import detect_conflicts
from core.matching import DEFAULT_WEIGHTS, match_records
from core.report import build_report_markdown, clean_source_data
from core.schema import SOURCE_FIELDS, auto_suggest_mapping, read_csv_bytes, standardize_source, validate_mapping
from core.scoring import compute_readiness

SAMPLE_DIR = ROOT / "data" / "sample"

st.set_page_config(
    page_title="Customer Identity & Data Readiness Checker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {border: 1px solid #e1e4e8; padding: 0.8rem; border-radius: 0.35rem; background: #fafbfc;}
    div[data-testid="stAlert"] {border-radius: 0.35rem;}
    </style>""",
    unsafe_allow_html=True,
)


def _read_csv_bytes(payload: bytes, label: str) -> pd.DataFrame:
    """Read an uploaded CSV through the tested schema utility."""
    return read_csv_bytes(payload, label)


def _load_sample_frames() -> dict[str, pd.DataFrame]:
    """Load the bundled deterministic data without using absolute paths."""
    frames: dict[str, pd.DataFrame] = {}
    files = {"crm": "crm.csv", "payments": "payments.csv", "events": "events.csv"}
    for source, filename in files.items():
        path = SAMPLE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Bundled sample file is missing: {path.name}")
        frames[source] = pd.read_csv(path, dtype=object, keep_default_na=False)
    return frames


def _human_field(field: str) -> str:
    """Make schema keys easier to read in selectbox labels."""
    return field.replace("_", " ").title()


def _get_mapping_defaults(frame: pd.DataFrame, source: str, config_mapping: dict[str, Any]) -> dict[str, str]:
    """Prefer a valid default config, otherwise create fuzzy auto-suggestions."""
    suggested = auto_suggest_mapping(list(frame.columns), source)
    configured = config_mapping.get(source, {}) if isinstance(config_mapping, dict) else {}
    for field in SOURCE_FIELDS[source]:
        configured_column = str(configured.get(field, "") or "")
        if configured_column in frame.columns:
            suggested[field] = configured_column
    return suggested


def _canonicalize(
    frames: dict[str, pd.DataFrame],
    mappings: dict[str, dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Standardize all source frames using user-approved column selections."""
    return {source: standardize_source(frames[source], source, mappings[source]) for source in ("crm", "payments", "events")}


@st.cache_data(show_spinner=False, max_entries=8)
def _analyze_data(
    crm: pd.DataFrame,
    payments: pd.DataFrame,
    events: pd.DataFrame,
    auto_threshold: int,
    review_threshold: int,
    weight_id: int,
    weight_email: int,
    weight_phone: int,
    weight_name: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run the pure matching, conflict and scoring logic with cached inputs."""
    weights = {
        "id": float(weight_id), "email": float(weight_email),
        "phone": float(weight_phone), "name": float(weight_name),
    }
    matches = match_records(
        crm, payments, events,
        auto_link_threshold=auto_threshold,
        review_threshold=review_threshold,
        weights=weights,
    )
    conflicts = detect_conflicts(crm, payments, events, matches)
    readiness = compute_readiness(crm, payments, events, matches, conflicts)
    return matches, conflicts, readiness, pd.DataFrame()


def _download_button(label: str, data: bytes, filename: str, mime: str, key: str) -> None:
    """Keep download-button calls compact and consistently keyed."""
    st.download_button(label, data=data, file_name=filename, mime=mime, key=key, use_container_width=False)


def main() -> None:
    """Render uploads, mapping controls, analysis tabs, and safe downloads."""
    st.title("Customer Identity & Data Readiness Checker")
    st.caption("Rule-based identity matching, conflict detection, and data-readiness scoring.")

    if "sample_loaded" not in st.session_state:
        st.session_state.sample_loaded = False
    if "review_decisions" not in st.session_state:
        st.session_state.review_decisions = {}

    st.sidebar.header("Inputs")
    if st.sidebar.button("Load sample data", type="primary", use_container_width=True):
        st.session_state.sample_loaded = True
        st.session_state.pop("upload_error", None)
        st.rerun()
    st.sidebar.caption("Use the bundled synthetic UAE/GCC dataset or upload your own CSV files.")

    uploaders = {
        "crm": st.sidebar.file_uploader("CRM CSV", type=["csv"], key="upload_crm"),
        "payments": st.sidebar.file_uploader("Payments CSV", type=["csv"], key="upload_payments"),
        "events": st.sidebar.file_uploader("Product events CSV", type=["csv"], key="upload_events"),
    }
    frames: dict[str, pd.DataFrame] = {}
    input_errors: list[str] = []
    if any(uploaders.values()):
        for source, uploaded in uploaders.items():
            if uploaded is None:
                input_errors.append(f"Upload the {source} CSV too, or clear the other uploads to use sample data.")
                continue
            try:
                frames[source] = _read_csv_bytes(uploaded.getvalue(), f"{source.title()} file")
            except ValueError as exc:
                input_errors.append(str(exc))
    elif st.session_state.sample_loaded:
        try:
            frames = _load_sample_frames()
        except (OSError, ValueError) as exc:
            input_errors.append(str(exc))

    if input_errors:
        for message in input_errors:
            st.sidebar.error(message)
        st.info("Fix the upload problem or clear the uploads and select **Load sample data**. The app has not modified any data.")
        return

    if not frames:
        st.info("Select **Load sample data** in the sidebar, or upload one CSV each for CRM, payments, and product events.")
        st.markdown("**Data flow:** source CSVs → column mapping → explainable identity matching → conflict checks → readiness score → audited downloads.")
        st.stop()

    config_path = ROOT / "config" / "default_mapping.yaml"
    try:
        config_mapping = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config_mapping = {}

    mapping_upload = st.sidebar.file_uploader("Load mapping YAML (optional)", type=["yaml", "yml"], key="mapping_yaml_upload")
    if mapping_upload is not None:
        try:
            loaded_mapping = yaml.safe_load(mapping_upload.getvalue().decode("utf-8")) or {}
            if not isinstance(loaded_mapping, dict):
                raise ValueError("YAML must contain a mapping per source.")
            config_mapping = loaded_mapping
        except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
            st.sidebar.error(f"Could not load mapping YAML: {exc}")

    st.sidebar.subheader("Column mapping")
    mappings: dict[str, dict[str, str]] = {}
    for source in ("crm", "payments", "events"):
        frame = frames.get(source)
        if frame is None:
            continue
        defaults = _get_mapping_defaults(frame, source, config_mapping)
        mappings[source] = {}
        with st.sidebar.expander(f"{source.title()} fields", expanded=False):
            options = [""] + [str(column) for column in frame.columns]
            for field in SOURCE_FIELDS[source]:
                suggestion = defaults.get(field, "")
                index = options.index(suggestion) if suggestion in options else 0
                chosen = st.selectbox(
                    _human_field(field), options, index=index,
                    key=f"map_{source}_{field}_{abs(hash(tuple(frame.columns))) % 1000000}",
                    help="Select the source column that represents this canonical field. Leave blank if unavailable.",
                )
                mappings[source][field] = chosen

    mapping_payload = {source: mapping for source, mapping in mappings.items()}
    try:
        mapping_yaml = yaml.safe_dump(mapping_payload, sort_keys=False, allow_unicode=True).encode("utf-8")
        _download_button("Download current mapping YAML", mapping_yaml, "default_mapping.yaml", "text/yaml", "download_mapping_yaml")
    except (TypeError, yaml.YAMLError):
        st.sidebar.warning("Current mapping could not be serialized to YAML.")

    st.sidebar.subheader("Matching thresholds")
    auto_threshold = int(st.sidebar.slider("Auto-link threshold", min_value=61, max_value=100, value=85, step=1))
    review_threshold = int(st.sidebar.slider("Review threshold", min_value=0, max_value=84, value=60, step=1))
    st.sidebar.caption("Scores below the review threshold are not linked. Review-band candidates are never auto-merged.")
    st.sidebar.subheader("Feature weights")
    st.sidebar.caption("Weights are normalized automatically; set a field to 0 to ignore it.")
    weight_id = st.sidebar.slider("Customer ID weight", 0, 100, 38, 1)
    weight_email = st.sidebar.slider("Email weight", 0, 100, 30, 1)
    weight_phone = st.sidebar.slider("Phone weight", 0, 100, 22, 1)
    weight_name = st.sidebar.slider("Name weight", 0, 100, 10, 1)
    if weight_id + weight_email + weight_phone + weight_name == 0:
        st.sidebar.warning("At least one feature weight must be greater than zero; default weights will be used.")

    mapping_errors: list[str] = []
    for source in ("crm", "payments", "events"):
        mapping_errors.extend(validate_mapping(mappings.get(source, {}), source, list(frames[source].columns)))
    if mapping_errors:
        for error in mapping_errors:
            st.warning(error)

    try:
        canonical = _canonicalize(frames, mappings)
    except Exception as exc:
        st.error(f"Could not standardize the uploaded data: {exc}")
        st.stop()

    with st.spinner("Matching identities and checking data quality…"):
        try:
            matches, conflicts, readiness, _ = _analyze_data(
                canonical["crm"], canonical["payments"], canonical["events"],
                auto_threshold, review_threshold,
                weight_id, weight_email, weight_phone, weight_name,
            )
        except Exception as exc:
            st.error(f"Analysis failed safely: {type(exc).__name__}: {exc}")
            st.info("Check the column mapping and confirm that the CSV files contain readable headers and records.")
            st.stop()

    tabs = st.tabs(["Overview", "Identity Matches", "Conflicts", "Cleaned Data", "Report"])

    with tabs[0]:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Readiness score", f"{readiness['score']:.1f}/100")
        metric_columns[1].metric("CRM records", f"{len(canonical['crm']):,}")
        metric_columns[2].metric("Downstream records", f"{len(canonical['payments']) + len(canonical['events']):,}")
        metric_columns[3].metric("Detected issues", f"{len(conflicts):,}")
        st.subheader("Readiness breakdown")
        component_names = list(readiness["components"])
        component_values = [readiness["components"][name]["score"] for name in component_names]
        score_frame = pd.DataFrame({"Component": [name.replace("_", " ").title() for name in component_names], "Score": component_values})
        chart = px.bar(score_frame, x="Score", y="Component", orientation="h", range_x=[0, 100], text="Score")
        chart.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_color="#406a8e")
        chart.update_layout(height=260, margin=dict(l=8, r=20, t=10, b=10), xaxis_title="Score out of 100", yaxis_title="", showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
        for component_name, item in readiness["components"].items():
            st.write(f"**{component_name.replace('_', ' ').title()} — {item['score']:.1f}/100**  ")
            st.caption(item["description"])
        st.subheader("Identity match bands")
        if matches.empty:
            st.info("No downstream rows were available for matching.")
        else:
            band_counts = matches["band"].value_counts().reindex(["auto-link", "needs review", "no match"], fill_value=0)
            st.dataframe(band_counts.rename_axis("Band").reset_index(name="Records"), hide_index=True, use_container_width=True)
        if mapping_errors:
            st.warning("Some required fields are unmapped. Interpret results cautiously until those mappings are corrected.")

    with tabs[1]:
        st.subheader("Identity Matches")
        if matches.empty:
            st.info("No downstream records are available to match.")
        else:
            filters = st.columns([1, 1, 2])
            band_filter = filters[0].selectbox("Band", ["All", "auto-link", "needs review", "no match"], key="match_band_filter")
            source_filter = filters[1].selectbox("Source", ["All", "payments", "events"], key="match_source_filter")
            search_text = filters[2].text_input("Search record or CRM ID", key="match_search")
            view = matches.copy()
            if band_filter != "All":
                view = view[view["band"] == band_filter]
            if source_filter != "All":
                view = view[view["source"] == source_filter]
            if search_text.strip():
                query = search_text.strip().lower()
                view = view[view.apply(lambda row: query in str(row.get("record_id", "")).lower() or query in str(row.get("matched_crm_record_id", "")).lower() or query in str(row.get("matched_customer_id", "")).lower(), axis=1)]
            st.caption(f"Showing {min(len(view), 1_000):,} of {len(view):,} filtered rows.")
            st.dataframe(view.head(1_000), hide_index=True, use_container_width=True, height=420)
            _download_button("Download all identity matches CSV", matches.to_csv(index=False).encode("utf-8"), "identity_matches.csv", "text/csv", "download_matches_csv")

            review = matches[matches["band"] == "needs review"]
            st.divider()
            st.subheader("Manual review queue")
            if review.empty:
                st.info("There are no candidates in the review band with the current thresholds.")
            else:
                labels = [f"{row.source} | {row.record_id} → {row.matched_crm_record_id} | {row.confidence:.1f}" for row in review.itertuples(index=False)]
                selected_label = st.selectbox("Select a candidate", labels, key="review_candidate")
                selected_index = labels.index(selected_label)
                selected = review.iloc[selected_index]
                decision_key = f"{selected['source']}:{selected['record_id']}"
                current_decision = st.session_state.review_decisions.get(decision_key, "Pending")
                st.write(f"**Reason:** {selected['reasons']}")
                st.write(f"**Current decision:** {current_decision}")
                accept_col, reject_col = st.columns(2)
                if accept_col.button("Accept candidate", key="accept_review_candidate"):
                    st.session_state.review_decisions[decision_key] = "Accepted for review only"
                    st.rerun()
                if reject_col.button("Reject candidate", key="reject_review_candidate"):
                    st.session_state.review_decisions[decision_key] = "Rejected"
                    st.rerun()
                if st.session_state.review_decisions:
                    review_log = pd.DataFrame([{"source_record": key, "decision": value} for key, value in st.session_state.review_decisions.items()])
                    st.dataframe(review_log, hide_index=True, use_container_width=True)
                    _download_button("Download review decisions CSV", review_log.to_csv(index=False).encode("utf-8"), "review_decisions.csv", "text/csv", "download_review_decisions")
                st.caption("Accept records a human review decision only. It does not merge or rewrite any source record.")

    with tabs[2]:
        st.subheader("Conflicts and quality findings")
        if conflicts.empty:
            st.success("No issues were detected by the configured checks.")
        else:
            issue_types = ["All"] + sorted(conflicts["issue_type"].dropna().unique().tolist())
            issue_filter = st.selectbox("Issue type", issue_types, key="conflict_issue_filter")
            severity_filter = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"], key="conflict_severity_filter")
            conflict_view = conflicts.copy()
            if issue_filter != "All":
                conflict_view = conflict_view[conflict_view["issue_type"] == issue_filter]
            if severity_filter:
                conflict_view = conflict_view[conflict_view["severity"].isin(severity_filter)]
            severity_order = {"high": 0, "medium": 1, "low": 2}
            conflict_view = conflict_view.assign(_severity_order=conflict_view["severity"].map(severity_order)).sort_values(["_severity_order", "source", "record_id"]).drop(columns="_severity_order")
            st.caption(f"Showing {len(conflict_view):,} findings. Each finding includes affected rows, impact, and a suggested fix.")
            st.dataframe(conflict_view.head(2_000), hide_index=True, use_container_width=True, height=480)
            _download_button("Download all conflicts CSV", conflicts.to_csv(index=False).encode("utf-8"), "conflicts.csv", "text/csv", "download_conflicts_csv")

    with tabs[3]:
        st.subheader("Cleaned Data")
        st.caption("Only logged normalizations, trimming, and parseable date formatting are applied. IDs and business values are not guessed or merged.")
        source_choice = st.selectbox("Source", ["crm", "payments", "events"], key="clean_source_choice")
        cleaned, change_log = clean_source_data(frames[source_choice], source_choice, mappings[source_choice])
        clean_metrics = st.columns(3)
        clean_metrics[0].metric("Source rows", f"{len(cleaned):,}")
        clean_metrics[1].metric("Logged cell changes", f"{len(change_log):,}")
        clean_metrics[2].metric("Unchanged source file", "Preserved")
        st.dataframe(cleaned.head(500), hide_index=True, use_container_width=True, height=360)
        _download_button(f"Download cleaned {source_choice} CSV", cleaned.to_csv(index=False).encode("utf-8"), f"{source_choice}_cleaned.csv", "text/csv", f"download_clean_{source_choice}")
        st.subheader("Change log")
        if change_log.empty:
            st.info("No safe transformations were needed for this source.")
        else:
            st.dataframe(change_log.head(2_000), hide_index=True, use_container_width=True, height=300)
            _download_button(f"Download {source_choice} change log", change_log.to_csv(index=False).encode("utf-8"), f"{source_choice}_change_log.csv", "text/csv", f"download_log_{source_choice}")

    with tabs[4]:
        st.subheader("Downloadable report")
        combined_changes = []
        for source in ("crm", "payments", "events"):
            _, log = clean_source_data(frames[source], source, mappings[source])
            if not log.empty:
                combined_changes.append(log)
        full_change_log = pd.concat(combined_changes, ignore_index=True) if combined_changes else pd.DataFrame(columns=["source", "record_id", "column", "old_value", "new_value", "reason"])
        report = build_report_markdown(
            readiness, matches, conflicts, full_change_log,
            sources={source: len(frame) for source, frame in frames.items()},
        )
        st.markdown(report)
        _download_button("Download Markdown report", report.encode("utf-8"), "customer_identity_readiness_report.md", "text/markdown", "download_report_md")
        _download_button("Download readiness breakdown CSV", pd.DataFrame([{"component": key, **value} for key, value in readiness["components"].items()]).to_csv(index=False).encode("utf-8"), "readiness_breakdown.csv", "text/csv", "download_readiness_csv")

    st.divider()
    st.caption("Independent prototype. Sample data is synthetic; matching suggestions do not prove identity and are not a substitute for approved production merge controls.")


if __name__ == "__main__":
    main()
