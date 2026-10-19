# Customer Identity & Data Readiness Checker

An independent, explainable data-engineering portfolio prototype. It reads customer records from CRM, payment, and product-event exports, suggests cross-system identity links, flags data conflicts, and calculates a documented 0–100 readiness score.

> **Data and scope:** the checked-in sample data is fully synthetic and uses reserved example email domains. This is an independent prototype, not a claim about any company's internal systems, data quality, or customer identity process. Do not use the matching output as a production merge instruction without separate controls and human review.

## Features

- Column mapping with fuzzy header suggestions, editable mapping selections, YAML import/export, and validation.
- Deterministic identity matching using customer IDs, normalized email, normalized phone, and fuzzy name similarity.
- Blocking keys narrow candidate pairs instead of comparing every row against every CRM row.
- Confidence bands: **auto-link ≥85**, **needs review 60–84**, **no match <60**. Review-band candidates are never merged automatically.
- Data-quality checks for email typos/casing/alternate emails, name and phone variants, duplicate CRM profiles, unknown/missing IDs, mixed/invalid dates, lifecycle/purchase conflicts, purchase-before-signup, zero/negative amounts, and unknown event types.
- Readiness score with a published component breakdown and formula.
- Safe, logged normalization only: trim whitespace, normalize email/phone representations, standardize dates that parse unambiguously. Invalid values, customer IDs, statuses, amounts, and identity conflicts are not silently repaired.
- CSV downloads for matches, conflicts, cleaned source data, change logs, and a Markdown report.
- One-click sample-data load, with no API keys or paid services.

## Repository layout

```text
customer_identity_readiness_checker/
├── app.py
├── core/
│   ├── __init__.py
│   ├── schema.py
│   ├── matching.py
│   ├── conflicts.py
│   ├── scoring.py
│   └── report.py
├── config/
│   └── default_mapping.yaml
├── data/
│   ├── generate_data.py
│   └── sample/
│       ├── crm.csv
│       ├── payments.csv
│       ├── events.csv
│       └── ground_truth.csv
├── tests/
│   ├── conftest.py
│   ├── test_schema.py
│   ├── test_matching.py
│   ├── test_conflicts.py
│   ├── test_scoring_report.py
│   └── test_matching_metrics.py
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Run locally

Python 3.11 is the intended runtime.

```bash
python -m venv .venv
```

Activate the environment, then install pinned dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

From the app sidebar, select **Load sample data**. The page runs locally; no external API is called. To regenerate the reproducible sample dataset:

```bash
python data/generate_data.py
```

The generator uses seed `20261019` and a fixed reference date of `2026-10-19` by default. All generated CSVs are written to `data/sample/`.

Run tests from the repository root:

```bash
pytest -q -s
```

## How matching works

1. The schema layer maps uploaded column names into a common field set but does not overwrite uploaded values.
2. Contact fields are normalized for comparison: email is lowercased; Gmail dots and `+tags` are normalized only for Gmail/Googlemail domains; phones are converted to a best-effort E.164 form using GCC country or currency context.
3. A blocking index retrieves likely CRM candidates with exact customer-ID, email, phone, or narrow name keys. This avoids a full Cartesian join.
4. Candidate evidence is scored from ID, email, phone, and RapidFuzz name similarity. The displayed result includes the winning CRM row, confidence band, and human-readable reasons. A name-only resemblance is capped below the match threshold.
5. Events sharing an anonymous ID may use an authenticated sibling event to provide temporary matching context. Original event columns are not rewritten. Records without usable identity evidence remain unresolved.
6. Candidate scores are for prioritization, not proof. An auto-link label is a high-confidence recommendation only; ambiguous and review-band records are not merged by this prototype.

## Readiness score

The overall score is a weighted sum of four 0–100 component scores:

```text
Readiness = 0.30 × Identity Coverage
          + 0.25 × Completeness
          + 0.25 × Consistency
          + 0.20 × Validity
```

- **Identity coverage (30%)**: share of payment and event rows assigned a CRM candidate with confidence at least 60.
- **Completeness (25%)**: average complete-row rate across the uploaded sources. Event completeness requires a timestamp, event type, and at least one identity field.
- **Consistency (25%)**: 100 minus the share of distinct source rows flagged for identity/cross-system consistency issues.
- **Validity (20%)**: 100 minus the share of distinct source rows flagged for invalid dates, unknown event types, or zero/negative amounts.

Component definitions, affected-record counts, and the formula are shown in the UI/report. It is an operational heuristic, not an industry-standard certification score.

## Measured synthetic-data evaluation

The repository test suite currently passes **18 tests** (`18 passed`). The figures below are generated by the repository's test suite from the committed sample files and `ground_truth.csv`; they are not estimates. Identity matching is evaluated at the **record-to-CRM candidate** level. A downstream row is eligible for recall when it has at least one identity key (including an anonymous ID). A correct prediction means the proposed CRM row and source row have the same synthetic `true_customer_id`. Predictions include auto-link and needs-review candidates (confidence ≥60). The detector rate for each issue is the share of ground-truth planted issue labels found for the same source, record ID, and issue type.

<!-- MEASURED_RESULTS_START -->
**Matching results (`pytest -q -s`, seeded sample):**

| Measure | Actual result |
|---|---:|
| Eligible downstream records | 17,950 |
| True positive record-to-CRM links | 17,780 |
| False positive links | 0 |
| False negative links | 170 |
| Precision | 100.0000% |
| Recall | 99.0529% |

The figures include auto-link and needs-review suggestions at confidence ≥60. On this generated dataset, every suggested link mapped to the correct synthetic person; 170 eligible event records remained unlinked rather than being guessed. These are synthetic benchmark results, not a prediction of production performance.

**Detection rates for each planted issue type:**

| Planted issue | Detected / planted | Detection rate |
|---|---:|---:|
| Duplicate CRM customer | 30 / 30 | 100.0% |
| Email casing | 145 / 145 | 100.0% |
| Email typo | 85 / 85 | 100.0% |
| Invalid date | 130 / 130 | 100.0% |
| Missing ID | 92 / 92 | 100.0% |
| Mixed date format | 420 / 420 | 100.0% |
| Name variant | 390 / 390 | 100.0% |
| Negative amount | 25 / 25 | 100.0% |
| Phone format variant | 400 / 400 | 100.0% |
| Purchase before signup | 70 / 70 | 100.0% |
| Second email | 100 / 100 | 100.0% |
| Churn status vs recent purchase | 40 / 40 | 100.0% |
| Unknown customer ID | 70 / 70 | 100.0% |
| Unknown event type | 120 / 120 | 100.0% |
| Zero amount | 25 / 25 | 100.0% |

The detection rate is measured recall against the generator's labelled issue records (same source, record ID, and issue type). It does not measure the precision of every conflict class; unplanted findings are not penalized by this particular metric.
<!-- MEASURED_RESULTS_END -->

## Limitations and safety

- A confidence score cannot establish legal or business identity. Shared phones, recycled emails, family accounts, and duplicate contacts can create ambiguous links.
- Country/currency context is best-effort phone normalization; production systems should use a vetted phone-number library and explicit source-country metadata.
- Blocking can miss a true pair if every blocking identifier is missing, badly corrupted, or represented in a way that does not share a key.
- Fuzzy name and email similarities can create false positives; audit and review are required before any merge.
- The synthetic data is designed to exercise common defects and is not a representative estimate of real business data.
- No model training, external API, or paid service is used. The prototype should be extended with access controls, a durable review trail, source contracts, privacy review, and production observability before business use.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the complete project folder with `app.py` and `requirements.txt` at the repository root.
2. Open Streamlit Community Cloud and connect the GitHub account/repository.
3. Set the main file path to `app.py` and select Python 3.11 if the setting is available.
4. Deploy. The app installs the pinned packages from `requirements.txt` and uses the committed synthetic sample files.
5. Open the deployed URL and select **Load sample data**.

For a phone-only upload workflow, use GitHub's website in your mobile browser or GitHub mobile app: create the repository, choose **Add file → Upload files**, select the project files, then commit. If uploading many files is awkward on mobile, downloading the ZIP to the phone and extracting it first makes the upload picker easier to use. Do not upload real customer data, credentials, or secrets to a public repository.
