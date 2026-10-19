"""Explainable, block-based customer identity matching; no trained model is used."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

DEFAULT_WEIGHTS: dict[str, float] = {"id": 0.38, "email": 0.30, "phone": 0.22, "name": 0.10}
COUNTRY_DIAL_CODES = {
    "united arab emirates": "971", "uae": "971", "saudi arabia": "966",
    "qatar": "974", "bahrain": "973", "kuwait": "965", "oman": "968",
}
COUNTRY_CURRENCIES = {"AED": "United Arab Emirates", "SAR": "Saudi Arabia", "QAR": "Qatar", "BHD": "Bahrain", "KWD": "Kuwait", "OMR": "Oman"}


def _text(value: Any) -> str:
    """Return an empty string for null-like values and trim ordinary strings."""
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", result):
        result = result[:-2]
    return result


def normalize_email(value: Any) -> str:
    """Lowercase email values and apply Gmail dot/plus normalization only on Gmail domains."""
    email = _text(value).lower()
    if not email or "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    domain = domain.strip()
    local = local.strip()
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.split("+", 1)[0].replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def normalize_phone(value: Any, country: Any = "") -> str:
    """Normalize GCC phone formats to a best-effort E.164 representation."""
    raw = _text(value)
    if not raw:
        return ""
    raw = raw.replace("\u00a0", " ")
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if raw.lstrip().startswith("00") and digits.startswith("00"):
        digits = digits[2:]
        return f"+{digits}" if len(digits) >= 8 else digits
    if raw.lstrip().startswith("+"):
        return f"+{digits}" if len(digits) >= 8 else digits
    # Numeric values may have lost the leading '+' during CSV type inference.
    for dial_code in sorted(set(COUNTRY_DIAL_CODES.values()), key=len, reverse=True):
        if digits.startswith(dial_code) and len(digits) >= len(dial_code) + 7:
            return f"+{digits}"
    normalized_country = _text(country).lower()
    dial_code = COUNTRY_DIAL_CODES.get(normalized_country, "")
    if not dial_code:
        return digits
    # UAE and Saudi phone numbers are often exported with a national trunk zero.
    if normalized_country in {"united arab emirates", "uae", "saudi arabia"} and digits.startswith("0"):
        digits = digits[1:]
    if digits.startswith(dial_code) and len(digits) >= len(dial_code) + 7:
        return f"+{digits}"
    if len(digits) >= 7:
        return f"+{dial_code}{digits}"
    return digits


def normalize_name(value: Any) -> str:
    """Lowercase names and retain only alphanumeric tokens and spaces."""
    return " ".join(_name_tokens(value))


def _name_tokens(value: Any) -> list[str]:
    """Tokenize a name after removing punctuation."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", _text(value).lower())
    return [token for token in normalized.split() if token]


def _name_keys(value: Any) -> set[str]:
    """Create narrow name blocks for exact, reordered, abbreviated, and initial names."""
    tokens = _name_tokens(value)
    if len(tokens) < 2:
        return set()
    first, last = tokens[0], tokens[-1]
    keys = {"name-sort:" + " ".join(sorted(tokens))}
    keys.add(f"name-first-last:{first}:{last}")
    keys.add(f"name-initial-last:{first[0]}:{last}")
    keys.add(f"name-first-lastinitial:{first}:{last[0]}")
    # An initial-only first name such as "A. Khan" still shares this key.
    if len(first) == 1:
        keys.add(f"name-initial-last:{first}:{last}")
    return keys


def _country(row: pd.Series) -> str:
    """Return country context, deriving it from currency for payment rows."""
    country = _text(row.get("country", ""))
    if country:
        return country
    return COUNTRY_CURRENCIES.get(_text(row.get("currency", "")).upper(), "")


def _field(row: pd.Series, field: str) -> str:
    """Get a normalized source field for blocking and scoring."""
    return _text(row.get(field, ""))


def _crm_index_fields(crm: pd.DataFrame) -> dict[str, list[int]]:
    """Index CRM rows by exact keys to avoid a Cartesian comparison."""
    index: dict[str, list[int]] = defaultdict(list)
    for position, (_, row) in enumerate(crm.iterrows()):
        customer_id = _field(row, "customer_id").lower()
        if customer_id and not customer_id.startswith("cust-unknown"):
            index[f"id:{customer_id}"].append(position)
        for field in ("email", "secondary_email"):
            email = normalize_email(row.get(field, ""))
            if email:
                index[f"email:{email}"].append(position)
        phone = normalize_phone(row.get("phone", ""), _country(row))
        if phone:
            index[f"phone:{phone}"].append(position)
        for key in _name_keys(row.get("name", "")):
            index[key].append(position)
    return index


def _row_block_keys(row: pd.Series, *, include_context: bool = True) -> set[str]:
    """Produce keys that can retrieve plausible CRM candidates for one source row."""
    keys: set[str] = set()
    values = {field: _field(row, field) for field in ("customer_id", "email", "name", "phone")}
    if include_context:
        for field in values:
            context_value = _field(row, f"context_{field}")
            if not values[field] and context_value:
                values[field] = context_value
            elif context_value:
                # Context keys add candidates but do not overwrite a source's observed value.
                if field == "customer_id":
                    keys.add(f"id:{context_value.lower()}")
                elif field == "email":
                    keys.add(f"email:{normalize_email(context_value)}")
                elif field == "phone":
                    keys.add(f"phone:{normalize_phone(context_value, _country(row))}")
                elif field == "name":
                    keys.update(_name_keys(context_value))
    if values["customer_id"] and not values["customer_id"].lower().startswith("cust-unknown"):
        keys.add(f"id:{values['customer_id'].lower()}")
    for email_value in (values["email"], _field(row, "secondary_email")):
        email = normalize_email(email_value)
        if email:
            keys.add(f"email:{email}")
    phone = normalize_phone(values["phone"], _country(row))
    if phone:
        keys.add(f"phone:{phone}")
    keys.update(_name_keys(values["name"]))
    return keys


def _mode(values: list[str], normalizer) -> str:
    """Return a deterministic modal value, ignoring blanks."""
    clean = [value for value in values if _text(value)]
    if not clean:
        return ""
    normalized = [normalizer(value) for value in clean]
    counts = Counter(value for value in normalized if value)
    if not counts:
        return ""
    best = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best)
    return winners[0]


def add_event_cohort_context(events: pd.DataFrame, crm: pd.DataFrame) -> pd.DataFrame:
    """Add temporary match-only context from a shared anonymous-ID cohort.

    Source columns are not edited. An authenticated event anchors the cohort first;
    otherwise exact CRM email, phone, or full-token name evidence can anchor it when
    that evidence points to one unambiguous CRM profile. Failing that, common observed
    contact values are used only as candidate-generation context.
    """
    output = events.copy().reset_index(drop=True)
    for field in ("customer_id", "email", "name", "phone"):
        output[f"context_{field}"] = ""
    if "anonymous_id" not in output.columns or output.empty:
        return output

    crm_by_id: dict[str, pd.Series] = {}
    email_profiles: dict[str, list[pd.Series]] = defaultdict(list)
    phone_profiles: dict[str, list[pd.Series]] = defaultdict(list)
    name_profiles: dict[str, list[pd.Series]] = defaultdict(list)
    for _, profile in crm.iterrows():
        cid = _field(profile, "customer_id").lower()
        if cid and not cid.startswith("cust-unknown"):
            crm_by_id.setdefault(cid, profile)
        for email_field in ("email", "secondary_email"):
            email = normalize_email(profile.get(email_field, ""))
            if email:
                email_profiles[email].append(profile)
        phone = normalize_phone(profile.get("phone", ""), _country(profile))
        if phone:
            phone_profiles[phone].append(profile)
        tokens = _name_tokens(profile.get("name", ""))
        if len(tokens) >= 2:
            name_profiles[" ".join(sorted(tokens))].append(profile)

    def unique_profile(profiles: list[pd.Series]) -> pd.Series | None:
        """Return a profile only if all anchors point to one customer/email-phone identity."""
        if not profiles:
            return None
        identities = {
            (_field(p, "customer_id").lower(), normalize_email(p.get("email", "")), normalize_phone(p.get("phone", ""), _country(p)))
            for p in profiles
        }
        if len(identities) != 1:
            return None
        return profiles[0]

    groups = output.groupby(output["anonymous_id"].map(_text), sort=False).groups
    for anonymous_id, positions in groups.items():
        if not anonymous_id:
            continue
        rows = output.loc[list(positions)]
        candidate_ids = [
            _field(row, "customer_id").lower()
            for _, row in rows.iterrows()
            if _field(row, "customer_id") and _field(row, "customer_id").lower() in crm_by_id
        ]
        profile: pd.Series | None = None
        chosen_id = ""
        if candidate_ids:
            counts = Counter(candidate_ids)
            most_common = counts.most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                chosen_id = most_common[0][0]
                profile = crm_by_id.get(chosen_id)

        # Exact contact anchors rescue cohorts where no authenticated event was recorded.
        if profile is None:
            anchors: list[pd.Series] = []
            for _, row in rows.iterrows():
                email = normalize_email(row.get("email", ""))
                if email and email in email_profiles:
                    anchors.extend(email_profiles[email])
                phone = normalize_phone(row.get("phone", ""), _country(row))
                if phone and phone in phone_profiles:
                    anchors.extend(phone_profiles[phone])
            profile = unique_profile(anchors)

        # A complete, exact token-set name can anchor only when it identifies one profile.
        if profile is None:
            name_anchors: list[pd.Series] = []
            for _, row in rows.iterrows():
                tokens = _name_tokens(row.get("name", ""))
                if len(tokens) >= 2:
                    name_anchors.extend(name_profiles.get(" ".join(sorted(tokens)), []))
            profile = unique_profile(name_anchors)

        if profile is not None:
            context = {
                "customer_id": _field(profile, "customer_id"),
                "email": _field(profile, "email"),
                "name": _field(profile, "name"),
                "phone": _field(profile, "phone"),
            }
        else:
            context = {
                "customer_id": "",
                "email": _mode(rows["email"].map(_text).tolist(), normalize_email),
                "name": _mode(rows["name"].map(_text).tolist(), lambda x: " ".join(sorted(_name_tokens(x)))),
                "phone": _mode(rows["phone"].map(_text).tolist(), lambda x: normalize_phone(x, _country(rows.iloc[0]))),
            }
        for position in positions:
            for field, value in context.items():
                output.at[position, f"context_{field}"] = value
    return output


def _similarity_pair(record: pd.Series, person: pd.Series, weights: dict[str, float] | None = None) -> tuple[float, dict[str, float], list[str]]:
    """Score a candidate pair from available normalized features and explain evidence."""
    record_id = _field(record, "customer_id")
    inherited_id = False
    if not record_id:
        record_id = _field(record, "context_customer_id")
        inherited_id = bool(record_id)
    crm_id = _field(person, "customer_id")

    record_email = _field(record, "email") or _field(record, "context_email")
    crm_emails = [normalize_email(person.get("email", "")), normalize_email(person.get("secondary_email", ""))]
    crm_emails = [email for email in crm_emails if email]
    email_value = normalize_email(record_email)
    email_sim = max((float(fuzz.ratio(email_value, candidate)) for candidate in crm_emails), default=-1.0) if email_value else -1.0
    email_exact = bool(email_value and email_value in crm_emails)

    record_phone = _field(record, "phone") or _field(record, "context_phone")
    source_phone = normalize_phone(record_phone, _country(record))
    crm_phone = normalize_phone(person.get("phone", ""), _country(person))
    phone_sim = 100.0 if source_phone and source_phone == crm_phone else (0.0 if source_phone and crm_phone else -1.0)
    phone_exact = bool(source_phone and crm_phone and source_phone == crm_phone)

    record_name = _field(record, "name") or _field(record, "context_name")
    crm_name = _field(person, "name")
    if record_name and crm_name:
        name_sim = float(fuzz.token_set_ratio(record_name, crm_name))
    else:
        name_sim = -1.0

    id_sim = 100.0 if record_id and crm_id and record_id.lower() == crm_id.lower() else (0.0 if record_id and crm_id else -1.0)
    id_exact = id_sim == 100.0
    feature_scores = {"id": id_sim, "email": email_sim, "phone": phone_sim, "name": name_sim}
    usable = {field: score for field, score in feature_scores.items() if score >= 0}
    if not usable:
        return 0.0, feature_scores, ["No usable identity fields were available."]
    active_weights = weights or DEFAULT_WEIGHTS
    active_total = sum(max(0.0, float(value)) for value in active_weights.values())
    if active_total <= 0:
        active_weights = DEFAULT_WEIGHTS
        active_total = sum(active_weights.values())
    active_weights = {key: max(0.0, float(value)) / active_total for key, value in active_weights.items()}
    denominator = sum(active_weights.get(field, 0.0) for field in usable)
    score = sum(active_weights.get(field, 0.0) * usable[field] for field in usable) / denominator if denominator else 0.0
    reliable_exact_count = int(id_exact) + int(email_exact) + int(phone_exact)
    id_conflict = bool(record_id and crm_id and record_id.lower() != crm_id.lower())
    contact_conflicts = sum(
        1 for field, value in (("email", email_sim), ("phone", phone_sim), ("name", name_sim))
        if value >= 0 and value < (95 if field in {"email", "name"} else 100)
    )

    # Exact IDs are strong only when the rest of the row does not contradict them badly.
    if id_exact and contact_conflicts == 0:
        score = max(score, 96.0)
    elif id_exact and contact_conflicts == 1:
        score = min(max(score, 85.0), 84.0)
    if email_exact and phone_exact:
        score = max(score, 96.0)
    elif email_exact:
        score = max(score, 90.0)
    elif phone_exact:
        score = max(score, 88.0)
    if id_conflict and (email_exact or phone_exact):
        score = min(score, 84.0)
    if reliable_exact_count == 0:
        # Names alone never auto-link, even when a common surname looks similar.
        score = min(score, 58.0)
    score = max(0.0, min(100.0, float(score)))

    reasons: list[str] = []
    if id_exact:
        reasons.append("Customer ID agrees" + (" (inherited from an authenticated event in the same anonymous-ID cohort)" if inherited_id else ""))
    elif id_conflict:
        reasons.append("Customer IDs disagree; treat the candidate cautiously")
    if email_exact:
        reasons.append("Normalized email matches a CRM primary or secondary email")
    elif email_sim >= 75:
        reasons.append(f"Email is similar to a CRM email ({email_sim:.0f}% string similarity)")
    if phone_exact:
        reasons.append("Phone matches after GCC/E.164 normalization")
    elif phone_sim >= 0:
        reasons.append("Phone values disagree after normalization")
    if name_sim >= 0:
        if name_sim >= 95:
            reasons.append(f"Name tokens agree ({name_sim:.0f}% similarity)")
        elif name_sim >= 60:
            reasons.append(f"Name is a plausible variant ({name_sim:.0f}% similarity)")
    if not reasons:
        reasons.append("Candidate was retrieved by a shared name block; evidence is weak")
    return round(score, 2), feature_scores, reasons


def match_records(
    crm: pd.DataFrame,
    payments: pd.DataFrame,
    events: pd.DataFrame,
    *,
    auto_link_threshold: int = 85,
    review_threshold: int = 60,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Match every payment/event row to its best CRM candidate using blocking.

    The function returns one best candidate per downstream record. Review-band
    candidates are labelled, never auto-merged. The ``weights`` argument is kept
    in feature-weight proportions (all values are normalized before scoring).
    """
    if review_threshold < 0 or auto_link_threshold > 100 or review_threshold >= auto_link_threshold:
        raise ValueError("Thresholds must satisfy 0 <= review threshold < auto-link threshold <= 100.")
    for frame_name, frame in (("crm", crm), ("payments", payments), ("events", events)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{frame_name} input must be a pandas DataFrame.")
    if crm.empty or (payments.empty and events.empty):
        return pd.DataFrame(columns=[
            "source", "record_id", "matched_crm_record_id", "matched_customer_id",
            "confidence", "band", "reasons", "candidate_count", "matched_by",
        ])

    crm_data = crm.reset_index(drop=True).copy()
    event_data = add_event_cohort_context(events, crm_data)
    active_weights = weights or DEFAULT_WEIGHTS
    # Attach currency-derived country to payment rows if schema mapping did not include it.
    payment_data = payments.reset_index(drop=True).copy()
    if "country" not in payment_data.columns:
        payment_data["country"] = ""
    payment_data["country"] = payment_data.apply(_country, axis=1)

    index = _crm_index_fields(crm_data)
    results: list[dict[str, Any]] = []
    for source, frame in (("payments", payment_data), ("events", event_data)):
        for _, record in frame.iterrows():
            keys = _row_block_keys(record)
            candidate_positions: set[int] = set()
            for key in keys:
                candidate_positions.update(index.get(key, []))
            # If the event cohort has a known customer ID, include that CRM candidate as a strong, traceable block.
            context_id = _field(record, "context_customer_id")
            if context_id:
                candidate_positions.update(index.get(f"id:{context_id.lower()}", []))
            scored: list[tuple[float, int, dict[str, float], list[str]]] = []
            for position in candidate_positions:
                score, features, reasons = _similarity_pair(record, crm_data.iloc[position], active_weights)
                scored.append((score, position, features, reasons))
            scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            if scored:
                confidence, best_position, features, reasons = scored[0]
                matched_row = crm_data.iloc[best_position]
                matched_crm_record_id = _field(matched_row, "record_id")
                matched_customer_id = _field(matched_row, "customer_id")
                if confidence >= auto_link_threshold:
                    band = "auto-link"
                elif confidence >= review_threshold:
                    band = "needs review"
                else:
                    band = "no match"
                matched_by = ", ".join(
                    field for field, value in features.items()
                    if value >= (95 if field in {"email", "name"} else 100)
                )
                reason_text = "; ".join(reasons)
            else:
                confidence, best_position, features = 0.0, -1, {}
                matched_crm_record_id = ""
                matched_customer_id = ""
                band = "no match"
                matched_by = ""
                reason_text = "No CRM candidate shared a blocking key."
            results.append({
                "source": source,
                "record_id": _field(record, "record_id"),
                "matched_crm_record_id": matched_crm_record_id,
                "matched_customer_id": matched_customer_id,
                "confidence": round(float(confidence), 2),
                "band": band,
                "reasons": reason_text,
                "candidate_count": len(candidate_positions),
                "matched_by": matched_by,
            })
    return pd.DataFrame(results)
