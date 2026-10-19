"""Generate deterministic synthetic data for the Customer Identity & Data Readiness Checker.

Run from any directory:
    python data/generate_data.py

By default, CSVs are written to data/sample/ beside this script. Use --output-dir
and --seed to customize the output location and random seed.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SEED = 20261019
DEFAULT_AS_OF = date(2026, 10, 19)
CUSTOMER_COUNT = 1_500
PAYMENT_COUNT = 6_000
EVENT_COUNT = 12_000

FIRST_NAMES = [
    "Ahmed", "Aisha", "Omar", "Fatima", "Khalid", "Mariam", "Yusuf", "Noor",
    "Hassan", "Sara", "Mohammed", "Layla", "Abdullah", "Amira", "Rashid", "Huda",
    "Ibrahim", "Zainab", "Tariq", "Noura", "Faisal", "Reem", "Saeed", "Dana",
    "Salman", "Maha", "Hamdan", "Lina", "Bilal", "Yasmin", "Majid", "Salma",
    "Rayan", "Muna", "Adnan", "Rania", "Farah", "Imran", "Kareem", "Dina",
    "Sultan", "Amina", "Jamal", "Hind", "Mustafa", "Shamma", "Nasser", "Joud",
    "Zayed", "Malik", "Samira", "Nabil", "Maryam", "Ismail", "Ruqayya", "Fadi",
    "Laila", "Anas", "Mariam", "Rami",
]

FAMILY_NAMES = [
    "Khan", "Al Mansoori", "Al Nuaimi", "Al Mazrouei", "Al Falasi", "Rahman",
    "Haddad", "Al Qasimi", "Siddiqui", "Farooq", "Al Shamsi", "Malik", "Hussein",
    "Al Ketbi", "Abbas", "Darwish", "Al Suwaidi", "Chaudhry", "Saleh", "Al Kaabi",
    "Qureshi", "Jaber", "Al Marri", "Ansari", "Hashmi", "Al Hamadi", "Rashid",
    "Nasser", "Al Dhaheri", "Mansour", "Sheikh", "Al Ali", "Bakri", "Younis",
    "Al Tayer", "Khalifa", "Bashir", "Zaman", "Al Ameri", "Khalil", "Mehta",
    "Al Balushi", "Rahimi", "Kassem", "Al Hajri", "Shah", "Al Jaberi", "Faruqi",
    "Saeed", "Al Muhairi",
]

COUNTRY_INFO: dict[str, dict[str, Any]] = {
    "United Arab Emirates": {
        "dial_code": "971", "prefixes": ["50", "52", "54", "55", "56", "58"],
        "local_zero": True, "currency": "AED",
    },
    "Saudi Arabia": {
        "dial_code": "966", "prefixes": ["50", "53", "54", "55", "56", "59"],
        "local_zero": True, "currency": "SAR",
    },
    "Qatar": {
        "dial_code": "974", "prefixes": ["33", "55", "66", "77"],
        "local_zero": False, "currency": "QAR",
    },
    "Bahrain": {
        "dial_code": "973", "prefixes": ["32", "33", "34", "35", "36", "39"],
        "local_zero": False, "currency": "BHD",
    },
    "Kuwait": {
        "dial_code": "965", "prefixes": ["50", "55", "60", "66"],
        "local_zero": False, "currency": "KWD",
    },
    "Oman": {
        "dial_code": "968", "prefixes": ["91", "92", "93", "94", "95", "96", "97", "98", "99"],
        "local_zero": False, "currency": "OMR",
    },
}

VALID_EVENT_TYPES = [
    "page_view", "product_view", "search", "add_to_cart", "checkout_started",
    "purchase", "login", "signup", "support_opened",
]

SEVERITY_BY_ISSUE = {
    "email_typo": "medium",
    "email_casing": "low",
    "second_email": "medium",
    "name_variant": "low",
    "phone_format_variant": "low",
    "duplicate_crm_customer": "high",
    "unknown_customer_id": "high",
    "missing_id": "high",
    "mixed_date_format": "medium",
    "invalid_date": "high",
    "status_purchase_conflict": "high",
    "purchase_before_signup": "high",
    "zero_amount": "high",
    "negative_amount": "high",
    "unknown_event_type": "medium",
}


@dataclass(frozen=True)
class Customer:
    """Canonical synthetic person profile used to create linked source records."""

    customer_id: str
    full_name: str
    email: str
    secondary_email: str
    phone: str
    phone_prefix: str
    phone_subscriber: str
    country: str
    signup_day: date
    status: str
    created_at: str


def iso_day(value: date) -> str:
    """Format a date in unambiguous ISO-8601 form."""
    return value.isoformat()


def random_day(rng: random.Random, start: date, end: date) -> date:
    """Return a random calendar day within an inclusive range."""
    if end < start:
        raise ValueError("end must be on or after start")
    return start + timedelta(days=rng.randint(0, (end - start).days))


def make_name(rng: random.Random) -> str:
    """Build a plausible GCC-style synthetic name."""
    first = rng.choice(FIRST_NAMES)
    family = rng.choice(FAMILY_NAMES)
    if rng.random() < 0.12:
        middle = rng.choice(["Ali", "Mohammed", "Hamad", "Abdul", "Saeed"])
        return f"{first} {middle} {family}"
    return f"{first} {family}"


def email_slug(name: str) -> str:
    """Create a simple email-local-part slug without external libraries."""
    keep = [char.lower() if char.isalnum() else "." for char in name]
    slug = "".join(keep)
    while ".." in slug:
        slug = slug.replace("..", ".")
    return slug.strip(".")


def make_email(name: str, sequence: int) -> str:
    """Use a reserved example domain so generated addresses are not real contacts."""
    return f"{email_slug(name)}.{sequence:04d}@example.com"


def make_secondary_email(rng: random.Random, customer_id: str) -> str:
    """Create a synthetic alternate email on a reserved example domain."""
    suffix = customer_id.split("-")[-1]
    provider = rng.choice(["mail.example.net", "inbox.example.org", "example.net"])
    return f"alt.{suffix}@{provider}"


def phone_for_customer(
    rng: random.Random, country: str
) -> tuple[str, str, str]:
    """Return canonical E.164 phone, its two-digit prefix, and seven-digit tail."""
    info = COUNTRY_INFO[country]
    prefix = rng.choice(info["prefixes"])
    subscriber = f"{rng.randint(0, 9_999_999):07d}"
    canonical = f"+{info['dial_code']}{prefix}{subscriber}"
    return canonical, prefix, subscriber


def format_phone_variant(customer: Customer, rng: random.Random) -> str:
    """Return a display-format variation of a customer's phone number."""
    info = COUNTRY_INFO[customer.country]
    tail = customer.phone_subscriber
    prefix = customer.phone_prefix
    style = rng.choice(["local", "international_spaces", "dashes", "compact_local"])
    if style == "local":
        national = f"0{prefix}" if info["local_zero"] else prefix
        return f"{national} {tail[:3]} {tail[3:]}"
    if style == "international_spaces":
        return f"+{info['dial_code']} {prefix} {tail[:3]} {tail[3:]}"
    if style == "dashes":
        return f"+{info['dial_code']}-{prefix}-{tail[:3]}-{tail[3:]}"
    national = f"0{prefix}" if info["local_zero"] else prefix
    return f"{national}{tail}"


def name_variant(name: str, rng: random.Random) -> str:
    """Return a common abbreviated or reordered name variant."""
    parts = name.split()
    if len(parts) < 2:
        return name
    first = parts[0]
    family = " ".join(parts[1:])
    options = [
        f"{first[0]}. {family}",
        f"{family}, {first}",
        family,
        f"{first} {family[0]}.",
        f"{first} {family}",
    ]
    options = [option for option in options if option != name]
    return rng.choice(options)


def email_typo(email: str, rng: random.Random) -> str:
    """Inject one easy-to-understand typo into a synthetic email address."""
    local, domain = email.split("@", maxsplit=1)
    choices = [
        f"{local}x@{domain}",
        f"{local[:-1] or local}q@{domain}",
        f"{local}@{domain.replace('example', 'examp1e', 1)}",
        f"{local}@{domain.replace('.com', '.con')}",
    ]
    return rng.choice(choices)


def mixed_day_format(value: date, rng: random.Random) -> str:
    """Render a valid date in a deliberately mixed but recognizable format."""
    if rng.random() < 0.5:
        return value.strftime("%d/%m/%Y")
    return value.strftime("%b %d, %Y")


def mixed_datetime_format(value: datetime, rng: random.Random) -> str:
    """Render a valid timestamp in a deliberately mixed but recognizable format."""
    if rng.random() < 0.5:
        return value.strftime("%d/%m/%Y %H:%M")
    return value.strftime("%b %d, %Y %I:%M %p")


def invalid_date_token(rng: random.Random, include_time: bool = False) -> str:
    """Return a deliberately unparsable date/timestamp token."""
    candidates = ["not-a-date", "2026-19-42", "31/02/2026", "unknown"]
    if include_time:
        candidates += ["2026-10-40 25:71:99", "yesterday-ish"]
    return rng.choice(candidates)


def allocate_groups(
    rng: random.Random,
    total: int,
    group_sizes: dict[str, int],
    reserved: dict[str, list[int]] | None = None,
) -> dict[str, list[int]]:
    """Allocate disjoint row indices to issue groups for measurable tests."""
    groups = {name: list(values) for name, values in (reserved or {}).items()}
    used = {index for values in groups.values() for index in values}
    remaining = [index for index in range(total) if index not in used]
    rng.shuffle(remaining)
    cursor = 0
    for name, size in group_sizes.items():
        if size < 0 or cursor + size > len(remaining):
            raise ValueError(f"Cannot allocate {size} rows to {name}")
        groups[name] = remaining[cursor : cursor + size]
        cursor += size
    return groups


def make_customers(rng: random.Random, as_of: date) -> list[Customer]:
    """Create the canonical customer population and fixed status-conflict cohort."""
    customers: list[Customer] = []
    for number in range(1, CUSTOMER_COUNT + 1):
        customer_id = f"CUST-{number:06d}"
        full_name = make_name(rng)
        country = rng.choice(list(COUNTRY_INFO))
        phone, phone_prefix, phone_subscriber = phone_for_customer(rng, country)
        email = make_email(full_name, number)
        # A subset of customers intentionally has an alternate email on file.
        secondary_email = make_secondary_email(rng, customer_id) if rng.random() < 0.38 else ""
        signup_start = as_of - timedelta(days=1_200)
        signup_end = as_of - timedelta(days=25)
        signup_day = random_day(rng, signup_start, signup_end)
        if number <= 40:
            # Ensure the special conflict cohort has enough history before its recent purchase.
            signup_day = random_day(rng, as_of - timedelta(days=900), as_of - timedelta(days=180))
            status = "churned"
        else:
            status = rng.choices(["active", "churned", "dormant"], weights=[62, 18, 20], k=1)[0]
        created_at = datetime.combine(signup_day, time(hour=rng.randint(8, 19), minute=rng.randint(0, 59)))
        customers.append(
            Customer(
                customer_id=customer_id,
                full_name=full_name,
                email=email,
                secondary_email=secondary_email,
                phone=phone,
                phone_prefix=phone_prefix,
                phone_subscriber=phone_subscriber,
                country=country,
                signup_day=signup_day,
                status=status,
                created_at=created_at.isoformat(sep=" ", timespec="seconds"),
            )
        )
    return customers


def create_issue(
    issues: list[dict[str, str]],
    *,
    source: str,
    record_id: str,
    customer_id: str,
    issue_type: str,
    field: str,
    expected_value: Any,
    observed_value: Any,
    details: str,
) -> None:
    """Record a deliberately planted issue without changing it back."""
    issues.append(
        {
            "truth_type": "error",
            "source": source,
            "record_id": str(record_id),
            "true_customer_id": customer_id,
            "issue_type": issue_type,
            "field": field,
            "expected_value": "" if expected_value is None else str(expected_value),
            "observed_value": "" if observed_value is None else str(observed_value),
            "severity": SEVERITY_BY_ISSUE[issue_type],
            "details": details,
        }
    )


def generate_crm(
    rng: random.Random,
    customers: list[Customer],
    as_of: date,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Create CRM records, including duplicate profiles and data-quality defects."""
    groups = allocate_groups(
        rng,
        len(customers),
        {"missing_id": 12, "mixed_date_format": 90, "invalid_date": 15},
    )
    rows: list[dict[str, Any]] = []
    true_ids: list[str] = []
    for idx, customer in enumerate(customers):
        record_id = f"CRM-{idx + 1:06d}"
        customer_id: str | None = customer.customer_id
        if idx in groups["missing_id"]:
            customer_id = None
        signup_value = iso_day(customer.signup_day)
        if idx in groups["mixed_date_format"]:
            observed = mixed_day_format(customer.signup_day, rng)
            create_issue(
                issues, source="crm", record_id=record_id, customer_id=customer.customer_id,
                issue_type="mixed_date_format", field="signup_date",
                expected_value=signup_value, observed_value=observed,
                details="The date is valid but uses a different format from the rest of the column.",
            )
            signup_value = observed
        elif idx in groups["invalid_date"]:
            observed = invalid_date_token(rng)
            create_issue(
                issues, source="crm", record_id=record_id, customer_id=customer.customer_id,
                issue_type="invalid_date", field="signup_date",
                expected_value=signup_value, observed_value=observed,
                details="The signup date cannot be safely parsed as a real calendar date.",
            )
            signup_value = observed
        if idx in groups["missing_id"]:
            create_issue(
                issues, source="crm", record_id=record_id, customer_id=customer.customer_id,
                issue_type="missing_id", field="customer_id",
                expected_value=customer.customer_id, observed_value="",
                details="This CRM row has no customer ID, weakening joins to other systems.",
            )
        rows.append(
            {
                "crm_record_id": record_id,
                "customer_id": customer_id or "",
                "full_name": customer.full_name,
                "email": customer.email,
                "secondary_email": customer.secondary_email,
                "phone": customer.phone,
                "signup_date": signup_value,
                "status": customer.status,
                "country": customer.country,
                "created_at": customer.created_at,
            }
        )
        true_ids.append(customer.customer_id)

    # Add duplicate CRM rows from otherwise clean source rows so each planted defect is labelled once.
    excluded_sources = set(groups["missing_id"] + groups["mixed_date_format"] + groups["invalid_date"])
    eligible_sources = [idx for idx in range(CUSTOMER_COUNT) if idx not in excluded_sources]
    duplicate_source_numbers = rng.sample(eligible_sources, 30)
    for duplicate_number, source_idx in enumerate(duplicate_source_numbers, start=1):
        original = rows[source_idx]
        original_customer = customers[source_idx]
        record_id = f"CRM-DUP-{duplicate_number:04d}"
        duplicate_customer_id = f"CUST-DUP-{duplicate_number:04d}"
        duplicate_row = dict(original)
        duplicate_row["crm_record_id"] = record_id
        duplicate_row["customer_id"] = duplicate_customer_id
        rows.append(duplicate_row)
        true_ids.append(original_customer.customer_id)
        create_issue(
            issues, source="crm", record_id=record_id, customer_id=original_customer.customer_id,
            issue_type="duplicate_crm_customer", field="customer_id",
            expected_value=original_customer.customer_id, observed_value=duplicate_customer_id,
            details="A second CRM row represents the same person under a different customer ID.",
        )
    return pd.DataFrame(rows), true_ids


def generate_payments(
    rng: random.Random,
    customers: list[Customer],
    as_of: date,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Create transactions with explainable, separately counted planted defects."""
    reserved = {"status_purchase_conflict": list(range(40))}
    groups = allocate_groups(
        rng,
        PAYMENT_COUNT,
        {
            "email_typo": 60,
            "email_casing": 90,
            "second_email": 100,
            "name_variant": 260,
            "phone_format_variant": 300,
            "unknown_customer_id": 30,
            "missing_id": 30,
            "mixed_date_format": 150,
            "invalid_date": 45,
            "purchase_before_signup": 70,
            "zero_amount": 25,
            "negative_amount": 25,
        },
        reserved=reserved,
    )
    customer_by_id = {customer.customer_id: customer for customer in customers}
    secondary_candidates = [customer for customer in customers if customer.secondary_email]
    assigned_customers: list[Customer] = [rng.choice(customers) for _ in range(PAYMENT_COUNT)]

    # Each of these 40 transactions belongs to a churned customer and is deliberately recent.
    for index in groups["status_purchase_conflict"]:
        assigned_customers[index] = customer_by_id[f"CUST-{index + 1:06d}"]
    # Ensure second-email cases use customers whose alternate email is also present in CRM.
    for index in groups["second_email"]:
        assigned_customers[index] = rng.choice(secondary_candidates)

    rows: list[dict[str, Any]] = []
    true_ids: list[str] = []
    for index, customer in enumerate(assigned_customers):
        transaction_id = f"PAY-{index + 1:07d}"
        start_day = min(customer.signup_day + timedelta(days=1), as_of)
        transaction_day = random_day(rng, start_day, as_of)
        if index in groups["status_purchase_conflict"]:
            transaction_day = as_of - timedelta(days=rng.randint(1, 12))
        amount = round(rng.uniform(8.0, 2_400.0), 2)
        payment_status = rng.choices(["completed", "pending", "failed"], weights=[86, 8, 6], k=1)[0]
        if index in groups["status_purchase_conflict"]:
            payment_status = "completed"
        transaction_date = iso_day(transaction_day)

        customer_id: str = customer.customer_id
        customer_email = customer.email
        customer_name = customer.full_name
        customer_phone = customer.phone

        # The issue groups are disjoint by construction, making measured detection rates clear.
        if index in groups["email_typo"]:
            customer_email = email_typo(customer.email, rng)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="email_typo", field="customer_email", expected_value=customer.email,
                observed_value=customer_email,
                details="The email contains a planted typo, which can prevent exact identity joins.",
            )
        elif index in groups["email_casing"]:
            customer_email = customer.email.upper()
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="email_casing", field="customer_email", expected_value=customer.email,
                observed_value=customer_email,
                details="The email uses different letter casing; this should be normalized, not treated as a new person.",
            )
        elif index in groups["second_email"]:
            customer_email = customer.secondary_email
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="second_email", field="customer_email", expected_value=customer.email,
                observed_value=customer_email,
                details="The payment uses the customer's alternate email instead of the primary CRM email.",
            )

        if index in groups["name_variant"]:
            customer_name = name_variant(customer.full_name, rng)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="name_variant", field="customer_name", expected_value=customer.full_name,
                observed_value=customer_name,
                details="The name is abbreviated or reordered, so exact text matching may fail.",
            )
        if index in groups["phone_format_variant"]:
            customer_phone = format_phone_variant(customer, rng)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="phone_format_variant", field="customer_phone", expected_value=customer.phone,
                observed_value=customer_phone,
                details="The same phone number is written in a different local or international format.",
            )
        if index in groups["unknown_customer_id"]:
            customer_id = f"CUST-UNKNOWN-P-{index + 1:05d}"
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="unknown_customer_id", field="customer_id", expected_value=customer.customer_id,
                observed_value=customer_id,
                details="The transaction references a customer ID that is not present in the CRM dataset.",
            )
        elif index in groups["missing_id"]:
            customer_id = ""
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="missing_id", field="customer_id", expected_value=customer.customer_id,
                observed_value="",
                details="The transaction has no customer ID, reducing join coverage.",
            )

        if index in groups["mixed_date_format"]:
            transaction_date = mixed_day_format(transaction_day, rng)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="mixed_date_format", field="transaction_date",
                expected_value=iso_day(transaction_day), observed_value=transaction_date,
                details="The transaction date is valid but formatted differently from the rest of the column.",
            )
        elif index in groups["invalid_date"]:
            transaction_date = invalid_date_token(rng)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="invalid_date", field="transaction_date",
                expected_value=iso_day(transaction_day), observed_value=transaction_date,
                details="The transaction date cannot be safely parsed as a real calendar date.",
            )
        elif index in groups["purchase_before_signup"]:
            transaction_day = customer.signup_day - timedelta(days=rng.randint(1, 180))
            transaction_date = iso_day(transaction_day)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="purchase_before_signup", field="transaction_date",
                expected_value=f"on or after signup date {iso_day(customer.signup_day)}",
                observed_value=transaction_date,
                details="A purchase is dated before the customer's CRM signup date, which may indicate bad dates or a wrong identity link.",
            )

        if index in groups["zero_amount"]:
            amount = 0.0
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="zero_amount", field="amount", expected_value="greater than 0",
                observed_value=amount,
                details="A zero-value transaction can distort revenue, average order value, and conversion metrics.",
            )
        elif index in groups["negative_amount"]:
            amount = -round(rng.uniform(1.0, 500.0), 2)
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="negative_amount", field="amount", expected_value="greater than 0 for a purchase",
                observed_value=amount,
                details="A negative purchase amount can distort revenue unless it is a clearly modelled refund or adjustment.",
            )

        if index in groups["status_purchase_conflict"]:
            create_issue(
                issues, source="payments", record_id=transaction_id, customer_id=customer.customer_id,
                issue_type="status_purchase_conflict", field="status / transaction_date",
                expected_value="A churned customer has no recent completed purchase, or the status is reviewed",
                observed_value=f"status=churned; completed transaction={transaction_date}",
                details="CRM says the customer is churned but payments show a recent completed purchase; retention and churn reports may be wrong.",
            )

        rows.append(
            {
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "customer_email": customer_email,
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "transaction_date": transaction_date,
                "amount": amount,
                "currency": COUNTRY_INFO[customer.country]["currency"],
                "payment_status": payment_status,
            }
        )
        true_ids.append(customer.customer_id)

    return pd.DataFrame(rows), true_ids


def generate_events(
    rng: random.Random,
    customers: list[Customer],
    as_of: date,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Create product analytics events with anonymous and sometimes authenticated IDs."""
    groups = allocate_groups(
        rng,
        EVENT_COUNT,
        {
            "unknown_customer_id": 40,
            "missing_id": 50,
            "mixed_date_format": 180,
            "invalid_date": 70,
            "unknown_event_type": 120,
            "email_typo": 25,
            "email_casing": 55,
            "name_variant": 130,
            "phone_format_variant": 100,
        },
    )
    rows: list[dict[str, Any]] = []
    true_ids: list[str] = []
    for index in range(EVENT_COUNT):
        customer = rng.choice(customers)
        event_id = f"EVT-{index + 1:08d}"
        anonymous_id = f"ANON-{int(customer.customer_id.split('-')[-1]):06d}"
        customer_id = customer.customer_id if rng.random() < 0.24 else ""
        user_email = customer.email if rng.random() < 0.13 else ""
        user_name = customer.full_name if rng.random() < 0.16 else ""
        user_phone = customer.phone if rng.random() < 0.06 else ""
        event_type = rng.choice(VALID_EVENT_TYPES)

        start_day = min(customer.signup_day, as_of)
        event_day = random_day(rng, start_day, as_of)
        event_time = datetime.combine(
            event_day,
            time(hour=rng.randint(0, 23), minute=rng.randint(0, 59), second=rng.randint(0, 59)),
        )
        event_timestamp = event_time.isoformat(sep=" ", timespec="seconds")

        if index in groups["unknown_customer_id"]:
            customer_id = f"CUST-UNKNOWN-E-{index + 1:05d}"
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="unknown_customer_id", field="customer_id",
                expected_value=customer.customer_id, observed_value=customer_id,
                details="The event references a customer ID that is not present in the CRM dataset.",
            )
        if index in groups["missing_id"]:
            # Remove every useful identity attribute to simulate an unjoinable anonymous event.
            anonymous_id = ""
            customer_id = ""
            user_email = ""
            user_name = ""
            user_phone = ""
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="missing_id", field="anonymous_id / customer_id",
                expected_value=customer.customer_id, observed_value="",
                details="The event has no anonymous ID or authenticated identity fields and cannot be stitched reliably.",
            )

        if index in groups["email_typo"]:
            user_email = email_typo(customer.email, rng)
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="email_typo", field="user_email", expected_value=customer.email,
                observed_value=user_email,
                details="The event email contains a planted typo and may not join to CRM or payments.",
            )
        elif index in groups["email_casing"]:
            user_email = customer.email.upper()
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="email_casing", field="user_email", expected_value=customer.email,
                observed_value=user_email,
                details="The event email uses different letter casing; case normalization should preserve the identity match.",
            )

        if index in groups["name_variant"]:
            user_name = name_variant(customer.full_name, rng)
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="name_variant", field="user_name", expected_value=customer.full_name,
                observed_value=user_name,
                details="The event user name is abbreviated or reordered compared with the CRM name.",
            )
        if index in groups["phone_format_variant"]:
            user_phone = format_phone_variant(customer, rng)
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="phone_format_variant", field="user_phone", expected_value=customer.phone,
                observed_value=user_phone,
                details="The event phone uses a different display format from the CRM phone.",
            )

        if index in groups["mixed_date_format"]:
            event_timestamp = mixed_datetime_format(event_time, rng)
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="mixed_date_format", field="event_timestamp",
                expected_value=event_time.isoformat(sep=" ", timespec="seconds"),
                observed_value=event_timestamp,
                details="The event timestamp is valid but differs from the column's normal timestamp format.",
            )
        elif index in groups["invalid_date"]:
            event_timestamp = invalid_date_token(rng, include_time=True)
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="invalid_date", field="event_timestamp",
                expected_value=event_time.isoformat(sep=" ", timespec="seconds"),
                observed_value=event_timestamp,
                details="The event timestamp cannot be safely parsed as a valid date/time.",
            )

        if index in groups["unknown_event_type"]:
            event_type = rng.choice(["legacy_event_v99", "super_like", "unknown_action", "checkout_finish_v3"])
            create_issue(
                issues, source="events", record_id=event_id, customer_id=customer.customer_id,
                issue_type="unknown_event_type", field="event_type",
                expected_value="one of the documented event types", observed_value=event_type,
                details="The event type is outside the known event vocabulary, so analytics grouping may silently omit it.",
            )

        rows.append(
            {
                "event_id": event_id,
                "anonymous_id": anonymous_id,
                "customer_id": customer_id,
                "user_email": user_email,
                "user_name": user_name,
                "user_phone": user_phone,
                "event_type": event_type,
                "event_timestamp": event_timestamp,
                "device_type": rng.choice(["mobile", "desktop", "tablet"]),
                "country": customer.country,
            }
        )
        true_ids.append(customer.customer_id)
    return pd.DataFrame(rows), true_ids


def create_ground_truth(
    crm: pd.DataFrame,
    crm_true_ids: list[str],
    payments: pd.DataFrame,
    payment_true_ids: list[str],
    events: pd.DataFrame,
    event_true_ids: list[str],
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    """Combine record-to-person mappings and injected-issue labels for evaluation."""
    truth_rows: list[dict[str, str]] = []

    def add_identity_rows(
        source: str,
        frame: pd.DataFrame,
        primary_key: str,
        true_ids: list[str],
        observed_id_col: str,
    ) -> None:
        for (_, row), true_customer_id in zip(frame.iterrows(), true_ids):
            truth_rows.append(
                {
                    "truth_type": "identity",
                    "source": source,
                    "record_id": str(row[primary_key]),
                    "true_customer_id": true_customer_id,
                    "issue_type": "",
                    "field": observed_id_col,
                    "expected_value": true_customer_id,
                    "observed_value": str(row[observed_id_col]) if row[observed_id_col] is not None else "",
                    "severity": "",
                    "details": "Ground-truth identity mapping for synthetic-data evaluation only.",
                }
            )

    add_identity_rows("crm", crm, "crm_record_id", crm_true_ids, "customer_id")
    add_identity_rows("payments", payments, "transaction_id", payment_true_ids, "customer_id")
    add_identity_rows("events", events, "event_id", event_true_ids, "anonymous_id")
    truth_rows.extend(issues)
    ground_truth = pd.DataFrame(
        truth_rows,
        columns=[
            "truth_type", "source", "record_id", "true_customer_id", "issue_type", "field",
            "expected_value", "observed_value", "severity", "details",
        ],
    )
    ground_truth.insert(0, "truth_id", [f"GT-{index:07d}" for index in range(1, len(ground_truth) + 1)])
    return ground_truth


def print_summary(
    output_dir: Path,
    crm: pd.DataFrame,
    payments: pd.DataFrame,
    events: pd.DataFrame,
    ground_truth: pd.DataFrame,
    seed: int,
    as_of: date,
) -> None:
    """Print deterministic row and injected-issue counts for the user."""
    issue_rows = ground_truth.loc[ground_truth["truth_type"] == "error"]
    counts = Counter(issue_rows["issue_type"].astype(str))
    identity_count = int((ground_truth["truth_type"] == "identity").sum())
    print("Customer Identity & Data Readiness Checker — Stage 1 data generation")
    print(f"Seed: {seed} | As-of date: {as_of.isoformat()}")
    print(f"Output directory: {output_dir.resolve()}")
    print("\nGenerated CSV row counts:")
    print(f"  crm.csv          {len(crm):>6,} rows (1,500 canonical customers + {len(crm) - CUSTOMER_COUNT} duplicate CRM rows)")
    print(f"  payments.csv     {len(payments):>6,} rows")
    print(f"  events.csv       {len(events):>6,} rows")
    print(f"  ground_truth.csv {len(ground_truth):>6,} rows ({identity_count:,} identity mappings + {len(issue_rows):,} planted issues)")
    print("\nPlanted issue counts (one row per known issue):")
    for issue_type in sorted(counts):
        print(f"  {issue_type:<29} {counts[issue_type]:>4,}")
    print(f"  {'TOTAL':<29} {sum(counts.values()):>4,}")


def generate_dataset(output_dir: Path, seed: int = DEFAULT_SEED, as_of: date = DEFAULT_AS_OF) -> dict[str, pd.DataFrame]:
    """Generate all four CSVs and return their DataFrames for validation or reuse."""
    rng = random.Random(seed)
    issues: list[dict[str, str]] = []
    customers = make_customers(rng, as_of)
    crm, crm_true_ids = generate_crm(rng, customers, as_of, issues)
    payments, payment_true_ids = generate_payments(rng, customers, as_of, issues)
    events, event_true_ids = generate_events(rng, customers, as_of, issues)
    ground_truth = create_ground_truth(
        crm, crm_true_ids, payments, payment_true_ids, events, event_true_ids, issues
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "crm.csv": crm,
        "payments.csv": payments,
        "events.csv": events,
        "ground_truth.csv": ground_truth,
    }
    for filename, frame in datasets.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8")
    print_summary(output_dir, crm, payments, events, ground_truth, seed, as_of)
    return datasets


def parse_args() -> argparse.Namespace:
    """Read optional output and reproducibility settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sample",
        help="Directory for generated CSV files (default: data/sample beside this script).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed; same seed produces same data.")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=DEFAULT_AS_OF,
        help="Fixed reference date in YYYY-MM-DD format (default: 2026-10-19).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the generator from the command line."""
    args = parse_args()
    generate_dataset(args.output_dir, seed=args.seed, as_of=args.as_of)


if __name__ == "__main__":
    main()
