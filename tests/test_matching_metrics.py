"""Evaluation against synthetic ground-truth labels; prints real measured metrics."""
from __future__ import annotations

import pandas as pd


def test_matching_precision_recall_against_ground_truth(raw_sample, canonical_sample, sample_analysis):
    ground_truth = raw_sample["ground_truth"]
    identity = ground_truth[ground_truth["truth_type"] == "identity"]
    crm_truth = identity[identity["source"] == "crm"].set_index("record_id")["true_customer_id"].to_dict()
    crm_profiles = canonical_sample["crm"].set_index("record_id")
    matches = sample_analysis["matches"].copy()
    id_to_true = crm_truth
    truth_by_source_record = {
        (row.source, row.record_id): row.true_customer_id
        for row in identity.itertuples(index=False)
    }
    eligible_keys: set[tuple[str, str]] = set()
    for source in ("payments", "events"):
        source_frame = canonical_sample[source]
        for row in source_frame.itertuples(index=False):
            # Rows without any identity key at all are not linkable and are excluded from recall.
            fields = [getattr(row, name, "") for name in ("customer_id", "email", "name", "phone", "anonymous_id")]
            if any(str(value).strip() for value in fields) and (source, row.record_id) in truth_by_source_record:
                eligible_keys.add((source, row.record_id))
    matched = matches[matches["confidence"] >= 60]
    predicted_keys = set(zip(matched["source"], matched["record_id"]))
    true_positive = 0
    false_positive = 0
    for row in matched.itertuples(index=False):
        key = (row.source, row.record_id)
        predicted_true_id = id_to_true.get(row.matched_crm_record_id, "")
        actual_true_id = truth_by_source_record.get(key, "")
        if actual_true_id and predicted_true_id == actual_true_id:
            true_positive += 1
        else:
            false_positive += 1
    matched_crm_for_key = {
        (row.source, row.record_id): row.matched_crm_record_id
        for row in matches.itertuples(index=False)
    }
    correct_prediction_keys = {
        key for key in predicted_keys & eligible_keys
        if id_to_true.get(matched_crm_for_key.get(key, ""), "") == truth_by_source_record.get(key, "")
    }
    false_negative = len(eligible_keys) - len(correct_prediction_keys)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(eligible_keys) if eligible_keys else 0.0
    print("\nMEASURED IDENTITY MATCHING METRICS")
    print(f"Eligible records: {len(eligible_keys):,}")
    print(f"True positives: {true_positive:,}; false positives: {false_positive:,}; false negatives: {false_negative:,}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall: {recall:.4%}")
    assert precision >= 0.999, f"Matching precision below target: {precision:.3%}"
    assert recall >= 0.98, f"Matching recall below target: {recall:.3%}"


def test_match_results_do_not_mark_review_as_auto_merge(sample_analysis):
    matches = sample_analysis["matches"]
    review = matches[matches["band"] == "needs review"]
    assert (review["confidence"] >= 60).all()
    assert (review["confidence"] < 85).all()
    assert not (review["band"] == "auto-link").any()
