"""Tests for the rule, the learned models, the statistics, and the confound control."""

from __future__ import annotations

import numpy as np
import pytest

from casting_qa.modeling.classify import (
    apply_or_gate,
    apply_rule,
    balanced_random_subset,
    build_models,
    decision_scores,
    evaluate,
    exposure_matched_pairs,
    fit_or_gate,
    fit_rule,
    headline_intervals,
    matched_control,
    mcnemar_exact,
    proportion_interval,
    report,
    rule_score,
    separation_table,
    time_inspection,
    tune,
)
from tests.conftest import make_feature_frame


def test_rule_scaler_is_fitted_on_the_training_partition_only():
    """
    A casting must score the same whichever batch it arrives in.

    Shifting a second batch must move its scores, which it would not if the
    scaler were refitted on whatever it was handed.
    """
    train = make_feature_frame(40, seed=0)
    rule = fit_rule(train, train.label.to_numpy())
    other = make_feature_frame(40, seed=1)
    shifted = other.copy()
    shifted[rule["columns"]] += 10.0
    assert np.allclose(rule_score(shifted, rule) - rule_score(other, rule),
                       np.mean(10.0 / rule["scaler"].scale_))


def test_rule_threshold_beats_predicting_one_class():
    frame = make_feature_frame(80)
    rule = fit_rule(frame, frame.label.to_numpy())
    predicted = apply_rule(frame, rule)
    assert 0 < predicted.mean() < 1
    assert rule["f1"] > 0.6


def test_or_gate_fires_when_any_feature_exceeds_its_sound_percentile():
    frame = make_feature_frame(80)
    gates = fit_or_gate(frame, frame.label.to_numpy(), columns=("tophat_max",), percentile=50)
    predicted = apply_or_gate(frame, gates)
    sound = frame[frame.label == 0]
    assert predicted[frame.label.to_numpy() == 0].mean() == pytest.approx(
        (sound["tophat_max"] > np.percentile(sound["tophat_max"], 50)).mean())


def test_evaluate_counts_misses_and_false_alarms_the_right_way_round():
    labels = np.array([1, 1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0, 1])
    result = evaluate(labels, predicted)
    assert result["missed_defects"] == 1
    assert result["false_alarms"] == 1
    assert result["recall_defect"] == pytest.approx(2 / 3)
    assert result["precision_defect"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(3 / 5)


def test_report_names_both_classes():
    text = report(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert "defective" in text and "ok" in text


def test_svm_scales_inside_its_pipeline_and_the_forest_does_not_need_to():
    models = build_models()
    svm, _ = models["svm_rbf"]
    forest, _ = models["random_forest"]
    assert [name for name, _ in svm.steps] == ["scale", "model"]
    assert [name for name, _ in forest.steps] == ["model"]


def test_tuning_and_decision_scores():
    frame = make_feature_frame(60)
    columns = ["tophat_max", "tophat_p95", "sobel_mean"]
    for name in ("random_forest", "svm_rbf"):
        model, grid = build_models()[name]
        small = {key: values[:2] for key, values in grid.items()}
        search = tune(model, small, frame[columns], frame.label.to_numpy(), folds=3)
        scores = decision_scores(search, frame[columns])
        assert scores.shape == (len(frame),) and np.isfinite(scores).all()


def test_wilson_interval_matches_a_hand_computed_value():
    """0 events out of 207 has Wilson upper bound z^2 / (n + z^2)."""
    low, high = proportion_interval(0, 207)
    z = 1.959963984540054
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(z**2 / (207 + z**2), rel=1e-6)


def test_headline_intervals_cover_the_estimates():
    labels = np.array([1] * 30 + [0] * 20)
    predicted = labels.copy()
    predicted[:3] = 0
    table = headline_intervals(labels, predicted)
    assert (table["95%_low"] <= table["estimate"]).all()
    assert (table["estimate"] <= table["95%_high"]).all()


def test_mcnemar_uses_only_the_disagreements():
    labels = np.zeros(20, dtype=int)
    first = np.zeros(20, dtype=int)
    second = np.zeros(20, dtype=int)
    second[:6] = 1  # six images only the first classifier gets right
    result = mcnemar_exact(labels, first, second)
    assert result["only_first_right"] == 6 and result["only_second_right"] == 0
    assert result["p_value"] == pytest.approx(2 * 0.5**6)


def test_timing_reports_per_image_statistics():
    calls = []
    result = time_inspection(["a", "b", "c"], lambda p: p, lambda image: calls.append(image) or {},
                             repeats=2)
    assert calls == ["a", "b", "c"] * 2
    assert result["images_timed"] == 6
    assert result["p95_ms"] >= result["median_ms"] >= 0


def test_exposure_matching_balances_the_classes_and_closes_the_gap():
    frame = make_feature_frame(120)
    matched = exposure_matched_pairs(frame, tolerance=3.0)
    assert len(matched) % 2 == 0 and matched["label"].mean() == pytest.approx(0.5)
    gap = abs(matched[matched.label == 1]["nuisance_corner_mean"].mean()
              - matched[matched.label == 0]["nuisance_corner_mean"].mean())
    unmatched = abs(frame[frame.label == 1]["nuisance_corner_mean"].mean()
                    - frame[frame.label == 0]["nuisance_corner_mean"].mean())
    assert gap < unmatched


def test_matching_requires_the_nuisance_column():
    frame = make_feature_frame(20).drop(columns=["nuisance_corner_mean"])
    with pytest.raises(ValueError, match="include_nuisance"):
        exposure_matched_pairs(frame)


def test_matched_control_compares_equal_sized_balanced_arms():
    frame = make_feature_frame(160)
    matched = exposure_matched_pairs(frame, tolerance=3.0)
    subset = balanced_random_subset(frame, len(matched) // 2, seed=0)
    assert len(subset) == len(matched) and subset["label"].mean() == pytest.approx(0.5)
    model, _ = build_models()["svm_rbf"]
    table = matched_control(frame, matched, model, ["tophat_max", "sobel_mean"],
                            ["nuisance_corner_mean"], draws=3, folds=3)
    assert list(table["images"]) == [len(matched), len(matched)]


def test_separation_table_orients_inverted_features():
    """A feature lower in defective castings separates as well as one that is higher."""
    frame = make_feature_frame(100)
    frame["inverted"] = -frame["tophat_max"]
    table = separation_table(frame, ["tophat_max", "inverted"]).set_index("feature")
    assert table.loc["inverted", "separation_all"] == pytest.approx(table.loc["tophat_max", "separation_all"])
    assert table.loc["inverted", "higher_in"] == "ok"
    assert table.loc["tophat_max", "higher_in"] == "defective"
