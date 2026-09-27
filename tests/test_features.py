"""
Tests for the feature pipeline.

The two properties worth pinning are that the detector finds a defect where one
was placed, and that no feature can see the background. The second is the one
that matters most: the dataset's lighting confound means a feature touching
background would score well for the wrong reason, and nothing about the
accuracy number would reveal it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casting_qa.config import RIM_OUTER_FACTOR
from casting_qa.features import (
    decomposition_check,
    defect_response,
    denoise,
    describe,
    describe_batch,
    feature_columns,
    features_from,
    normalize_to_part,
    pipeline_stages,
    prefilter,
    quality_metrics,
    texture_input,
    wedge_features,
)
from casting_qa.geometry import find_anchor, radius_map, rim_mask
from tests.conftest import add_defect, make_casting, rim_point


def test_response_fires_on_a_defect_and_stays_quiet_without_one():
    """The detector has to separate a pitted surface from a clean one."""
    good = make_casting()
    bad = add_defect(good, rim_point())

    def peak(image):
        anchor = find_anchor(image)
        band = rim_mask(image, anchor)
        return defect_response(normalize_to_part(denoise(image), band), band).max()

    assert peak(bad) > peak(good) * 2, "a blow hole must respond far above a clean surface"


def test_no_feature_can_see_the_background():
    """
    Changing the background alone must not move a single shipped feature.

    This is the guard for the dataset's confound. Defective images were shot
    against a darker table, so any feature reacting to background inherits a
    separation that has nothing to do with the casting. Repainting the
    background is the direct test of whether that leak exists.
    """
    image = make_casting(background=200)
    anchor = find_anchor(image)
    outside = ~rim_mask(image, anchor)

    darker = image.copy()
    darker[outside] = np.clip(darker[outside].astype(int) - 45, 0, 255).astype(np.uint8)

    before = describe(image)
    after = describe(darker)

    # Not exactly zero, and the reason is worth stating: the 3x3 median runs
    # before the band exists, so pixels at the band's edge have background in
    # their filter window no matter what happens afterwards. The bound below is
    # what remains once every operator with a spatial footprint reads from the
    # isolated image. Before that fix sobel_max moved by 0.16 under this same
    # 45-level repaint, 160 times the tolerance here.
    for name in feature_columns(pd.DataFrame([before])):
        assert before[name] == pytest.approx(after[name], abs=1e-3), (
            f"{name} moved when only the background was repainted"
        )


def test_nuisance_features_do_see_the_background():
    """
    The mirror of the previous test: the confound probes must react.

    If they did not, the notebook's measurement of how much a cheating model
    can score would be meaningless.
    """
    image = make_casting(background=200)
    darker = image.copy()
    darker[:40, :40] = 150

    before = describe(image, include_nuisance=True)
    after = describe(darker, include_nuisance=True)
    assert before["nuisance_corner_mean"] != pytest.approx(after["nuisance_corner_mean"])


def test_nuisance_probes_never_read_the_casting():
    """Repainting the part must leave every background probe exactly where it was."""
    image = make_casting(background=200)
    stages = pipeline_stages(image)
    repainted = image.copy()
    inside = radius_map(image.shape, stages.anchor) <= RIM_OUTER_FACTOR
    repainted[inside] = np.clip(repainted[inside].astype(int) - 60, 0, 255).astype(np.uint8)
    before = describe(image, include_nuisance=True)
    after = describe(repainted, include_nuisance=True)
    for name in ("nuisance_corner_mean", "nuisance_background_mean", "nuisance_background_std"):
        assert before[name] == pytest.approx(after[name]), f"{name} moved when only the part changed"


def test_normalization_cancels_a_uniform_exposure_change_pixel_by_pixel():
    """
    Scaling the whole photograph's brightness must leave every normalized pixel alone.

    Comparing medians would pass by construction, since each is 1.0 after
    dividing by itself, so the check is on the pixels, up to 8-bit rounding.
    """
    image = make_casting(rim=150, background=140)
    band = rim_mask(image, find_anchor(image))
    brighter = np.round(image.astype(float) * 1.3).astype(np.uint8)

    a = normalize_to_part(image, band)[band]
    b = normalize_to_part(brighter, band)[band]
    assert np.abs(a - b).max() < 0.02


def test_wedge_statistics_are_orientation_independent():
    """Rolling the unwrapping must not change order statistics over wedges."""
    polar = np.random.default_rng(0).normal(1.0, 0.1, (360, 64))
    polar[40:60] -= 0.5
    a = wedge_features(polar)
    b = wedge_features(np.roll(polar, 137, axis=0))
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=0.02), f"{key} moved when the part rotated"


def test_describe_returns_finite_values_for_a_blank_frame():
    """A degenerate image must not produce NaN that poisons the feature table."""
    blank = np.full((256, 256), 200, dtype=np.uint8)
    assert all(np.isfinite(v) for v in describe(blank).values())


def test_texture_input_keeps_the_bright_half_of_the_band_unsaturated():
    """
    Scaling the normalized band by 255 directly would clip every pixel brighter
    than the median. The mapping must leave almost none at the ceiling.
    """
    image = make_casting()
    stages = pipeline_stages(image)
    levels = texture_input(stages.isolated)[stages.band]
    assert (levels == 255).mean() < 0.01
    assert 100 < np.median(levels) < 155


def test_shared_stages_give_the_same_features_as_describe():
    image = make_casting()
    assert features_from(pipeline_stages(image)) == describe(image)


def test_describe_batch_labels_each_row():
    images = {"a": make_casting(seed=1), "b": make_casting(seed=2)}
    inventory = pd.DataFrame({"path": ["a", "b"], "filename": ["a.png", "b.png"], "label": [1, 0]})
    frame = describe_batch(inventory, images.__getitem__, n_jobs=1)
    assert list(frame["filename"]) == ["a.png", "b.png"] and list(frame["label"]) == [1, 0]


def test_quality_metrics_see_blur():
    from scipy import ndimage

    sharp = make_casting()
    blurred = ndimage.gaussian_filter(sharp, 2)
    assert quality_metrics(blurred)["sharpness"] < quality_metrics(sharp)["sharpness"]
    assert quality_metrics(sharp)["anchor_found"] == 1.0


def test_feature_columns_excludes_identifiers_and_nuisance():
    frame = pd.DataFrame([describe(make_casting(), include_nuisance=True)])
    frame["filename"] = "x.jpeg"
    frame["label"] = 1
    columns = feature_columns(frame)
    assert "label" not in columns and "filename" not in columns
    assert not any(c.startswith("nuisance_") for c in columns)
    assert "nuisance_corner_mean" in feature_columns(frame, include_nuisance=True)


def test_prefilters_differ_and_unknown_ones_fail():
    image = make_casting()
    assert not np.array_equal(prefilter(image, "median"), prefilter(image, "gaussian"))
    with pytest.raises(ValueError, match="unknown prefilter"):
        prefilter(image, "bilateral")


def test_group_travels_with_the_features_but_is_not_one():
    images = {"a": make_casting(seed=1), "b": make_casting(seed=2)}
    inventory = pd.DataFrame({"path": ["a", "b"], "filename": ["a.png", "b.png"],
                              "label": [1, 0], "group": [3, 3]})
    frame = describe_batch(inventory, images.__getitem__, n_jobs=1)
    assert list(frame["group"]) == [3, 3]
    assert "group" not in feature_columns(frame)


def test_decomposition_check_reports_speed_and_change():
    images = {"a": make_casting(seed=1)}
    result = decomposition_check(["a"], images.__getitem__)
    assert result["images"] == 1
    assert result["plain_ms"] > 0 and result["decomposed_ms"] > 0
    assert result["mean_pixel_change_%"] >= 0
