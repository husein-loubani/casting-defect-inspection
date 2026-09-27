"""
Tests for locating the part.

Everything downstream reads from the anchor, so if it is wrong every feature is
measured in the wrong place and no later test would notice. These pin the
property that actually matters: the anchor finds the cavity, and the band it
produces contains metal rather than background.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casting_qa.features import wedge_features
from casting_qa.geometry import (
    Anchor,
    background_mask,
    class_profiles,
    find_anchor,
    otsu_largest_share,
    radial_profile,
    radius_map,
    rim_mask,
    structuring_disk,
    unwrap,
)
from tests.conftest import add_defect, make_casting, rim_point


def test_anchor_finds_the_cavity_center():
    """The cavity is centered by construction, so the anchor must land there."""
    image = make_casting(size=256)
    anchor = find_anchor(image)
    assert anchor.found
    assert anchor.center[0] == pytest.approx(128, abs=8)
    assert anchor.center[1] == pytest.approx(128, abs=8)


def test_anchor_radius_tracks_the_cavity_not_the_frame():
    """
    A larger cavity must give a larger radius.

    This is the property that lets every other measurement be expressed in
    units of the part rather than in pixels, so it survives the part being
    photographed nearer or further away.
    """
    small = find_anchor(make_casting(size=256))
    large = find_anchor(make_casting(size=384))
    assert large.radius > small.radius * 1.2


def test_anchor_reports_failure_rather_than_guessing():
    """
    An image with nothing dark in it has no cavity to find.

    Returning `found=False` instead of a plausible center is what lets the
    notebook count failures. A silent fallback would put the inspection band in
    the wrong place and nothing downstream would complain.
    """
    blank = np.full((256, 256), 200, dtype=np.uint8)
    anchor = find_anchor(blank)
    assert not anchor.found


def test_rim_mask_excludes_background_and_cavity():
    """
    The band must contain machined surface only.

    Background is the confounded region and the cavity is dark in every image,
    so a band that includes either would measure something other than the
    surface being inspected.
    """
    image = make_casting(size=256)
    anchor = find_anchor(image)
    band = rim_mask(image, anchor)

    assert band.any()
    scaled = radius_map(image.shape, anchor)
    assert scaled[band].min() >= 1.15 - 1e-9
    assert scaled[band].max() <= 2.05 + 1e-9

    corners = np.zeros_like(band)
    corners[:20, :20] = corners[:20, -20:] = True
    corners[-20:, :20] = corners[-20:, -20:] = True
    assert not (band & corners).any(), "the inspection band reaches image corners"


def test_unwrapping_turns_rotation_into_a_shift():
    """
    The reason polar coordinates are used at all.

    A defect at one angle and the same defect at another are the same event.
    After unwrapping, the two differ by a cyclic shift of the rows, so a
    statistic taken over the rows is unchanged. Sorting the wedge means is the
    cheapest way to assert that without depending on the shift's size.
    """
    base = make_casting(size=256)
    first = add_defect(base, rim_point(256, angle=0.0))
    second = add_defect(base, rim_point(256, angle=np.pi / 2))

    a = wedge_features(unwrap(first))
    b = wedge_features(unwrap(second))
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=0.03), (
            f"{key} moved when the same defect was placed at a different angle"
        )


def test_anchor_repr_is_readable():
    anchor = Anchor((10.0, 20.0), 5.0, True)
    assert repr(anchor) == "Anchor(center=(10, 20), radius=5, found=True)"


def test_radial_profile_is_dark_in_the_cavity_and_bright_on_the_rim():
    """The profile must reproduce the structure the anchor was built on."""
    image = make_casting(size=256)
    values, edges = radial_profile(image, bins=24, extent=2.8)
    assert len(values) == 24 and len(edges) == 25
    inside = values[(edges[:-1] >= 0.3) & (edges[1:] <= 0.9)]
    rim = values[(edges[:-1] >= 1.2) & (edges[1:] <= 2.0)]
    assert np.nanmean(inside) < np.nanmean(rim), "cavity should be darker than the rim"


def test_background_mask_stays_clear_of_the_part():
    image = make_casting(size=256)
    anchor = find_anchor(image)
    assert not (background_mask(image, anchor) & rim_mask(image, anchor)).any()
    assert radius_map(image.shape, anchor)[background_mask(image, anchor)].min() >= 2.3 - 1e-9
    assert background_mask(image, anchor)[0, 0], "the corner is workbench"


def test_otsu_largest_share_is_the_fraction_of_the_frame():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, :60] = 200
    assert otsu_largest_share(image) == pytest.approx(0.6)


def test_decomposed_disk_approximates_the_plain_disk():
    """The speed-up must not change what the element covers by much."""
    from skimage import morphology

    point = np.zeros((61, 61), dtype=np.uint8)
    point[30, 30] = 1
    plain = morphology.dilation(point, structuring_disk(21, decompose=False))
    decomposed = morphology.dilation(point, structuring_disk(21))
    assert plain.sum() == 1373
    # The octagon covers about 6% more pixels than the disk it replaces.
    assert abs(int(decomposed.sum()) - 1373) / 1373 < 0.10


def test_axis_ratio_is_one_for_a_round_cavity_and_less_for_a_foreshortened_one():
    round_image = make_casting(size=256)
    # Every other row: the part photographed at a steep tilt, half as tall as it is wide.
    squashed = np.vstack([round_image[::2], np.full((128, 256), 200, dtype=np.uint8)])
    assert find_anchor(round_image).axis_ratio > 0.95
    assert find_anchor(squashed).axis_ratio < 0.7


def test_class_profiles_average_each_class():
    images = {"d": make_casting(seed=1), "o": make_casting(seed=2)}
    inventory = pd.DataFrame({"path": ["d", "o"], "label": [1, 0]})
    profiles, edges = class_profiles(inventory, images.__getitem__, per_class=1)
    assert set(profiles) == {"defective", "ok"}
    assert len(edges) == len(profiles["ok"]) + 1
