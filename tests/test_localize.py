"""
Tests for defect localization.

The localizer is judged by where it points. A planted defect must be boxed at
the coordinate where it was drawn, and the circular structure every casting
shares must not be boxed, because the reference fitted on sound castings
already contains it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casting_qa.config import DEFECT, OK
from casting_qa.features import pipeline_stages
from casting_qa.localize import (
    fit_localizer,
    inspect_casting,
    inspection_gallery,
    locate,
    quantile_curve,
)
from tests.conftest import add_defect, make_casting, rim_point


@pytest.fixture(scope="module")
def fitted():
    images = {f"ok{i}": make_casting(seed=i) for i in range(8)}
    images.update({f"def{i}": add_defect(make_casting(seed=50 + i), rim_point(angle=0.5 * i), radius=6)
                   for i in range(8)})
    explore = pd.DataFrame({"path": list(images), "label": [OK] * 8 + [DEFECT] * 8})
    return fit_localizer(explore, images.__getitem__, n_jobs=1)


def test_quantile_curve_reads_the_histogram_by_hand():
    # One ring, response values spread evenly over the first 10 histogram bins.
    histograms = np.zeros((1, 4000))
    histograms[0, :10] = 1
    assert quantile_curve(histograms, 0.5)[0] == pytest.approx(5 * 2.0 / 4000)
    assert quantile_curve(histograms, 1.0)[0] == pytest.approx(10 * 2.0 / 4000)


def test_the_planted_defect_is_boxed_where_it_was_drawn(fitted):
    where = rim_point(angle=2.0)
    image = add_defect(make_casting(seed=99), where, radius=7, depth=110)
    found = locate(pipeline_stages(image), fitted)
    assert found["strong"], "a deep blow hole must clear the sound-metal reference"
    r0, c0, r1, c1 = found["strong"][0][0]
    assert r0 <= where[0] <= r1 and c0 <= where[1] <= c1


def test_a_sound_casting_gets_no_strong_box(fitted):
    found = locate(pipeline_stages(make_casting(seed=123)), fitted)
    assert not found["strong"], "the rings every casting has must stay under the reference"
    assert len(found["weak"]) == 1, "the fallback always names one most anomalous spot"


def test_the_selection_respects_the_false_box_target(fitted):
    chosen = fitted.selection[(fitted.selection["quantile"] == fitted.quantile)
                              & (fitted.selection["min_area"] == fitted.min_area)].iloc[0]
    assert chosen["sound_boxed"] <= 0.10


class _AlwaysDefect:
    def predict(self, frame):
        return np.full(len(frame), DEFECT)


class _AlwaysSound:
    def predict(self, frame):
        return np.full(len(frame), OK)


def test_inspection_boxes_only_when_the_classifier_flags(fitted):
    image = add_defect(make_casting(seed=7), rim_point(angle=1.0), radius=7, depth=110)
    columns = ["tophat_max", "sobel_mean"]
    flagged = inspect_casting(image, _AlwaysDefect(), columns, fitted)
    passed = inspect_casting(image, _AlwaysSound(), columns, fitted)
    assert flagged["decision"] == DEFECT and (flagged["boxes"]["strong"] or flagged["boxes"]["weak"])
    assert passed["decision"] == OK and not passed["boxes"]["strong"] and not passed["boxes"]["weak"]


def test_gallery_captions_follow_the_decision(fitted):
    images = {"x": make_casting(seed=3)}
    path_of = pd.Series({"x.png": "x"})
    flagged = inspection_gallery(["x.png"], path_of, images.__getitem__, _AlwaysDefect(), ["tophat_max"], fitted)
    passed = inspection_gallery(["x.png"], path_of, images.__getitem__, _AlwaysSound(), ["tophat_max"], fitted)
    assert "flagged" in flagged[0][2] and "passed" in passed[0][2]
