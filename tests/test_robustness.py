"""Tests for the perturbations used in the robustness check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from casting_qa.config import DEFECT, OK
from casting_qa.robustness import (
    augment_batch,
    augment_image,
    perturb,
    perturbed_features,
    robustness_table,
)
from tests.conftest import make_casting


def test_neutral_settings_leave_the_image_alone():
    image = make_casting()
    assert np.array_equal(perturb(image, "gain", 1.0), image)
    assert np.array_equal(perturb(image, "gamma", 1.0), image)
    assert np.abs(perturb(image, "rotate", 0.0).astype(int) - image).max() <= 1


def test_gain_and_gamma_move_brightness_the_expected_way():
    image = make_casting()
    assert perturb(image, "gain", 1.1).mean() > image.mean()
    assert perturb(image, "gamma", 1.1).mean() < image.mean()  # gamma above 1 darkens mid-tones


def test_jpeg_keeps_shape_and_adds_small_errors():
    image = make_casting()
    recompressed = perturb(image, "jpeg", 50)
    assert recompressed.shape == image.shape and recompressed.dtype == np.uint8
    assert 0 < np.abs(recompressed.astype(int) - image).mean() < 5


def test_unknown_perturbations_fail_loudly():
    with pytest.raises(ValueError, match="unknown"):
        perturb(make_casting(), "blur", 1.0)


def test_augment_image_always_changes_something():
    rng = np.random.default_rng(0)
    image = make_casting()
    for _ in range(5):
        variant = augment_image(image, rng)
        assert variant.shape == image.shape and variant.dtype == np.uint8
        assert not np.array_equal(variant, image)


def test_augment_batch_keeps_each_copy_with_its_source():
    images = {"a": make_casting(seed=1), "b": make_casting(seed=2)}
    inventory = pd.DataFrame({"path": ["a", "b"], "filename": ["a.png", "b.png"],
                              "label": [DEFECT, OK], "group": [7, 9]})
    augmented = augment_batch(inventory, images.__getitem__, copies=2, n_jobs=1)
    assert list(augmented["filename"]) == ["a.png", "a.png", "b.png", "b.png"]
    assert list(augmented["group"]) == [7, 7, 9, 9]
    assert augmented["augmented"].all()


class _Threshold:
    def predict(self, frame):
        return (frame["tophat_max"].to_numpy() > 0.5).astype(int)


def test_robustness_table_counts_flips_against_the_unmodified_decision():
    images = {"a": make_casting(seed=1)}
    inventory = pd.DataFrame({"path": ["a"], "filename": ["a.png"], "label": [OK]})
    perturbed = perturbed_features(inventory, images.__getitem__, perturbations=(("gain", 1.0),), n_jobs=1)
    clean = perturbed["gain 1"]
    table = robustness_table(clean, perturbed, {"rule": _Threshold()}, ["tophat_max"])
    assert list(table["perturbation"]) == ["none", "gain 1"]
    assert table.loc[table.perturbation == "gain 1", "flipped"].iloc[0] == 0
