"""Tests for the perturbations used in the robustness check."""

from __future__ import annotations

import numpy as np
import pytest

from casting_qa.robustness import perturb
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
