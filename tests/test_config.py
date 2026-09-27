"""The constants that other modules assume without checking."""

from __future__ import annotations

import numpy as np

from casting_qa import config


def test_split_fractions_are_whole_folds_summing_to_one():
    fractions = (config.EXPLORE_FRACTION, config.VALIDATION_FRACTION, config.TEST_FRACTION)
    assert np.isclose(sum(fractions), 1.0)
    assert all(np.isclose(f * config.SPLIT_FOLDS, round(f * config.SPLIT_FOLDS)) for f in fractions)


def test_radii_are_ordered_from_cavity_to_background():
    assert 1.0 < config.RIM_INNER_FACTOR < config.RIM_OUTER_FACTOR < config.BACKGROUND_INNER_FACTOR
    assert config.BACKGROUND_INNER_FACTOR < config.RADIAL_PROFILE_EXTENT


def test_localizer_grid_and_quantiles_are_sensible():
    assert all(0.99 < q < 1.0 for q in config.LOCALIZER_QUANTILES)
    assert list(config.LOCALIZER_AREA_GRID) == sorted(config.LOCALIZER_AREA_GRID)
    assert 0 < config.LOCALIZER_FALSE_BOX_TARGET < 1


def test_rule_features_are_named_model_features():
    assert len(set(config.RULE_FEATURES)) == len(config.RULE_FEATURES)
    assert not any(name.startswith("nuisance_") for name in config.RULE_FEATURES)
