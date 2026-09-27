"""
Tests for near-duplicate detection.

The property that matters is the one the split depends on: the same casting
photographed at another rotation must score high, and two different castings
of the same design must score low, even though they share every structural
feature of the part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from casting_qa.duplicates import (
    cross_partition_pairs,
    detail_signature,
    duplicate_groups,
    images_with_a_twin_in,
    pairwise_similarity,
    similar_pairs,
    threshold_sweep,
)
from tests.conftest import add_marks, make_casting


def test_a_rotated_photograph_of_the_same_casting_matches_and_another_casting_does_not():
    same = add_marks(make_casting(seed=1), seed=1)
    rotated = np.rot90(same).copy()
    other = add_marks(make_casting(seed=2), seed=2)

    stack = np.stack([detail_signature(image) for image in (same, rotated, other)])
    similarity = pairwise_similarity(stack)
    assert similarity[0, 1] > 0.6, "a quarter turn should not hide the same casting"
    assert similarity[0, 2] < 0.3, "two castings of one design must not look identical"


def test_groups_follow_chains_of_matches():
    """If A matches B and B matches C, all three belong to one group."""
    similarity = np.zeros((4, 4), dtype=np.float32)
    similarity[0, 1] = similarity[1, 2] = 0.9
    groups = duplicate_groups(similarity, threshold=0.6)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_threshold_sweep_counts_cross_label_pairs():
    similarity = np.zeros((4, 4), dtype=np.float32)
    similarity[0, 1] = 0.9   # same label
    similarity[2, 3] = 0.7   # different labels
    sweep = threshold_sweep(similarity, [1, 1, 0, 1], thresholds=(0.8, 0.6)).set_index("threshold")
    assert sweep.loc[0.8, "pairs"] == 1 and sweep.loc[0.8, "cross_label_pairs"] == 0
    assert sweep.loc[0.6, "pairs"] == 2 and sweep.loc[0.6, "cross_label_pairs"] == 1
    assert sweep.loc[0.6, "images_with_a_twin"] == 4


def test_cross_partition_pairs_finds_only_the_leaks():
    similarity = np.zeros((3, 3), dtype=np.float32)
    similarity[0, 1] = similarity[0, 2] = 0.9
    pairs = similar_pairs(similarity, ["a", "b", "c"], [1, 1, 1], threshold=0.6)
    partition = pd.Series({"a": "explore", "b": "explore", "c": "test"})
    leaked = cross_partition_pairs(pairs, partition)
    assert len(leaked) == 1
    assert set(leaked.iloc[0][["first", "second"]]) == {"a", "c"}


def test_twins_are_counted_from_the_target_partition_only():
    pairs = pd.DataFrame({"first": ["a", "b", "c"], "second": ["x", "y", "z"]})
    partition = pd.Series({"a": "test", "x": "explore", "b": "explore", "y": "test",
                           "c": "test", "z": "validation"})
    assert images_with_a_twin_in(pairs, partition, "test", "explore") == {"a", "y"}
