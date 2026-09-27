"""Tests for inventory, auditing, cleaning and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from casting_qa.config import DEFECT, OK
from casting_qa.dataset import (
    audit_inventory,
    audit_summary,
    build_inventory,
    class_balance,
    clean_inventory,
    load_grayscale,
    partition_of,
    split_inventory,
    split_summary,
)
from tests.conftest import make_casting


def write_set(root, defective: int, ok: int, duplicate: bool = False, blank: bool = False,
              corrupt: bool = False):
    """A miniature dataset on disk, shaped like the real one."""
    for folder, count, tag in (("def_front", defective, "def"), ("ok_front", ok, "ok")):
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            # Different seeds per folder, so no sound and defective file share pixels by accident.
            seed = i if tag == "def" else 1000 + i
            Image.fromarray(make_casting(size=64, seed=seed)).save(directory / f"cast_{tag}_{i}.png")
    if duplicate:
        Image.fromarray(make_casting(size=64, seed=0)).save(root / "def_front" / "copy.png")
    if blank:
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(root / "ok_front" / "blank.png")
    if corrupt:
        (root / "ok_front" / "broken.png").write_bytes(b"not an image")
    return root


def test_inventory_labels_come_from_the_folder(tmp_path):
    inventory = build_inventory(write_set(tmp_path, defective=3, ok=2), glob="*.png")
    assert len(inventory) == 5
    assert (inventory[inventory.filename.str.contains("def")].label == DEFECT).all()
    assert (inventory[inventory.filename.str.contains("ok")].label == OK).all()


def test_missing_folders_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_inventory(tmp_path / "nothing")


def test_grayscale_load_collapses_identical_planes_and_the_audit_checks_them(tmp_path):
    """The files are three-channel with identical planes; the audit verifies it per file."""
    for folder in ("def_front", "ok_front"):
        (tmp_path / folder).mkdir()
    same = np.dstack([make_casting(size=64)] * 3)
    different = same.copy()
    different[..., 2] = 0
    Image.fromarray(same).save(tmp_path / "def_front" / "same.png")
    Image.fromarray(different).save(tmp_path / "ok_front" / "different.png")

    audited = audit_inventory(build_inventory(tmp_path, glob="*.png"))
    assert audited.set_index("filename").loc["same.png", "planes_identical"]
    assert not audited.set_index("filename").loc["different.png", "planes_identical"]
    assert load_grayscale(tmp_path / "def_front" / "same.png").ndim == 2


def test_each_problem_is_counted_once_and_removed_for_its_own_reason(tmp_path):
    """One duplicate, one blank and one corrupt file, each reported under its own name."""
    inventory = build_inventory(
        write_set(tmp_path, defective=4, ok=4, duplicate=True, blank=True, corrupt=True), glob="*.png")
    audited = audit_inventory(inventory)
    summary = audit_summary(audited).set_index("check")["count"]
    assert summary["unreadable"] == 1
    assert summary["not 512x512"] == len(inventory) - 1  # the corrupt file is not counted twice
    assert summary["identical decoded pixels"] == 2
    assert summary["blank (zero variance)"] == 1

    cleaned, removed = clean_inventory(audited)
    assert sorted(removed["reason"]) == ["blank frame", "identical pixels", "unreadable"]
    assert len(cleaned) == len(inventory) - 3
    assert not cleaned["digest"].duplicated().any()


def test_identical_pixels_with_conflicting_labels_drop_every_copy(tmp_path):
    write_set(tmp_path, defective=2, ok=2)
    Image.fromarray(make_casting(size=64, seed=0)).save(tmp_path / "ok_front" / "clash.png")
    cleaned, removed = clean_inventory(audit_inventory(build_inventory(tmp_path, glob="*.png")))
    assert (removed["reason"] == "identical pixels, conflicting labels").sum() == 2
    assert "clash.png" not in set(cleaned["filename"])


def test_split_fractions_and_stratification_hold():
    frame = pd.DataFrame({
        "path": [f"p{i}" for i in range(1000)],
        "filename": [f"f{i}" for i in range(1000)],
        "label": [DEFECT] * 600 + [OK] * 400,
    })
    explore, validation, test = split_inventory(frame)
    assert len(explore) == 300 and len(validation) == 300 and len(test) == 400
    for part in (explore, validation, test):
        assert (part.label == DEFECT).mean() == pytest.approx(0.6, abs=0.02)


def test_splits_do_not_overlap():
    """A file in two partitions would turn part of the test score into memory."""
    frame = pd.DataFrame({
        "path": [f"p{i}" for i in range(200)],
        "filename": [f"f{i}" for i in range(200)],
        "label": [DEFECT] * 120 + [OK] * 80,
    })
    explore, validation, test = split_inventory(frame)
    assert not (set(explore.path) & set(validation.path))
    assert not (set(explore.path) & set(test.path))
    assert not (set(validation.path) & set(test.path))


def test_split_fractions_must_sum_to_one():
    frame = pd.DataFrame({"path": ["a"], "filename": ["a"], "label": [OK]})
    with pytest.raises(ValueError, match="sum to 1"):
        split_inventory(frame, explore_fraction=0.5, validation_fraction=0.3, test_fraction=0.1)


def test_split_fractions_must_be_whole_folds():
    frame = pd.DataFrame({"path": ["a"], "filename": ["a"], "label": [OK]})
    with pytest.raises(ValueError, match="whole number"):
        split_inventory(frame, explore_fraction=0.25, validation_fraction=0.35, test_fraction=0.4)


def test_grouped_split_keeps_every_group_in_one_partition():
    """The same casting photographed twice must never straddle two partitions."""
    n = 600
    frame = pd.DataFrame({
        "path": [f"p{i}" for i in range(n)],
        "filename": [f"f{i}" for i in range(n)],
        "label": [DEFECT if i % 5 < 3 else OK for i in range(n)],
    })
    groups = np.arange(n) // 3  # every consecutive triple is one casting
    explore, validation, test = split_inventory(frame, groups)
    where = partition_of(explore, validation, test)
    group_of = pd.Series(groups, index=frame["filename"])
    assert (where.groupby(group_of.reindex(where.index)).nunique() == 1).all()
    assert len(explore) + len(validation) + len(test) == n
    assert len(test) == pytest.approx(0.4 * n, abs=15)


def test_summaries_describe_the_partitions():
    frame = pd.DataFrame({
        "path": [f"p{i}" for i in range(100)],
        "filename": [f"f{i}" for i in range(100)],
        "label": [DEFECT] * 60 + [OK] * 40,
    })
    balance = class_balance(frame)
    assert set(balance["label"]) == {"ok", "defective"}
    assert balance["images"].sum() == 100

    summary = split_summary(*split_inventory(frame))
    assert list(summary["partition"]) == ["explore", "validation", "test"]
    assert summary["share_%"].sum() == pytest.approx(100, abs=0.5)
