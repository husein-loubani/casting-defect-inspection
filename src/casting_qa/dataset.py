"""
Loading, auditing, cleaning and splitting the casting images.

The images arrive as two folders of JPEGs. Everything here works on the file
inventory rather than on pixel arrays wherever it can, because 1,300 images at
512x512 is 340 MB in memory and most questions can be answered without paying
that.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from casting_qa.config import (
    CLASS_NAMES,
    DEFECT,
    DEFECT_DIR,
    EXPLORE_FRACTION,
    IMAGE_GLOB,
    IMAGE_SIZE,
    OK,
    OK_DIR,
    RANDOM_SEED,
    RAW_DIR,
    SPLIT_FOLDS,
    TEST_FRACTION,
    VALIDATION_FRACTION,
)

PARTITIONS = ("explore", "validation", "test")


def build_inventory(raw_dir: str | Path = RAW_DIR, glob: str = IMAGE_GLOB) -> pd.DataFrame:
    """
    One row per image file, with its label taken from the folder it sits in.

    The folder name is the ground truth here; there is no label file to join
    against and no ambiguity to resolve.
    """
    raw_dir = Path(raw_dir)
    rows = []
    for folder, label in ((DEFECT_DIR, DEFECT), (OK_DIR, OK)):
        directory = raw_dir / folder
        if not directory.exists():
            raise FileNotFoundError(f"missing {directory}; expected the casting_512x512 folders")
        for path in sorted(directory.glob(glob)):
            rows.append({"path": str(path), "filename": path.name, "label": label})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no {glob} files under {raw_dir}")
    return frame.reset_index(drop=True)


def load_grayscale(path: str | Path) -> np.ndarray:
    """
    Read one image as a single-channel uint8 array.

    The files are stored as three-channel JPEG with identical planes (the audit
    checks every file), so `convert("L")` discards two thirds of the bytes
    without discarding any information.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _planes_identical(array: np.ndarray) -> bool:
    if array.ndim == 2:
        return True
    return bool(np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2]))


def audit_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Open every image once and record what it actually contains.

    Returns the inventory with size, mode, whether the color planes are
    identical, an intensity summary and a hash of the decoded pixels attached.
    The hash is of pixels rather than file bytes because two JPEGs can differ
    byte for byte and decode to the same picture.
    """
    records = []
    for path in inventory["path"]:
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                stored = np.asarray(image)
            gray = load_grayscale(path)
            records.append({
                "width": width,
                "height": height,
                "mode": mode,
                "planes_identical": _planes_identical(stored),
                "mean": float(gray.mean()),
                "std": float(gray.std()),
                "digest": hashlib.sha256(gray.tobytes()).hexdigest(),
                "readable": True,
            })
        except (OSError, ValueError):  # a corrupt file is data to report, not a crash
            records.append({
                "width": np.nan, "height": np.nan, "mode": None, "planes_identical": False,
                "mean": np.nan, "std": np.nan, "digest": None, "readable": False,
            })
    return pd.concat([inventory.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def audit_summary(audited: pd.DataFrame) -> pd.DataFrame:
    """The audit as a short table. Each check counts readable files only, once."""
    readable = audited[audited["readable"]]
    wrong_size = (readable["width"] != IMAGE_SIZE) | (readable["height"] != IMAGE_SIZE)
    return pd.DataFrame([
        {"check": "images", "count": len(audited)},
        {"check": "unreadable", "count": int((~audited["readable"]).sum())},
        {"check": f"not {IMAGE_SIZE}x{IMAGE_SIZE}", "count": int(wrong_size.sum())},
        {"check": "color planes differ", "count": int((~readable["planes_identical"]).sum())},
        {"check": "identical decoded pixels", "count": int(readable["digest"].duplicated(keep=False).sum())},
        {"check": "blank (zero variance)", "count": int((readable["std"] == 0).sum())},
    ])


def clean_inventory(audited: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop what cannot be used, and report what was dropped and why.

    Three rules, each aimed at a defect that would otherwise corrupt a result:

    - unreadable files cannot be featurized at all;
    - blank frames carry no signal and would train the classifier on nothing;
    - identical pixels must go before the split, because the same picture
      landing in two partitions turns part of the test score into a memory check.

    An identical picture carrying two different labels is a contradiction
    rather than a redundancy, so every copy is dropped instead of keeping one
    arbitrarily. Near-duplicates, the same casting photographed twice, are a
    different problem: both photographs are genuine data, so they are kept and
    held in the same partition by the group-aware split instead.
    """
    removed = []

    frame = audited.copy()
    unreadable = frame[~frame["readable"]]
    if len(unreadable):
        removed.append(pd.DataFrame({"filename": unreadable["filename"], "reason": "unreadable"}))
    frame = frame[frame["readable"]]

    blank = frame[frame["std"] == 0]
    if len(blank):
        removed.append(pd.DataFrame({"filename": blank["filename"], "reason": "blank frame"}))
    frame = frame[frame["std"] > 0]

    label_count = frame.groupby("digest")["label"].nunique()
    contradictory = set(label_count[label_count > 1].index)
    if contradictory:
        rows = frame[frame["digest"].isin(contradictory)]
        removed.append(pd.DataFrame({"filename": rows["filename"],
                                     "reason": "identical pixels, conflicting labels"}))
        frame = frame[~frame["digest"].isin(contradictory)]

    repeated = frame["digest"].duplicated(keep="first")
    if repeated.any():
        rows = frame[repeated]
        removed.append(pd.DataFrame({"filename": rows["filename"], "reason": "identical pixels"}))
        frame = frame[~repeated]

    report = (pd.concat(removed, ignore_index=True) if removed
              else pd.DataFrame(columns=["filename", "reason"]))
    return frame.reset_index(drop=True), report


def _fold_counts(explore_fraction: float, validation_fraction: float,
                 test_fraction: float, folds: int) -> tuple[int, int, int]:
    total = explore_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"fractions must sum to 1, got {total}")
    counts = [fraction * folds for fraction in (explore_fraction, validation_fraction, test_fraction)]
    if not all(np.isclose(c, round(c)) for c in counts):
        raise ValueError(f"each fraction must be a whole number of 1/{folds} folds")
    return tuple(int(round(c)) for c in counts)


def split_inventory(
    cleaned: pd.DataFrame,
    groups: np.ndarray | None = None,
    explore_fraction: float = EXPLORE_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
    folds: int = SPLIT_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three stratified partitions, in the proportions the brief specifies.

    With `groups`, every image sharing a group id (the same casting photographed
    more than once) lands in the same partition. `StratifiedGroupKFold` cuts the
    data into ten folds that keep groups whole while balancing the class mix,
    and the folds are dealt out three, three and four. Without `groups` the
    split is a plain stratified one, kept so the notebook can measure how much
    it leaks.
    """
    explore_folds, validation_folds, _ = _fold_counts(
        explore_fraction, validation_fraction, test_fraction, folds)

    if groups is None:
        explore, remainder = train_test_split(
            cleaned, train_size=explore_fraction, stratify=cleaned["label"], random_state=seed)
        # Validation is a fraction of what is left, not of the whole.
        validation_share = validation_fraction / (validation_fraction + test_fraction)
        validation, test = train_test_split(
            remainder, train_size=validation_share, stratify=remainder["label"], random_state=seed)
    else:
        fold_of = np.empty(len(cleaned), dtype=int)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, held) in enumerate(splitter.split(cleaned, cleaned["label"], groups)):
            fold_of[held] = fold
        explore = cleaned[fold_of < explore_folds]
        validation = cleaned[(fold_of >= explore_folds)
                             & (fold_of < explore_folds + validation_folds)]
        test = cleaned[fold_of >= explore_folds + validation_folds]

    return (explore.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True))


def split_summary(explore: pd.DataFrame, validation: pd.DataFrame,
                  test: pd.DataFrame) -> pd.DataFrame:
    """Sizes and class shares per partition, to show the stratification held."""
    rows = []
    total = len(explore) + len(validation) + len(test)
    for name, part in zip(PARTITIONS, (explore, validation, test), strict=True):
        defect_share = float((part["label"] == DEFECT).mean())
        rows.append({
            "partition": name,
            "images": len(part),
            "share_%": round(100 * len(part) / total, 1),
            "defective_%": round(100 * defect_share, 1),
            "ok_%": round(100 * (1 - defect_share), 1),
        })
    return pd.DataFrame(rows)


def partition_of(explore: pd.DataFrame, validation: pd.DataFrame,
                 test: pd.DataFrame) -> pd.Series:
    """Which partition each filename landed in."""
    return pd.concat([
        pd.Series(name, index=part["filename"])
        for name, part in zip(PARTITIONS, (explore, validation, test), strict=True)
    ])


def class_balance(inventory: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares per class, for the EDA."""
    counts = inventory["label"].value_counts().sort_index()
    return pd.DataFrame({
        "label": counts.index.map(CLASS_NAMES),
        "images": counts.to_numpy(),
        "share_%": (100 * counts / counts.sum()).round(1).to_numpy(),
    }).reset_index(drop=True)
