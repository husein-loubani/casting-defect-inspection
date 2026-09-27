"""
How the classifier holds up when the photograph changes, and how to train it to.

The brief asks for robust classification, and the review asks how variation in
image quality and lighting is handled. Normalizing by the band's median cancels
a pure change of exposure in theory. This module measures what happens in
practice under rotation, gain, gamma and JPEG re-encoding, on the validation
split so the sealed test set stays opened once, and it builds the augmented
training copies that teach the model to ignore those changes.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage import transform

from casting_qa.config import (
    AUGMENT_COPIES,
    AUGMENT_GAMMA,
    AUGMENT_JPEG_QUALITY,
    AUGMENT_PROBABILITY,
    AUGMENT_ROTATION_DEGREES,
    DEFECT,
    N_JOBS,
    OK,
    PIXEL_MAX,
    RANDOM_SEED,
    ROBUSTNESS_PERTURBATIONS,
)
from casting_qa.features import attach_identifiers, describe


def perturb(image: np.ndarray, kind: str, value: float) -> np.ndarray:
    """
    One controlled change to a photograph.

    rotate: degrees about the frame center, edges filled with the nearest
    pixel. gain: every pixel multiplied, as a longer exposure would. gamma: a
    non-linear tone curve, which a median division cannot cancel. jpeg:
    re-encoded at the given quality, adding compression artifacts.
    """
    if kind == "rotate":
        rotated = transform.rotate(image, value, mode="edge", preserve_range=True)
        return np.clip(np.round(rotated), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "gain":
        return np.clip(np.round(image.astype(np.float64) * value), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "gamma":
        scaled = (image.astype(np.float64) / PIXEL_MAX) ** value
        return np.clip(np.round(scaled * PIXEL_MAX), 0, PIXEL_MAX).astype(np.uint8)
    if kind == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as reloaded:
            return np.asarray(reloaded.convert("L"), dtype=np.uint8)
    raise ValueError(f"unknown perturbation {kind!r}")


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    A random variant of a photograph for training.

    Each transform is applied independently with the configured probability,
    and at least one always is. Composing them every time would only ever show
    the model compression on top of resampling blur, so it would never learn
    what compression alone looks like.
    """
    while True:
        variant = image
        if rng.random() < AUGMENT_PROBABILITY:
            variant = perturb(variant, "rotate", float(rng.uniform(*AUGMENT_ROTATION_DEGREES)))
        if rng.random() < AUGMENT_PROBABILITY:
            variant = perturb(variant, "gamma", float(rng.uniform(*AUGMENT_GAMMA)))
        if rng.random() < AUGMENT_PROBABILITY:
            variant = perturb(variant, "jpeg", float(rng.integers(AUGMENT_JPEG_QUALITY[0],
                                                                   AUGMENT_JPEG_QUALITY[1] + 1)))
        if variant is not image:
            return variant


def _augmented_rows(image: np.ndarray, copies: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [describe(augment_image(image, rng)) for _ in range(copies)]


def augment_batch(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                  copies: int = AUGMENT_COPIES, seed: int = RANDOM_SEED,
                  n_jobs: int = N_JOBS) -> pd.DataFrame:
    """
    Features of `copies` augmented variants of every image, marked as augmented.

    Each variant keeps its source's filename and group, so group-aware
    cross-validation holds a photograph and its variants on the same side.
    """
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_augmented_rows)(loader(path), copies, seed + position)
        for position, path in enumerate(inventory["path"]))
    repeated = inventory.iloc[np.repeat(np.arange(len(inventory)), copies)].reset_index(drop=True)
    frame = attach_identifiers([row for image_rows in rows for row in image_rows], repeated)
    frame["augmented"] = True
    return frame


def _perturbed_rows(image: np.ndarray, perturbations) -> list[dict[str, float]]:
    return [describe(perturb(image, kind, value)) for kind, value in perturbations]


def perturbed_features(inventory: pd.DataFrame, loader: Callable[[str], np.ndarray],
                       perturbations=ROBUSTNESS_PERTURBATIONS,
                       n_jobs: int = N_JOBS) -> dict[str, pd.DataFrame]:
    """Features of every image under every perturbation, one table per perturbation."""
    per_image = Parallel(n_jobs=n_jobs)(
        delayed(_perturbed_rows)(loader(path), perturbations) for path in inventory["path"])
    return {f"{kind} {value:g}": attach_identifiers([rows[position] for rows in per_image], inventory)
            for position, (kind, value) in enumerate(perturbations)}


def robustness_table(clean: pd.DataFrame, perturbed: dict[str, pd.DataFrame],
                     models: dict, columns: list[str]) -> pd.DataFrame:
    """
    For each model and perturbation: accuracy, and decisions that flipped.

    A flip is a decision that changed because of the perturbation alone,
    compared with the same model on the unmodified image, whether the change
    was toward the truth or away from it.
    """
    labels = clean["label"].to_numpy()
    rows = []
    for name, model in models.items():
        baseline = model.predict(clean[columns])
        rows.append({"model": name, "perturbation": "none",
                     "accuracy": round(float((baseline == labels).mean()), 3), "flipped": 0,
                     "missed_defects": int(np.sum((labels == DEFECT) & (baseline == OK))),
                     "false_alarms": int(np.sum((labels == OK) & (baseline == DEFECT)))})
        for perturbation, frame in perturbed.items():
            predicted = model.predict(frame[columns])
            rows.append({
                "model": name,
                "perturbation": perturbation,
                "accuracy": round(float((predicted == labels).mean()), 3),
                "flipped": int((predicted != baseline).sum()),
                "missed_defects": int(np.sum((labels == DEFECT) & (predicted == OK))),
                "false_alarms": int(np.sum((labels == OK) & (predicted == DEFECT))),
            })
    return pd.DataFrame(rows)
